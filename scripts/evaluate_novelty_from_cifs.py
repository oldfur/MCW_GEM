#!/usr/bin/env python
"""Offline novelty evaluation for recursively collected generated CIF files."""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pymatgen.core.structure import Structure
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mp20.crystal import smact_validity, structure_validity  # noqa: E402
from mp20.novelty import (  # noqa: E402
    DEFAULT_MATCHER_PARAMS,
    DEFAULT_SKIP_STRUCTURE_REDUCTION,
    build_structure_matcher,
    is_structure_novel,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


@dataclass
class SampleFile:
    sample_index: int
    path: Path
    root: Path
    rel_path: str
    worker: str
    filename: str


@dataclass
class ReferenceInput:
    reference_index: int
    reference_id: str
    reference_path: str
    cif: str


@dataclass
class ReferenceRecord:
    reference_index: int
    reference_id: str
    reference_path: str
    formula: str
    reduced_formula: str
    num_sites: int
    structure: Structure


_REFERENCE_BUCKETS: Dict[str, List[ReferenceRecord]] = {}
_MATCHER_PARAMS: Dict[str, Any] = dict(DEFAULT_MATCHER_PARAMS)
_MATCHER = None
_SKIP_STRUCTURE_REDUCTION = DEFAULT_SKIP_STRUCTURE_REDUCTION


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def composition_key(structure: Structure) -> str:
    return structure.composition.reduced_formula


def structure_fields(structure: Structure) -> Dict[str, Any]:
    return {
        "formula": structure.composition.formula,
        "reduced_formula": structure.composition.reduced_formula,
        "num_sites": int(len(structure)),
    }


def normalized_composition_from_atomic_numbers(
    atomic_numbers: Iterable[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    counts_by_element = Counter(int(num) for num in atomic_numbers)
    if not counts_by_element:
        return tuple(), tuple()
    elems = tuple(sorted(counts_by_element))
    counts = np.array([counts_by_element[elem] for elem in elems], dtype=int)
    gcd = int(np.gcd.reduce(counts)) if counts.size else 1
    gcd = max(gcd, 1)
    return elems, tuple((counts // gcd).astype(int).tolist())


def existing_validity(structure: Structure) -> Tuple[bool, bool, bool, str]:
    elems, comps = normalized_composition_from_atomic_numbers(structure.atomic_numbers)
    if not elems:
        return False, False, False, "empty_structure"
    try:
        comp_valid = bool(smact_validity(elems, comps))
    except Exception as exc:
        return False, False, False, f"smact_validity failed: {exc}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            struct_valid = bool(structure_validity(structure))
    except Exception as exc:
        return bool(comp_valid), False, False, f"structure_validity failed: {exc}"
    valid = bool(comp_valid and struct_valid)
    if valid:
        return True, bool(comp_valid), bool(struct_valid), ""
    if not struct_valid:
        return False, bool(comp_valid), False, "structure invalid by mp20.crystal.structure_validity"
    return False, False, bool(struct_valid), "composition invalid by mp20.crystal.smact_validity"


def resolve_processed_dir(mp20_root: Path, processed_dir: Path) -> Tuple[Optional[Path], List[str]]:
    warnings_out: List[str] = []
    candidates = [processed_dir]
    if processed_dir.name == "precessed":
        candidates.append(processed_dir.with_name("processed"))
    candidates.extend([mp20_root / "precessed", mp20_root / "processed"])

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate, warnings_out

    warnings_out.append(
        "No MP20 processed/precessed directory was found; falling back to raw/all.csv where possible."
    )
    return None, warnings_out


def resolve_reference_csv(mp20_root: Path, processed_dir: Optional[Path]) -> Path:
    candidates = [
        mp20_root / "raw" / "all.csv",
        mp20_root / "all.csv",
    ]
    if processed_dir is not None:
        candidates.extend(
            [
                processed_dir / "all.csv",
                processed_dir.parent / "raw" / "all.csv",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find MP20 all.csv. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def collect_sample_files(sample_glob: str, sample_dir: str, max_samples: Optional[int]) -> Tuple[List[SampleFile], List[Path]]:
    roots: List[Path] = []
    if sample_dir:
        roots = [Path(sample_dir).expanduser()]
    else:
        matches = [Path(path).expanduser() for path in glob.glob(sample_glob)]
        if not matches and Path(sample_glob).exists():
            matches = [Path(sample_glob).expanduser()]
        roots = sorted(matches, key=lambda path: str(path))

    if not roots:
        raise FileNotFoundError(f"No sample roots matched {sample_glob!r}")

    records: List[SampleFile] = []
    seen = set()
    existing_roots: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        existing_roots.append(root)
        if root.is_file():
            paths = [root] if root.suffix.lower() == ".cif" else []
        else:
            paths = sorted(root.rglob("*.cif"), key=lambda path: str(path))

        for path in paths:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                rel_path = str(path.relative_to(root))
            except ValueError:
                rel_path = str(path)
            worker = next((part for part in path.parts if part.startswith("worker_")), "")
            records.append(
                SampleFile(
                    sample_index=len(records),
                    path=path,
                    root=root,
                    rel_path=rel_path,
                    worker=worker,
                    filename=path.name,
                )
            )

    if not existing_roots:
        raise FileNotFoundError(
            "Sample roots were matched but none exists: "
            + ", ".join(str(path) for path in roots)
        )
    if not records:
        raise FileNotFoundError(
            "No .cif files were found recursively under: "
            + ", ".join(str(path) for path in existing_roots)
        )

    if max_samples is not None:
        records = records[: max(0, max_samples)]
        records = [
            SampleFile(
                sample_index=idx,
                path=record.path,
                root=record.root,
                rel_path=record.rel_path,
                worker=record.worker,
                filename=record.filename,
            )
            for idx, record in enumerate(records)
        ]

    return records, existing_roots


def read_reference_inputs(
    csv_path: Path,
    reference_split: str,
    train_seed: int,
    num_train: int,
) -> Tuple[List[ReferenceInput], str, bool, List[str]]:
    warnings_out: List[str] = []
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "cif" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain a 'cif' column.")
        for row in reader:
            rows.append(row)

    selected_indices = list(range(len(rows)))
    reference_split_uncertain = True
    if reference_split in {"existing_eval", "all"}:
        split_name = "all.csv (matches mp20.analyze_test.CrystalGenerationEvaluator)"
        warnings_out.append(
            "Existing sampling novelty reads args.dataset_folder_path/all.csv directly; "
            "no explicit train split file is used by mp20.analyze_test."
        )
    elif reference_split == "train":
        rng = np.random.RandomState(train_seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        selected_indices = [int(idx) for idx in indices[:num_train]]
        split_name = (
            f"train split inferred from project defaults "
            f"(seed={train_seed}, num_train={num_train})"
        )
        reference_split_uncertain = False
    else:
        raise ValueError(f"Unsupported reference split: {reference_split}")

    inputs: List[ReferenceInput] = []
    for reference_index, row_index in enumerate(selected_indices):
        row = rows[row_index]
        reference_id = row.get("material_id") or row.get("mp_id") or str(row_index)
        inputs.append(
            ReferenceInput(
                reference_index=reference_index,
                reference_id=reference_id,
                reference_path=f"{csv_path}#{row_index}",
                cif=row["cif"],
            )
        )

    return inputs, split_name, reference_split_uncertain, warnings_out


def parse_reference_input(ref_input: ReferenceInput) -> Dict[str, Any]:
    try:
        structure = Structure.from_str(ref_input.cif, fmt="cif")
        fields = structure_fields(structure)
        return {
            "ok": True,
            "reference_index": ref_input.reference_index,
            "reference_id": ref_input.reference_id,
            "reference_path": ref_input.reference_path,
            "formula": fields["formula"],
            "reduced_formula": fields["reduced_formula"],
            "num_sites": fields["num_sites"],
            "structure": structure,
        }
    except Exception as exc:
        return {
            "ok": False,
            "reference_index": ref_input.reference_index,
            "reference_id": ref_input.reference_id,
            "reference_path": ref_input.reference_path,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def parse_sample_file(record: SampleFile, check_validity: bool = True) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_index": record.sample_index,
        "cif_path": str(record.path),
        "sample_root": str(record.root),
        "relative_path": record.rel_path,
        "worker": record.worker,
        "filename": record.filename,
        "parse_success": False,
        "evaluated": False,
        "novel": "",
        "matched_reference_index": "",
        "matched_reference_id": "",
        "matched_reference_path": "",
        "formula": "",
        "reduced_formula": "",
        "num_sites": "",
        "comp_valid": "",
        "struct_valid": "",
        "valid": "",
        "unique_group_index": "",
        "unique_group_size": "",
        "is_unique_representative": "",
        "error_message": "",
    }
    try:
        structure = Structure.from_file(str(record.path))
        fields = structure_fields(structure)
        if check_validity:
            valid, comp_valid, struct_valid, validity_error = existing_validity(structure)
        else:
            valid, comp_valid, struct_valid, validity_error = "", "", "", ""
        row.update(
            {
                "parse_success": True,
                "formula": fields["formula"],
                "reduced_formula": fields["reduced_formula"],
                "num_sites": fields["num_sites"],
                "comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": valid,
                "error_message": validity_error,
                "structure": structure,
            }
        )
    except Exception as exc:
        row["error_message"] = str(exc)
        row["traceback"] = traceback.format_exc(limit=5)
    return row


def load_reference_records(
    inputs: List[ReferenceInput],
    num_workers: int,
) -> Tuple[List[ReferenceRecord], List[Dict[str, Any]]]:
    records: List[ReferenceRecord] = []
    failures: List[Dict[str, Any]] = []

    if num_workers <= 1:
        iterator = (parse_reference_input(item) for item in inputs)
        results = tqdm(iterator, total=len(inputs), desc="Load reference CIFs")
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = [executor.submit(parse_reference_input, item) for item in inputs]
        results = (
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures), desc="Load reference CIFs")
        )

    try:
        for result in results:
            if result["ok"]:
                records.append(
                    ReferenceRecord(
                        reference_index=int(result["reference_index"]),
                        reference_id=str(result["reference_id"]),
                        reference_path=str(result["reference_path"]),
                        formula=str(result["formula"]),
                        reduced_formula=str(result["reduced_formula"]),
                        num_sites=int(result["num_sites"]),
                        structure=result["structure"],
                    )
                )
            else:
                failure = dict(result)
                failure["kind"] = "reference_parse_failure"
                failures.append(failure)
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)

    records.sort(key=lambda record: record.reference_index)
    return records, failures


def load_sample_rows(
    samples: List[SampleFile],
    num_workers: int,
    check_validity: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if num_workers <= 1:
        iterator = (parse_sample_file(sample, check_validity) for sample in samples)
        results = tqdm(iterator, total=len(samples), desc="Parse sample CIFs")
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = [
            executor.submit(parse_sample_file, sample, check_validity)
            for sample in samples
        ]
        results = (
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parse sample CIFs")
        )

    try:
        rows.extend(results)
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)

    rows.sort(key=lambda row: int(row["sample_index"]))
    return rows


def bucket_reference_records(records: List[ReferenceRecord]) -> Dict[str, List[ReferenceRecord]]:
    buckets: Dict[str, List[ReferenceRecord]] = defaultdict(list)
    for record in records:
        buckets[record.reduced_formula].append(record)
    return dict(buckets)


def clean_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"structure", "traceback"}}


def init_match_worker(matcher_params: Dict[str, Any], skip_structure_reduction: bool) -> None:
    global _MATCHER_PARAMS, _MATCHER, _SKIP_STRUCTURE_REDUCTION
    _MATCHER_PARAMS = dict(matcher_params)
    _MATCHER = build_structure_matcher(**_MATCHER_PARAMS)
    _SKIP_STRUCTURE_REDUCTION = skip_structure_reduction


def match_group_task(task: Tuple[int, int, Structure]) -> Dict[str, Any]:
    global _MATCHER
    group_index, representative_sample_index, structure = task
    if _MATCHER is None:
        _MATCHER = build_structure_matcher(**_MATCHER_PARAMS)
    bucket_key = composition_key(structure)
    references = _REFERENCE_BUCKETS.get(bucket_key, [])
    try:
        for bucket_index, reference in enumerate(references):
            if _MATCHER.fit(
                structure,
                reference.structure,
                skip_structure_reduction=_SKIP_STRUCTURE_REDUCTION,
            ):
                return {
                    "group_index": group_index,
                    "representative_sample_index": representative_sample_index,
                    "novel": False,
                    "matched_reference_index": reference.reference_index,
                    "matched_reference_id": reference.reference_id,
                    "matched_reference_path": reference.reference_path,
                    "matched_reference_bucket_index": bucket_index,
                    "reference_bucket_size": len(references),
                    "error_message": "",
                }
        return {
            "group_index": group_index,
            "representative_sample_index": representative_sample_index,
            "novel": True,
            "matched_reference_index": "",
            "matched_reference_id": "",
            "matched_reference_path": "",
            "matched_reference_bucket_index": "",
            "reference_bucket_size": len(references),
            "error_message": "",
        }
    except Exception as exc:
        return {
            "group_index": group_index,
            "representative_sample_index": representative_sample_index,
            "novel": "",
            "matched_reference_index": "",
            "matched_reference_id": "",
            "matched_reference_path": "",
            "matched_reference_bucket_index": "",
            "reference_bucket_size": len(references),
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def group_candidate_structures(
    sample_rows: List[Dict[str, Any]],
    valid_only: bool,
    matcher_params: Dict[str, Any],
) -> Tuple[List[Tuple[int, int, Structure]], List[Dict[str, Any]], Optional[str]]:
    candidate_indices = [
        idx
        for idx, row in enumerate(sample_rows)
        if row.get("parse_success") and (not valid_only or row.get("valid"))
    ]
    if not candidate_indices:
        return [], [], None

    structures = [sample_rows[idx]["structure"] for idx in candidate_indices]
    matcher = build_structure_matcher(**matcher_params)
    fallback_error = None
    try:
        groups = matcher.group_structures(structures)
    except Exception as exc:
        fallback_error = f"group_structures failed; evaluating each candidate separately: {exc}"
        groups = [[structure] for structure in structures]

    id_to_candidate_idx = {id(sample_rows[idx]["structure"]): idx for idx in candidate_indices}
    tasks: List[Tuple[int, int, Structure]] = []
    group_members: List[Dict[str, Any]] = []

    for group_index, group in enumerate(groups):
        member_row_indices = [
            id_to_candidate_idx[id(structure)]
            for structure in group
            if id(structure) in id_to_candidate_idx
        ]
        if not member_row_indices:
            continue
        representative_row_idx = member_row_indices[0]
        representative_sample_index = int(sample_rows[representative_row_idx]["sample_index"])
        group_size = len(member_row_indices)
        for row_idx in member_row_indices:
            sample_rows[row_idx]["evaluated"] = True
            sample_rows[row_idx]["unique_group_index"] = group_index
            sample_rows[row_idx]["unique_group_size"] = group_size
            sample_rows[row_idx]["is_unique_representative"] = row_idx == representative_row_idx
        tasks.append((group_index, representative_sample_index, sample_rows[representative_row_idx]["structure"]))
        group_members.append(
            {
                "group_index": group_index,
                "representative_sample_index": representative_sample_index,
                "member_sample_indices": [
                    int(sample_rows[row_idx]["sample_index"]) for row_idx in member_row_indices
                ],
            }
        )

    return tasks, group_members, fallback_error


def evaluate_group_novelty(
    tasks: List[Tuple[int, int, Structure]],
    reference_buckets: Dict[str, List[ReferenceRecord]],
    matcher_params: Dict[str, Any],
    skip_structure_reduction: bool,
    num_workers: int,
) -> List[Dict[str, Any]]:
    global _REFERENCE_BUCKETS, _MATCHER_PARAMS, _SKIP_STRUCTURE_REDUCTION, _MATCHER
    _REFERENCE_BUCKETS = reference_buckets
    _MATCHER_PARAMS = dict(matcher_params)
    _SKIP_STRUCTURE_REDUCTION = skip_structure_reduction
    _MATCHER = build_structure_matcher(**matcher_params)

    if not tasks:
        return []

    if num_workers <= 1:
        return [
            match_group_task(task)
            for task in tqdm(tasks, total=len(tasks), desc="Novelty")
        ]

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None

    if ctx is None:
        print("Warning: multiprocessing fork context unavailable; falling back to sequential novelty.")
        return [
            match_group_task(task)
            for task in tqdm(tasks, total=len(tasks), desc="Novelty")
        ]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_match_worker,
        initargs=(matcher_params, skip_structure_reduction),
    ) as executor:
        futures = [executor.submit(match_group_task, task) for task in tasks]
        return [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures), desc="Novelty")
        ]


def apply_group_results(
    sample_rows: List[Dict[str, Any]],
    group_members: List[Dict[str, Any]],
    group_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    members_by_group = {
        int(group["group_index"]): group["member_sample_indices"] for group in group_members
    }
    rows_by_sample_index = {int(row["sample_index"]): row for row in sample_rows}
    failures: List[Dict[str, Any]] = []
    matched_pairs: List[Dict[str, Any]] = []

    for result in sorted(group_results, key=lambda item: int(item["group_index"])):
        group_index = int(result["group_index"])
        member_indices = members_by_group.get(group_index, [])
        if result.get("error_message"):
            failure = dict(result)
            failure["kind"] = "matcher_failure"
            failures.append(failure)
        for sample_index in member_indices:
            row = rows_by_sample_index[sample_index]
            row["novel"] = result.get("novel", "")
            row["matched_reference_index"] = result.get("matched_reference_index", "")
            row["matched_reference_id"] = result.get("matched_reference_id", "")
            row["matched_reference_path"] = result.get("matched_reference_path", "")
            if result.get("error_message"):
                row["evaluated"] = False
                row["error_message"] = result["error_message"]
        if result.get("novel") is False:
            matched_pairs.append(
                {
                    "group_index": group_index,
                    "representative_sample_index": result["representative_sample_index"],
                    "member_sample_indices": member_indices,
                    "matched_reference_index": result["matched_reference_index"],
                    "matched_reference_id": result["matched_reference_id"],
                    "matched_reference_path": result["matched_reference_path"],
                }
            )

    return failures, matched_pairs


def run_self_check(
    tasks: List[Tuple[int, int, Structure]],
    reference_records: List[ReferenceRecord],
    reference_buckets: Dict[str, List[ReferenceRecord]],
    matcher_params: Dict[str, Any],
    sample_count: int,
) -> Dict[str, Any]:
    if not tasks:
        return {"status": "skipped", "reason": "No generated structures to check."}

    try:
        from mp20.analyze_test import CrystalGenerationEvaluator
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Could not import old evaluator: {exc}",
        }

    selected = random.Random(0).sample(tasks, min(sample_count, len(tasks)))
    global _REFERENCE_BUCKETS, _MATCHER_PARAMS, _MATCHER, _SKIP_STRUCTURE_REDUCTION
    _REFERENCE_BUCKETS = reference_buckets
    _MATCHER_PARAMS = dict(matcher_params)
    _MATCHER = build_structure_matcher(**matcher_params)
    _SKIP_STRUCTURE_REDUCTION = DEFAULT_SKIP_STRUCTURE_REDUCTION

    old_evaluator = CrystalGenerationEvaluator(
        dataset_cif_list=[],
        compute_novelty=True,
        **matcher_params,
    )
    old_evaluator.dataset_struct_list = [record.structure for record in reference_records]
    mismatches = []
    for task in tqdm(selected, total=len(selected), desc="Self-check"):
        group_index, representative_sample_index, structure = task
        new_result = match_group_task(task)
        new_novel = new_result.get("novel")
        old_novel = bool(old_evaluator._get_novelty(structure))
        if new_novel != old_novel:
            mismatches.append(
                {
                    "group_index": group_index,
                    "representative_sample_index": representative_sample_index,
                    "old_novel": old_novel,
                    "new_novel": new_novel,
                    "new_error": new_result.get("error_message", ""),
                }
            )

    if mismatches:
        return {"status": "failed", "mismatches": mismatches}
    return {"status": "passed", "checked_count": len(selected)}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=json_default) + "\n")


def write_per_sample_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "cif_path",
        "sample_root",
        "relative_path",
        "worker",
        "filename",
        "parse_success",
        "evaluated",
        "novel",
        "matched_reference_index",
        "matched_reference_id",
        "matched_reference_path",
        "formula",
        "reduced_formula",
        "num_sites",
        "comp_valid",
        "struct_valid",
        "valid",
        "unique_group_index",
        "unique_group_size",
        "is_unique_representative",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_row(row).get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated CIF novelty against the MP20 reference CIFs."
    )
    parser.add_argument("--sample-glob", default="./outputs/sample_LF_mp20_2026*")
    parser.add_argument("--sample-dir", default="")
    parser.add_argument("--mp20-root", default="./mp20")
    parser.add_argument("--processed-dir", default="./mp20/precessed")
    parser.add_argument("--output-dir", default="./outputs/novelty_eval")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--valid-only", type=parse_bool, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--matcher-preset", default="existing", choices=["existing"])
    parser.add_argument("--save-per-sample", type=parse_bool, default=True)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-samples", type=int, default=5)
    parser.add_argument(
        "--reference-split",
        default="existing_eval",
        choices=["existing_eval", "all", "train"],
        help=(
            "existing_eval/all matches the current sampling evaluator by using raw/all.csv. "
            "train infers the seeded project train split."
        ),
    )
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--num-train", type=int, default=27138)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    num_workers = max(1, int(args.num_workers))
    mp20_root = Path(args.mp20_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    matcher_params = dict(DEFAULT_MATCHER_PARAMS)
    skip_structure_reduction = DEFAULT_SKIP_STRUCTURE_REDUCTION
    all_warnings: List[str] = []

    processed_dir, processed_warnings = resolve_processed_dir(
        mp20_root, Path(args.processed_dir).expanduser()
    )
    all_warnings.extend(processed_warnings)
    reference_csv = resolve_reference_csv(mp20_root, processed_dir)

    samples, sample_roots = collect_sample_files(
        args.sample_glob,
        args.sample_dir,
        args.max_samples,
    )
    print(f"Found {len(samples)} CIF files under {len(sample_roots)} sample root(s).")
    print(f"Using MP20 reference CSV: {reference_csv}")

    reference_inputs, reference_split, reference_split_uncertain, split_warnings = read_reference_inputs(
        reference_csv,
        args.reference_split,
        args.train_seed,
        args.num_train,
    )
    all_warnings.extend(split_warnings)
    for warning in all_warnings:
        print(f"Warning: {warning}")

    reference_records, reference_failures = load_reference_records(reference_inputs, num_workers)
    if not reference_records:
        raise RuntimeError("No reference structures could be parsed.")
    reference_buckets = bucket_reference_records(reference_records)
    print(
        f"Loaded {len(reference_records)} reference structures in "
        f"{len(reference_buckets)} reduced-formula buckets."
    )

    check_validity = bool(args.valid_only)
    if not check_validity:
        print(
            "Skipping existing validity checks because the novelty denominator "
            "does not require valid-only filtering."
        )

    sample_rows = load_sample_rows(samples, num_workers, check_validity)
    parse_failures = [
        {
            "kind": "sample_parse_failure",
            **clean_csv_row(row),
            "traceback": row.get("traceback", ""),
        }
        for row in sample_rows
        if not row.get("parse_success")
    ]

    tasks, group_members, group_error = group_candidate_structures(
        sample_rows,
        bool(args.valid_only),
        matcher_params,
    )
    if group_error:
        all_warnings.append(group_error)
        print(f"Warning: {group_error}")

    self_check_result = None
    if args.self_check:
        self_check_result = run_self_check(
            tasks,
            reference_records,
            reference_buckets,
            matcher_params,
            max(1, int(args.self_check_samples)),
        )
        print("Self-check:", self_check_result)
        if self_check_result.get("status") == "failed":
            write_json(output_dir / "self_check_failure.json", self_check_result)
            return 2

    group_results = evaluate_group_novelty(
        tasks,
        reference_buckets,
        matcher_params,
        skip_structure_reduction,
        num_workers,
    )
    matcher_failures, matched_pairs = apply_group_results(sample_rows, group_members, group_results)

    successful_group_results = [result for result in group_results if not result.get("error_message")]
    novel_count = sum(1 for result in successful_group_results if result.get("novel") is True)
    non_novel_count = sum(1 for result in successful_group_results if result.get("novel") is False)
    evaluated_count = len(successful_group_results)
    novelty_rate = novel_count / evaluated_count if evaluated_count else None

    total_parsed = sum(1 for row in sample_rows if row.get("parse_success"))
    valid_count = (
        sum(1 for row in sample_rows if row.get("parse_success") and row.get("valid"))
        if check_validity
        else None
    )
    candidate_sample_count = sum(
        1
        for row in sample_rows
        if row.get("parse_success") and (not args.valid_only or row.get("valid"))
    )

    denominator_definition = (
        "unique valid generated structures after StructureMatcher.group_structures"
        if args.valid_only
        else "unique successfully parsed generated structures after StructureMatcher.group_structures"
    )

    summary = {
        "total_cif_files_found": len(samples),
        "total_cif_files_parsed": total_parsed,
        "parse_failures": len(parse_failures),
        "valid_count": valid_count,
        "candidate_sample_count": candidate_sample_count,
        "unique_group_count": len(tasks),
        "evaluated_count": evaluated_count,
        "matcher_failures": len(matcher_failures),
        "novel_count": novel_count,
        "non_novel_count": non_novel_count,
        "novelty_rate": novelty_rate,
        "denominator_definition": denominator_definition,
        "valid_only": bool(args.valid_only),
        "validity_checked": check_validity,
        "reference_structure_count": len(reference_records),
        "reference_parse_failures": len(reference_failures),
        "reference_split": reference_split,
        "reference_split_uncertain": reference_split_uncertain,
        "reference_source": str(reference_csv),
        "processed_dir": str(processed_dir) if processed_dir else "",
        "matcher_parameters": matcher_params,
        "matcher_fit_kwargs": {
            "skip_structure_reduction": skip_structure_reduction,
        },
        "matcher_preset": args.matcher_preset,
        "sample_roots": [str(root) for root in sample_roots],
        "timestamp": now_iso(),
        "runtime_seconds": time.time() - start,
        "warnings": all_warnings,
        "self_check": self_check_result,
        "existing_novelty_audit": {
            "implementation": "mp20.analyze_test.CrystalGenerationEvaluator",
            "reference_loader": "pd.read_csv(args.dataset_folder_path/all.csv)['cif'].tolist()",
            "reference_used_by_existing_evaluator": "all.csv, not an explicit train split",
            "matcher": "StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)",
            "novelty_fit_kwargs": {"skip_structure_reduction": True},
            "validity_filter": "valid generated crystals only",
            "uniqueness": "matcher.group_structures(valid_structs)",
            "novelty_denominator": "one representative per unique valid generated group",
            "primitive_or_reduced_before_matching": "no explicit primitive/reduced conversion; fit uses skip_structure_reduction=True",
            "oxidation_states_ignored": False,
            "reference_composition_bucket": "existing evaluator does not bucket; this script buckets by reduced_formula before applying the same matcher",
        },
    }

    write_json(output_dir / "novelty_summary.json", summary)
    if args.save_per_sample:
        write_per_sample_csv(output_dir / "novelty_per_sample.csv", sample_rows)
    failures = reference_failures + parse_failures + matcher_failures
    write_jsonl(output_dir / "novelty_failures.jsonl", failures)
    write_jsonl(output_dir / "matched_pairs.jsonl", matched_pairs)

    print(f"Novelty rate: {novelty_rate} ({novel_count}/{evaluated_count})")
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
