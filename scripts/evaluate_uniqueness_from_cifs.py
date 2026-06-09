#!/usr/bin/env python
"""Offline uniqueness evaluation for recursively collected generated CIF files."""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import random
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pymatgen.core.structure import Structure
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mp20.crystal import array_dict_to_crystal  # noqa: E402
from mp20.novelty import (  # noqa: E402
    DEFAULT_MATCHER_PARAMS,
    build_structure_matcher,
    compute_uniqueness_rate,
    group_structures_for_uniqueness,
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


class _OldEvaluatorCrystal:
    def __init__(self, structure: Structure):
        self.structure = structure
        self.valid = True
        self.comp_valid = True
        self.struct_valid = True


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


def structure_fields(structure: Structure) -> Dict[str, Any]:
    return {
        "formula": structure.composition.formula,
        "reduced_formula": structure.composition.reduced_formula,
        "num_sites": int(len(structure)),
    }


def structure_to_array_dict(structure: Structure, sample_index: int) -> Dict[str, np.ndarray]:
    return {
        "frac_coords": np.array(structure.frac_coords, dtype=float),
        "atom_types": np.array(structure.atomic_numbers, dtype=int),
        "lengths": np.array(structure.lattice.abc, dtype=float),
        "angles": np.array(structure.lattice.angles, dtype=float),
        "sample_idx": int(sample_index),
    }


def existing_validity_from_structure(
    structure: Structure,
    sample_index: int,
) -> Tuple[bool, bool, bool, str]:
    array_dict = structure_to_array_dict(structure, sample_index)
    angles = array_dict["angles"]
    angles_in_existing_range = bool(np.all(50 <= angles) and np.all(angles <= 130))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            crystal = array_dict_to_crystal(array_dict, save=False)
    except Exception as exc:
        return False, False, False, f"array_dict_to_crystal failed: {exc}"

    valid = bool(getattr(crystal, "valid", False))
    comp_valid = bool(getattr(crystal, "comp_valid", False))
    struct_valid = bool(getattr(crystal, "struct_valid", False))
    if valid:
        return True, comp_valid, struct_valid, ""

    invalid_reason = getattr(crystal, "invalid_reason", "") or ""
    if not angles_in_existing_range:
        invalid_reason = "angles outside mp20.crystal.array_dict_to_crystal 50-130 degree gate"
    elif comp_valid and struct_valid:
        invalid_reason = (
            invalid_reason
            or "valid false after mp20.crystal.Crystal.get_fingerprints"
        )
    elif not struct_valid:
        invalid_reason = invalid_reason or "structure invalid by mp20.crystal.structure_validity"
    elif not comp_valid:
        invalid_reason = invalid_reason or "composition invalid by mp20.crystal.smact_validity"
    else:
        invalid_reason = invalid_reason or "invalid by mp20.crystal.array_dict_to_crystal"
    return False, comp_valid, struct_valid, invalid_reason


def collect_sample_files(
    sample_glob: str,
    sample_dir: str,
    max_samples: Optional[int],
) -> Tuple[List[SampleFile], List[Path]]:
    roots: List[Path]
    if sample_dir:
        roots = [Path(sample_dir).expanduser()]
    else:
        matches = [Path(path).expanduser() for path in glob.glob(sample_glob)]
        if not matches and Path(sample_glob).expanduser().exists():
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
        "unique_representative": "",
        "cluster_id": "",
        "matched_representative_index": "",
        "matched_representative_path": "",
        "formula": "",
        "reduced_formula": "",
        "num_sites": "",
        "comp_valid": "",
        "struct_valid": "",
        "valid": "",
        "error_message": "",
    }
    try:
        structure = Structure.from_file(str(record.path))
        fields = structure_fields(structure)
        if check_validity:
            valid, comp_valid, struct_valid, validity_error = existing_validity_from_structure(
                structure,
                record.sample_index,
            )
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


def should_evaluate(row: Dict[str, Any], valid_only: bool, include_invalid_cifs: bool) -> bool:
    if not row.get("parse_success"):
        return False
    if include_invalid_cifs:
        return True
    if valid_only:
        return bool(row.get("valid"))
    return True


def candidate_row_indices(
    sample_rows: List[Dict[str, Any]],
    valid_only: bool,
    include_invalid_cifs: bool,
) -> List[int]:
    return [
        idx
        for idx, row in enumerate(sample_rows)
        if should_evaluate(row, valid_only, include_invalid_cifs)
    ]


def cluster_candidate_structures(
    sample_rows: List[Dict[str, Any]],
    valid_only: bool,
    include_invalid_cifs: bool,
    matcher_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_indices = candidate_row_indices(sample_rows, valid_only, include_invalid_cifs)
    if not candidate_indices:
        return [], []

    structures = [sample_rows[idx]["structure"] for idx in candidate_indices]
    matcher = build_structure_matcher(**matcher_params)
    try:
        groups = group_structures_for_uniqueness(structures, matcher=matcher)
    except Exception as exc:
        failure = {
            "kind": "matcher_failure",
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=5),
            "candidate_count": len(candidate_indices),
        }
        return [], [failure]

    row_indices_by_structure_id: Dict[int, List[int]] = {}
    for position, row_idx in enumerate(candidate_indices):
        row_indices_by_structure_id.setdefault(id(structures[position]), []).append(row_idx)

    clusters: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for cluster_id, group in enumerate(groups):
        member_row_indices: List[int] = []
        for structure in group:
            row_idx_queue = row_indices_by_structure_id.get(id(structure), [])
            if row_idx_queue:
                member_row_indices.append(row_idx_queue.pop(0))

        if len(member_row_indices) != len(group):
            failures.append(
                {
                    "kind": "matcher_mapping_failure",
                    "cluster_id": cluster_id,
                    "expected_group_size": len(group),
                    "mapped_group_size": len(member_row_indices),
                    "error_message": (
                        "Could not map every StructureMatcher group member back to "
                        "its source CIF row by object identity."
                    ),
                }
            )
        if not member_row_indices:
            continue

        representative_row_idx = member_row_indices[0]
        representative = sample_rows[representative_row_idx]
        member_indices = [int(sample_rows[idx]["sample_index"]) for idx in member_row_indices]
        member_paths = [sample_rows[idx]["cif_path"] for idx in member_row_indices]

        for row_idx in member_row_indices:
            row = sample_rows[row_idx]
            row["evaluated"] = True
            row["unique_representative"] = row_idx == representative_row_idx
            row["cluster_id"] = cluster_id
            row["matched_representative_index"] = int(representative["sample_index"])
            row["matched_representative_path"] = representative["cif_path"]

        clusters.append(
            {
                "cluster_id": cluster_id,
                "representative_index": int(representative["sample_index"]),
                "representative_path": representative["cif_path"],
                "representative_formula": representative.get("formula", ""),
                "cluster_size": len(member_row_indices),
                "member_indices": member_indices,
                "member_paths": member_paths,
            }
        )

    return clusters, failures


def run_self_check(
    sample_rows: List[Dict[str, Any]],
    valid_only: bool,
    include_invalid_cifs: bool,
    matcher_params: Dict[str, Any],
    sample_count: int,
) -> Dict[str, Any]:
    indices = candidate_row_indices(sample_rows, valid_only, include_invalid_cifs)
    if not indices:
        return {"status": "skipped", "reason": "No generated structures to check."}

    try:
        from mp20.analyze_test import CrystalGenerationEvaluator
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Could not import old evaluator: {exc}",
        }

    selected_indices = random.Random(0).sample(indices, min(sample_count, len(indices)))
    structures = [sample_rows[idx]["structure"] for idx in selected_indices]
    try:
        new_rate, new_groups = compute_uniqueness_rate(structures, **matcher_params)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"New uniqueness helper failed: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }

    try:
        evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=[],
            compute_novelty=False,
            device="cpu",
            **matcher_params,
        )
        evaluator.pred_arrays_list = [{} for _ in structures]
        evaluator.pred_crys_list = [_OldEvaluatorCrystal(structure) for structure in structures]
        evaluator._arrays_to_crystals = lambda save=False, save_dir="": None
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluator.get_metrics(save=False)
        old_rate = float(metrics["unique_rate"].detach().cpu().item())
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Could not directly call old get_metrics path: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }

    if abs(old_rate - new_rate) > 1e-12:
        return {
            "status": "failed",
            "checked_count": len(structures),
            "old_rate": old_rate,
            "new_rate": new_rate,
            "old_unique_count_inferred": int(round(old_rate * len(structures))),
            "new_unique_count": len(new_groups),
            "sample_indices": [int(sample_rows[idx]["sample_index"]) for idx in selected_indices],
        }
    return {
        "status": "passed",
        "checked_count": len(structures),
        "old_rate": old_rate,
        "new_rate": new_rate,
        "unique_count": len(new_groups),
    }


def clean_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"structure", "traceback"}}


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
        "unique_representative",
        "cluster_id",
        "matched_representative_index",
        "matched_representative_path",
        "formula",
        "reduced_formula",
        "num_sites",
        "comp_valid",
        "struct_valid",
        "valid",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_row(row).get(key, "") for key in fieldnames})


def matcher_internal_options(matcher_params: Dict[str, Any]) -> Dict[str, Any]:
    matcher = build_structure_matcher(**matcher_params)
    return {
        "primitive_cell": getattr(matcher, "_primitive_cell", None),
        "scale": getattr(matcher, "_scale", None),
        "attempt_supercell": getattr(matcher, "_attempt_supercell", None),
        "allow_subset": getattr(matcher, "_allow_subset", None),
        "supercell_size": getattr(matcher, "_supercell_size", None),
        "comparator": getattr(getattr(matcher, "_comparator", None), "__class__", type(None)).__name__,
    }


def denominator_definition(valid_only: bool, include_invalid_cifs: bool) -> str:
    if include_invalid_cifs:
        return (
            "successfully parsed generated CIFs, including CIFs that fail the "
            "existing mp20.crystal validity checks"
        )
    if valid_only:
        return (
            "valid generated CIFs only: parse_success and "
            "mp20.crystal.array_dict_to_crystal(...).valid"
        )
    return "all successfully parsed generated CIFs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated CIF uniqueness within the generated sample set."
    )
    parser.add_argument("--sample-glob", default="./outputs/sample_LF_mp20_2026*")
    parser.add_argument("--sample-dir", default="")
    parser.add_argument("--output-dir", default="./outputs/uniqueness_eval")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--valid-only", type=parse_bool, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--matcher-preset", default="existing", choices=["existing"])
    parser.add_argument("--save-per-sample", type=parse_bool, default=True)
    parser.add_argument("--include-invalid-cifs", type=parse_bool, default=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-samples", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    num_workers = max(1, int(args.num_workers))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    matcher_params = dict(DEFAULT_MATCHER_PARAMS)
    warnings_out: List[str] = []

    if bool(args.include_invalid_cifs) and bool(args.valid_only):
        warnings_out.append(
            "--include-invalid-cifs=True overrides --valid-only=True for the "
            "uniqueness denominator."
        )

    samples, sample_roots = collect_sample_files(
        args.sample_glob,
        args.sample_dir,
        args.max_samples,
    )
    print(f"Found {len(samples)} CIF files under {len(sample_roots)} sample root(s).")
    print(
        "Using existing uniqueness matcher: "
        f"StructureMatcher(stol={matcher_params['stol']}, "
        f"angle_tol={matcher_params['angle_tol']}, ltol={matcher_params['ltol']})"
    )

    check_validity = bool(args.valid_only) and not bool(args.include_invalid_cifs)
    if not check_validity:
        print(
            "Skipping existing validity checks because the uniqueness denominator "
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
    validity_failures = [
        {
            "kind": "validity_check_failure",
            **clean_csv_row(row),
        }
        for row in sample_rows
        if check_validity and row.get("parse_success") and not row.get("valid")
    ]

    self_check_result = None
    if args.self_check:
        self_check_result = run_self_check(
            sample_rows,
            bool(args.valid_only),
            bool(args.include_invalid_cifs),
            matcher_params,
            max(1, int(args.self_check_samples)),
        )
        print("Self-check:", self_check_result)
        if self_check_result.get("status") == "failed":
            write_json(output_dir / "self_check_failure.json", self_check_result)
            return 2

    candidate_count = len(
        candidate_row_indices(
            sample_rows,
            bool(args.valid_only),
            bool(args.include_invalid_cifs),
        )
    )
    print(f"Clustering {candidate_count} generated structure(s) for uniqueness.")
    clusters, matcher_failures = cluster_candidate_structures(
        sample_rows,
        bool(args.valid_only),
        bool(args.include_invalid_cifs),
        matcher_params,
    )

    evaluated_count = sum(1 for row in sample_rows if row.get("evaluated"))
    unique_count = len(clusters)
    duplicate_count = max(evaluated_count - unique_count, 0)
    uniqueness_rate = unique_count / evaluated_count if evaluated_count else 0.0
    cluster_sizes = [int(cluster["cluster_size"]) for cluster in clusters]

    total_parsed = sum(1 for row in sample_rows if row.get("parse_success"))
    valid_count = (
        sum(1 for row in sample_rows if row.get("parse_success") and row.get("valid"))
        if check_validity
        else None
    )
    comp_valid_count = (
        sum(1 for row in sample_rows if row.get("parse_success") and row.get("comp_valid"))
        if check_validity
        else None
    )
    struct_valid_count = (
        sum(1 for row in sample_rows if row.get("parse_success") and row.get("struct_valid"))
        if check_validity
        else None
    )

    if matcher_failures:
        warnings_out.append("One or more StructureMatcher failures prevented full clustering.")

    summary = {
        "total_cif_files_found": len(samples),
        "total_cif_files_parsed": total_parsed,
        "parse_failures": len(parse_failures),
        "valid_count": valid_count,
        "invalid_count": total_parsed - valid_count if check_validity else None,
        "comp_valid_count": comp_valid_count,
        "struct_valid_count": struct_valid_count,
        "evaluated_count": evaluated_count,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "uniqueness_rate": uniqueness_rate,
        "denominator_definition": denominator_definition(
            bool(args.valid_only),
            bool(args.include_invalid_cifs),
        ),
        "valid_only": bool(args.valid_only),
        "include_invalid_cifs": bool(args.include_invalid_cifs),
        "validity_checked": check_validity,
        "valid_filter_uncertain": False,
        "validity_definition": (
            "Parsed CIFs are converted back through "
            "mp20.crystal.array_dict_to_crystal(...).valid, matching the "
            "CrystalGenerationEvaluator valid filter as closely as possible "
            "from saved CIFs."
            if check_validity
            else "Validity checks were skipped because valid-only filtering was disabled."
        ),
        "matcher_parameters": matcher_params,
        "matcher_internal_options": matcher_internal_options(matcher_params),
        "matcher_preset": args.matcher_preset,
        "sample_roots": [str(root) for root in sample_roots],
        "timestamp": now_iso(),
        "runtime_seconds": time.time() - start,
        "num_clusters": len(clusters),
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "mean_cluster_size": float(mean(cluster_sizes)) if cluster_sizes else 0.0,
        "median_cluster_size": float(median(cluster_sizes)) if cluster_sizes else 0.0,
        "matcher_failures": len(matcher_failures),
        "save_per_sample": bool(args.save_per_sample),
        "warnings": warnings_out,
        "self_check": self_check_result,
        "existing_uniqueness_audit": {
            "implementation": "mp20.analyze_test.CrystalGenerationEvaluator.get_metrics",
            "matcher": "StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)",
            "uniqueness_code": (
                "valid_structs = [c.structure for c in pred_crys_list if c.valid]; "
                "unique_struct_groups = matcher.group_structures(valid_structs); "
                "unique_rate = len(unique_struct_groups) / len(valid_structs)"
            ),
            "denominator": "valid generated Crystal objects only",
            "validity_filter": (
                "c.valid from mp20.crystal.Crystal, including composition validity, "
                "structure distance/volume validity, and fingerprint construction"
            ),
            "primitive_or_reduced_before_matching": (
                "no explicit primitive/reduced conversion before group_structures; "
                "StructureMatcher internal defaults apply"
            ),
            "oxidation_states_ignored": (
                "no explicit oxidation-state stripping; generated structures are "
                "constructed from atomic numbers/elements"
            ),
            "composition_bucket": (
                "no explicit formula, reduced_formula, or anonymous_formula bucketing "
                "in the existing uniqueness evaluator"
            ),
            "novelty_distinction": (
                "uniqueness is generated-vs-generated; novelty is generated-vs-reference "
                "and is not computed by this script"
            ),
        },
    }

    write_json(output_dir / "uniqueness_summary.json", summary)
    if args.save_per_sample:
        write_per_sample_csv(output_dir / "uniqueness_per_sample.csv", sample_rows)
    write_jsonl(output_dir / "uniqueness_clusters.jsonl", clusters)
    write_jsonl(
        output_dir / "uniqueness_failures.jsonl",
        parse_failures + validity_failures + matcher_failures,
    )

    print(f"Uniqueness rate: {uniqueness_rate} ({unique_count}/{evaluated_count})")
    print(f"Wrote results to {output_dir}")
    return 1 if matcher_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
