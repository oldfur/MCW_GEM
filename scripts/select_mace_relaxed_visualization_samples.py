#!/usr/bin/env python
"""Select MACE-relaxed generated CIFs for Figure 5 visualization.

Example:
    python scripts/select_mace_relaxed_visualization_samples.py \
      --cif-dir outputs/20260601_mace_relax_final_256 \
      --target-count 100 \
      --output-dir outputs/visualization/generated_samples_mace_relaxed_100 \
      --min-distance 0.5 \
      --exclude-single-element \
      --exclude-hydrogen \
      --validity-csv outputs/eval/validity_per_sample.csv \
      --uniqueness-csv outputs/eval/uniqueness_per_sample.csv \
      --novelty-csv outputs/eval/novelty_per_sample.csv

When uniqueness or novelty CSVs are not provided, the script can call the
existing project evaluation helpers and merge their per-sample outputs. It does
not replace those metrics with local heuristics, because Figure 5 bookkeeping
needs generated-vs-generated uniqueness and generated-vs-reference novelty.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    from pymatgen.core import Structure
except ImportError as exc:  # pragma: no cover - exercised in user envs only
    raise SystemExit(
        "Missing required dependencies for CIF selection. Install/use the project "
        f"environment with pymatgen and numpy. Original error: {exc}"
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("select_mace_relaxed_visualization_samples")
SAMPLE_TOKEN_RE = re.compile(r"(?:crystal_|comp_invalid_crystal_|struct_invalid_crystal_)?epoch_(\d+)_sample_(\d+)")
BOOL_TRUE = {"1", "true", "t", "yes", "y"}
BOOL_FALSE = {"0", "false", "f", "no", "n", ""}


try:
    from mp20.crystal import smact_validity as _project_smact_validity
except Exception as exc:  # pragma: no cover - depends on optional smact install
    _project_smact_validity = None
    _SMACT_IMPORT_ERROR = exc
else:
    _SMACT_IMPORT_ERROR = None


@dataclass
class MergeHit:
    row: dict[str, Any]
    strategy: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    return None


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return ""


def safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return int(float(text))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=json_default) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_path_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve(strict=False))


def normalize_path_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def sample_key_from_text(text: Any) -> str:
    norm = normalize_path_text(text)
    matches = SAMPLE_TOKEN_RE.findall(norm)
    if not matches:
        return ""
    epoch, sample = matches[-1]
    return f"epoch_{int(epoch)}_sample_{int(sample)}"


def worker_from_text(text: Any) -> str:
    for part in Path(normalize_path_text(text)).parts:
        if part.startswith("worker_"):
            return part
    return ""


def path_keys(path: Path) -> dict[str, str]:
    resolved = str(path.resolve(strict=False))
    worker = worker_from_text(path)
    raw_stem = path.parent.name if path.name == "relaxed.cif" else path.stem
    keys = {
        "resolved_absolute_path": resolved,
        "filename": path.name,
        "stem": path.stem,
        "raw_stem": raw_stem,
        "sample_key": sample_key_from_text(path),
    }
    if worker and raw_stem:
        keys["worker_raw_stem"] = f"{worker}|{raw_stem}"
    parts = normalize_path_text(path)
    marker = "relaxed_cifs_success_only/"
    if marker in parts:
        keys["relaxed_success_suffix"] = parts.split(marker, 1)[1]
    if path.name == "relaxed.cif":
        keys["parent_name"] = path.parent.name
    return {key: value for key, value in keys.items() if value}


def row_keys(row: dict[str, Any]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for field in ("source_cif_path", "cif_path", "copied_cif_path", "relaxed_cif", "input", "filename"):
        value = row.get(field, "")
        if not value:
            continue
        if field in {"source_cif_path", "cif_path", "copied_cif_path", "relaxed_cif"}:
            resolved = resolve_path_text(value)
            if resolved:
                keys.setdefault("resolved_absolute_path", resolved)
        norm = normalize_path_text(value)
        if norm:
            path = Path(norm)
            filename = path.name
            stem = path.stem
            worker = str(row.get("worker") or worker_from_text(norm))
            raw_stem = path.parent.name if filename == "relaxed.cif" else stem
            keys.setdefault("filename", filename)
            keys.setdefault("stem", stem)
            keys.setdefault("raw_stem", raw_stem)
            if worker and raw_stem:
                keys.setdefault("worker_raw_stem", f"{worker}|{raw_stem}")
            sample_key = sample_key_from_text(norm)
            if sample_key:
                keys.setdefault("sample_key", sample_key)
            if "relaxed_cifs_success_only/" in norm:
                keys.setdefault("relaxed_success_suffix", norm.split("relaxed_cifs_success_only/", 1)[1])
            if filename == "relaxed.cif":
                keys.setdefault("parent_name", path.parent.name)
    return keys


def build_merge_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for strategy, key in row_keys(row).items():
            index[strategy][key].append(row)
    return index


def find_merge_hit(path: Path, index: dict[str, dict[str, list[dict[str, Any]]]]) -> MergeHit | None:
    for strategy, key in path_keys(path).items():
        candidates = index.get(strategy, {}).get(key, [])
        if len(candidates) == 1:
            return MergeHit(candidates[0], strategy)
    return None


def bool_from_row(row: dict[str, Any], field_candidates: Iterable[str]) -> bool | None:
    value = first_present(row, field_candidates)
    return parse_bool(value)


def composition_components(structure: Structure) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = Counter(int(z) for z in structure.atomic_numbers)
    elems = tuple(sorted(counts))
    raw_counts = np.array([counts[z] for z in elems], dtype=int)
    gcd = int(np.gcd.reduce(raw_counts)) if len(raw_counts) else 1
    gcd = max(gcd, 1)
    return elems, tuple((raw_counts // gcd).astype(int).tolist())


def compute_comp_valid(structure: Structure) -> tuple[bool | None, str]:
    if _project_smact_validity is None:
        return None, f"mp20.crystal.smact_validity import failed: {_SMACT_IMPORT_ERROR}"
    elems, comps = composition_components(structure)
    try:
        return bool(_project_smact_validity(elems, comps)), "mp20.crystal.smact_validity"
    except Exception as exc:
        return None, f"smact_validity failed: {exc}"


def pbc_min_distance(structure: Structure, cutoff: float) -> tuple[float | None, int]:
    if len(structure) < 2:
        return None, 0
    dist_mat = np.array(structure.distance_matrix, dtype=float)
    iu = np.triu_indices(len(structure), k=1)
    pair_dist = dist_mat[iu]
    min_dist = float(pair_dist.min()) if pair_dist.size else None
    close_pairs = int((pair_dist < cutoff).sum()) if pair_dist.size else 0
    return min_dist, close_pairs


def structure_metadata(path: Path, min_distance: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        structure = Structure.from_file(str(path))
        min_pbc_distance, close_pairs = pbc_min_distance(structure, min_distance)
        comp_valid, comp_source = compute_comp_valid(structure)
        elements = [str(el) for el in structure.composition.elements]
        volume = float(structure.volume)
        num_sites = int(len(structure))
        metadata = {
            "source_cif_path": str(path.resolve(strict=False)),
            "filename": path.name,
            "stem": path.stem,
            "sample_key": sample_key_from_text(path),
            "formula": structure.composition.formula,
            "reduced_formula": structure.composition.reduced_formula,
            "num_sites": num_sites,
            "density": float(structure.density),
            "volume": volume,
            "volume_per_atom": float(volume / num_sites) if num_sites else math.nan,
            "elements": ";".join(elements),
            "num_unique_elements": len(elements),
            "contains_hydrogen": "H" in elements,
            "min_pbc_distance": min_pbc_distance,
            "pbc_close_pair_count": close_pairs,
            "struct_valid": bool(volume >= 0.1 and (min_pbc_distance is None or min_pbc_distance >= min_distance)),
            "comp_valid": comp_valid,
            "comp_valid_source": comp_source,
            "parse_success": True,
            "_structure": structure,
        }
        return metadata, None
    except Exception as exc:
        return None, {
            "kind": "parse_or_metadata_failure",
            "cif_path": str(path),
            "error_message": str(exc),
        }


def load_optional_csv(path_text: str | None) -> tuple[list[dict[str, Any]], str | None]:
    if not path_text:
        return [], None
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    return read_csv_rows(path), str(path.resolve(strict=False))


def run_helper(cmd: list[str], kind: str) -> None:
    LOGGER.info("Running %s helper: %s", kind, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def ensure_metric_csvs(args: argparse.Namespace, output_dir: Path) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Return uniqueness/novelty CSV paths, computing them when requested."""
    failures: list[dict[str, Any]] = []
    uniqueness_csv = args.uniqueness_csv
    novelty_csv = args.novelty_csv
    eval_dir = output_dir / "_computed_eval"

    if uniqueness_csv and novelty_csv:
        return uniqueness_csv, novelty_csv, failures
    if not args.compute_missing_evals:
        if not uniqueness_csv:
            failures.append(
                {
                    "kind": "missing_uniqueness_csv",
                    "error_message": "Pass --uniqueness-csv or enable --compute-missing-evals.",
                }
            )
        if not novelty_csv:
            failures.append(
                {
                    "kind": "missing_novelty_csv",
                    "error_message": "Pass --novelty-csv or enable --compute-missing-evals.",
                }
            )
        return uniqueness_csv, novelty_csv, failures

    if not uniqueness_csv:
        uniqueness_dir = eval_dir / "uniqueness"
        uniqueness_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_uniqueness_from_cifs.py"),
                "--sample-dir",
                str(args.cif_dir),
                "--output-dir",
                str(uniqueness_dir),
                "--num-workers",
                str(args.num_workers),
                "--valid-only",
                "False",
                "--save-per-sample",
                "True",
            ]
            if args.max_cifs is not None:
                cmd.extend(["--max-samples", str(args.max_cifs)])
            run_helper(cmd, "uniqueness")
            candidate = uniqueness_dir / "uniqueness_per_sample.csv"
            if candidate.exists():
                uniqueness_csv = str(candidate)
            else:
                failures.append(
                    {
                        "kind": "missing_computed_uniqueness_csv",
                        "directory": str(uniqueness_dir),
                        "error_message": "Uniqueness helper completed but did not create uniqueness_per_sample.csv.",
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "kind": "uniqueness_helper_failure",
                    "error_message": str(exc),
                }
            )

    if not novelty_csv:
        novelty_dir = eval_dir / "novelty"
        novelty_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_novelty_from_cifs.py"),
                "--sample-dir",
                str(args.cif_dir),
                "--mp20-root",
                str(args.mp20_root),
                "--processed-dir",
                str(args.processed_dir),
                "--output-dir",
                str(novelty_dir),
                "--num-workers",
                str(args.num_workers),
                "--valid-only",
                "False",
                "--save-per-sample",
                "True",
            ]
            if args.max_cifs is not None:
                cmd.extend(["--max-samples", str(args.max_cifs)])
            run_helper(cmd, "novelty")
            candidate = novelty_dir / "novelty_per_sample.csv"
            if candidate.exists():
                novelty_csv = str(candidate)
            else:
                failures.append(
                    {
                        "kind": "missing_computed_novelty_csv",
                        "directory": str(novelty_dir),
                        "error_message": "Novelty helper completed but did not create novelty_per_sample.csv.",
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "kind": "novelty_helper_failure",
                    "error_message": str(exc),
                }
            )

    return uniqueness_csv, novelty_csv, failures


def apply_external_rows(
    records: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
    *,
    kind: str,
    merge_counts: Counter[str],
) -> None:
    if not csv_rows:
        return
    index = build_merge_index(csv_rows)
    for record in records:
        hit = find_merge_hit(Path(record["source_cif_path"]), index)
        if not hit:
            continue
        merge_counts[f"{kind}:{hit.strategy}"] += 1
        row = hit.row
        record[f"{kind}_merge_strategy"] = hit.strategy
        if kind == "validity":
            for field, candidates in {
                "struct_valid": ("struct_valid", "structure_valid", "geometry_valid"),
                "comp_valid": ("comp_valid", "composition_valid"),
                "mace_success": ("mace_success", "converged", "success"),
            }.items():
                value = bool_from_row(row, candidates)
                if value is not None:
                    record[field] = value
            min_dist = safe_float(first_present(row, ("min_pbc_distance", "pymatgen_min_dist", "min_distance")))
            if min_dist is not None:
                record["min_pbc_distance"] = min_dist
        elif kind == "uniqueness":
            value = bool_from_row(row, ("unique", "unique_representative", "is_unique_representative"))
            if value is not None:
                record["unique"] = value
        elif kind == "novelty":
            value = bool_from_row(row, ("novel", "is_novel"))
            if value is not None:
                record["novel"] = value
        elif kind == "mace":
            status = str(first_present(row, ("status", "mace_status"))).strip().lower()
            converged = bool_from_row(row, ("mace_success", "converged", "success"))
            record["mace_status"] = status or record.get("mace_status", "")
            if converged is not None:
                record["mace_success"] = bool(converged)
            elif status:
                record["mace_success"] = status == "success"
            for field in ("steps", "final_energy_eV", "final_max_force_eV_A", "elapsed_s"):
                if field in row and str(row[field]).strip():
                    record[field] = row[field]


def infer_mace_summary_csv(cif_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    candidates = [
        cif_dir / "summary" / "final_after_continue_records.csv",
        cif_dir / "summary" / "relax_records.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def initialize_default_flags(records: list[dict[str, Any]]) -> None:
    for record in records:
        record.setdefault("unique", None)
        record.setdefault("novel", None)
        if "mace_success" not in record:
            path = Path(record["source_cif_path"])
            record["mace_success"] = "relaxed_cifs_success_only" in path.parts
            record["mace_status"] = "success_inferred_from_directory" if record["mace_success"] else ""


def filter_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "parsed_count": len(records),
        "struct_valid_count": sum(r.get("struct_valid") is True for r in records),
        "comp_valid_count": sum(r.get("comp_valid") is True for r in records),
        "unique_count": sum(r.get("unique") is True for r in records),
        "novel_count": sum(r.get("novel") is True for r in records),
        "struct_and_comp_valid_count": sum(r.get("struct_valid") is True and r.get("comp_valid") is True for r in records),
        "valid_unique_novel_count": sum(
            r.get("struct_valid") is True
            and r.get("comp_valid") is True
            and r.get("unique") is True
            and r.get("novel") is True
            for r in records
        ),
    }


def build_candidates(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        if not (
            record.get("struct_valid") is True
            and record.get("comp_valid") is True
            and record.get("unique") is True
            and record.get("novel") is True
        ):
            continue
        if args.require_mace_success and record.get("mace_success") is False:
            continue
        if args.exclude_single_element and int(record.get("num_unique_elements") or 0) <= 1:
            continue
        if args.exclude_hydrogen and record.get("contains_hydrogen"):
            continue
        candidates.append(record)
    return candidates


def diversity_bin(record: dict[str, Any]) -> tuple[int, int, int]:
    num_elements = int(record.get("num_unique_elements") or 0)
    num_sites = int(record.get("num_sites") or 0)
    volume_per_atom = float(record.get("volume_per_atom") or 0.0)
    site_bin = min(num_sites // 4, 8)
    volume_bin = min(int(volume_per_atom // 5), 8)
    return num_elements, site_bin, volume_bin


def select_diverse(candidates: list[dict[str, Any]], target_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_formula[str(record.get("reduced_formula", ""))].append(record)

    formula_order = list(by_formula)
    rng.shuffle(formula_order)
    formula_order.sort(
        key=lambda formula: (
            -len({diversity_bin(row) for row in by_formula[formula]}),
            formula,
        )
    )

    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    used_bins: Counter[tuple[int, int, int]] = Counter()

    for formula in formula_order:
        group = by_formula[formula]
        rng.shuffle(group)
        group.sort(key=lambda row: (used_bins[diversity_bin(row)], diversity_bin(row), row["source_cif_path"]))
        row = group[0]
        selected.append(row)
        used_ids.add(id(row))
        used_bins[diversity_bin(row)] += 1
        if len(selected) >= target_count:
            return selected

    remaining = [row for row in candidates if id(row) not in used_ids]
    rng.shuffle(remaining)
    remaining.sort(key=lambda row: (used_bins[diversity_bin(row)], diversity_bin(row), row["source_cif_path"]))
    for row in remaining:
        selected.append(row)
        used_bins[diversity_bin(row)] += 1
        if len(selected) >= target_count:
            break
    return selected


def copy_selected(selected: list[dict[str, Any]], selected_dir: Path) -> list[dict[str, Any]]:
    cifs_dir = selected_dir / "cifs"
    cifs_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for idx, record in enumerate(selected):
        sample_id = f"sample_{idx:03d}"
        dst = cifs_dir / f"{sample_id}.cif"
        shutil.copy2(record["source_cif_path"], dst)
        out = dict(record)
        out.pop("_structure", None)
        out["sample_id"] = sample_id
        out["copied_cif_path"] = str(dst.resolve(strict=False))
        copied.append(out)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cif-dir", type=Path, default=Path("outputs/20260601_mace_relax_final_256"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization/generated_samples_mace_relaxed_100"))
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--max-cifs", type=int, default=None)
    parser.add_argument("--min-distance", type=float, default=0.5)
    parser.add_argument("--validity-csv", default=None)
    parser.add_argument("--uniqueness-csv", default=None)
    parser.add_argument("--novelty-csv", default=None)
    parser.add_argument("--mace-summary-csv", default=None)
    parser.add_argument("--compute-missing-evals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mp20-root", type=Path, default=Path("mp20"))
    parser.add_argument("--processed-dir", type=Path, default=Path("mp20/precessed"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exclude-single-element", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-hydrogen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-mace-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    cif_dir = args.cif_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    selected_dir = output_dir / "selected"
    failures_path = selected_dir / "selection_failures.jsonl"

    if _project_smact_validity is None:
        LOGGER.warning(
            "Could not import mp20.crystal.smact_validity; local comp_valid will be unset. "
            "Pass --validity-csv with comp_valid to merge external composition validity. "
            "Import error: %s",
            _SMACT_IMPORT_ERROR,
        )

    all_cifs = sorted(cif_dir.rglob("*.cif"), key=lambda path: str(path))
    if args.max_cifs is not None:
        all_cifs = all_cifs[: max(0, args.max_cifs)]
    LOGGER.info("Found %d CIF files under %s", len(all_cifs), cif_dir)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in all_cifs:
        metadata, failure = structure_metadata(path, args.min_distance)
        if failure:
            failures.append(failure)
        elif metadata is not None:
            records.append(metadata)
    initialize_default_flags(records)

    merge_counts: Counter[str] = Counter()
    uniqueness_csv, novelty_csv, helper_failures = ensure_metric_csvs(args, output_dir)
    failures.extend(helper_failures)

    validity_rows, validity_source = load_optional_csv(args.validity_csv)
    uniqueness_rows, uniqueness_source = load_optional_csv(uniqueness_csv)
    novelty_rows, novelty_source = load_optional_csv(novelty_csv)
    mace_csv = infer_mace_summary_csv(cif_dir, args.mace_summary_csv)
    mace_rows = read_csv_rows(mace_csv) if mace_csv else []

    if validity_rows:
        apply_external_rows(records, validity_rows, kind="validity", merge_counts=merge_counts)
    if uniqueness_rows:
        apply_external_rows(records, uniqueness_rows, kind="uniqueness", merge_counts=merge_counts)
    if novelty_rows:
        apply_external_rows(records, novelty_rows, kind="novelty", merge_counts=merge_counts)
    if mace_rows:
        apply_external_rows(records, mace_rows, kind="mace", merge_counts=merge_counts)

    if not uniqueness_rows:
        failures.append(
            {
                "kind": "missing_uniqueness_csv",
                "error_message": "Pass --uniqueness-csv from scripts/evaluate_uniqueness_from_cifs.py or allow helper computation.",
            }
        )
    if not novelty_rows:
        failures.append(
            {
                "kind": "missing_novelty_csv",
                "error_message": "Pass --novelty-csv from scripts/evaluate_novelty_from_cifs.py or allow helper computation.",
            }
        )

    counts = {"total_cifs": len(all_cifs), **filter_counts(records)}
    candidates = build_candidates(records, args)
    counts["candidate_after_exclusions_count"] = len(candidates)

    if len(candidates) < args.target_count:
        counts["final_selected_count"] = 0
        summary = {
            "created_at": now_iso(),
            "status": "failed_insufficient_candidates",
            "args": vars(args),
            "counts": counts,
            "merge_counts": dict(merge_counts),
            "sources": {
                "validity_csv": validity_source,
                "uniqueness_csv": uniqueness_source,
                "novelty_csv": novelty_source,
                "mace_summary_csv": str(mace_csv.resolve(strict=False)) if mace_csv else None,
            },
            "warnings": [
                "SMACT composition validity is local; uniqueness/novelty must come from external evaluation CSVs."
            ],
        }
        write_json(selected_dir / "selection_summary.json", summary)
        write_jsonl(failures_path, failures)
        raise SystemExit(
            "Not enough candidates for requested target count. "
            f"Need {args.target_count}, found {len(candidates)}. "
            f"Counts: {counts}"
        )

    selected = select_diverse(candidates, args.target_count, args.seed)
    copied = copy_selected(selected, selected_dir)
    counts["final_selected_count"] = len(copied)

    fieldnames = [
        "sample_id",
        "source_cif_path",
        "copied_cif_path",
        "filename",
        "stem",
        "sample_key",
        "formula",
        "reduced_formula",
        "num_sites",
        "density",
        "volume",
        "volume_per_atom",
        "elements",
        "num_unique_elements",
        "contains_hydrogen",
        "min_pbc_distance",
        "pbc_close_pair_count",
        "struct_valid",
        "comp_valid",
        "unique",
        "novel",
        "mace_success",
        "mace_status",
        "validity_merge_strategy",
        "uniqueness_merge_strategy",
        "novelty_merge_strategy",
        "mace_merge_strategy",
    ]
    write_csv(selected_dir / "selected_samples.csv", copied, fieldnames)
    write_json(selected_dir / "selected_samples.json", copied)
    write_jsonl(failures_path, failures)

    summary = {
        "created_at": now_iso(),
        "status": "ok",
        "args": vars(args),
        "counts": counts,
        "merge_counts": dict(merge_counts),
        "sources": {
            "validity_csv": validity_source,
            "uniqueness_csv": uniqueness_source,
            "novelty_csv": novelty_source,
            "mace_summary_csv": str(mace_csv.resolve(strict=False)) if mace_csv else None,
        },
        "selection_strategy": {
            "formula_deduplication_preferred": True,
            "diversity_bins": "num_unique_elements, num_sites//4, volume_per_atom//5",
            "seed": args.seed,
        },
        "failures_jsonl": str(failures_path.resolve(strict=False)),
    }
    write_json(selected_dir / "selection_summary.json", summary)
    LOGGER.info("Selected %d samples into %s", len(copied), selected_dir)


if __name__ == "__main__":
    main()
