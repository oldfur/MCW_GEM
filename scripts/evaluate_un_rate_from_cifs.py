#!/usr/bin/env python
"""Offline CrystalDiT-style UN Rate evaluation from generated CIF files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

UNIQUE_PER_SAMPLE = "uniqueness_per_sample.csv"
UNIQUE_SUMMARY = "uniqueness_summary.json"
NOVELTY_PER_SAMPLE = "novelty_per_sample.csv"
NOVELTY_SUMMARY = "novelty_summary.json"
UN_SAMPLE_DIRNAME = "unique_novel_cifs"
UN_SAMPLE_MANIFEST = "manifest.csv"

MERGE_STRATEGIES = [
    "resolved_absolute_path",
    "relative_path",
    "worker_epoch_filename",
    "filename",
]


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=json_default) + "\n")


def normalize_path_text(path_text: str) -> str:
    return path_text.replace("\\", "/").strip()


def path_parts_from_row(row: Dict[str, str]) -> List[str]:
    values = [
        row.get("relative_path", ""),
        row.get("cif_path", ""),
    ]
    for value in values:
        if value:
            return [part for part in Path(value).parts if part not in {"", "."}]
    return []


def find_epoch(row: Dict[str, str]) -> str:
    for part in path_parts_from_row(row):
        if part.startswith("epoch_"):
            return part
    return ""


def find_worker(row: Dict[str, str]) -> str:
    worker = row.get("worker", "").strip()
    if worker:
        return worker
    for part in path_parts_from_row(row):
        if part.startswith("worker_"):
            return part
    return ""


def filename_for_row(row: Dict[str, str]) -> str:
    filename = row.get("filename", "").strip()
    if filename:
        return filename
    cif_path = row.get("cif_path", "").strip()
    if cif_path:
        return Path(cif_path).name
    return ""


def key_for_row(row: Dict[str, str], strategy: str) -> str:
    cif_path = row.get("cif_path", "").strip()
    if strategy == "resolved_absolute_path":
        if not cif_path:
            return ""
        return str(Path(cif_path).expanduser().resolve(strict=False))
    if strategy == "relative_path":
        rel_path = row.get("relative_path", "").strip()
        return normalize_path_text(rel_path)
    if strategy == "worker_epoch_filename":
        worker = find_worker(row)
        epoch = find_epoch(row)
        filename = filename_for_row(row)
        if not filename or (not worker and not epoch):
            return ""
        return "|".join([worker, epoch, filename])
    if strategy == "filename":
        return filename_for_row(row)
    raise ValueError(f"Unsupported merge strategy: {strategy}")


def row_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {
        strategy: key
        for strategy in MERGE_STRATEGIES
        if (key := key_for_row(row, strategy))
    }


def build_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[int]]]:
    indexes: Dict[str, Dict[str, List[int]]] = {
        strategy: {} for strategy in MERGE_STRATEGIES
    }
    for idx, row in enumerate(rows):
        for strategy, key in row_keys(row).items():
            indexes[strategy].setdefault(key, []).append(idx)
    return indexes


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def choose_value(*values: Any) -> str:
    for value in values:
        text = str(value).strip()
        if text != "":
            return text
    return ""


def combine_errors(*messages: Any) -> str:
    parts = [str(message).strip() for message in messages if str(message).strip()]
    return " | ".join(parts)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_helper(cmd: List[str], kind: str) -> Optional[Dict[str, Any]]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    print("Running", kind, "helper:", flush=True)
    print(" ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    except subprocess.CalledProcessError as exc:
        return {
            "kind": "helper_execution_failure",
            "helper": kind,
            "returncode": exc.returncode,
            "command": cmd,
            "error_message": f"{kind} helper exited with code {exc.returncode}",
        }
    return None


def uniqueness_outputs_exist(directory: Path) -> bool:
    return (directory / UNIQUE_PER_SAMPLE).exists() and (directory / UNIQUE_SUMMARY).exists()


def novelty_outputs_exist(directory: Path) -> bool:
    return (directory / NOVELTY_PER_SAMPLE).exists() and (directory / NOVELTY_SUMMARY).exists()


def resolve_eval_dirs(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path, str, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    tmp_base = Path(args.tmp_dir).expanduser() if args.tmp_dir else output_dir / "_tmp"
    uniqueness_dir = Path(args.uniqueness_dir).expanduser() if args.uniqueness_dir else tmp_base / "uniqueness"
    novelty_dir = Path(args.novelty_dir).expanduser() if args.novelty_dir else tmp_base / "novelty"

    need_uniqueness = (
        not uniqueness_outputs_exist(uniqueness_dir)
        if args.uniqueness_dir
        else True
    )
    need_novelty = (
        not novelty_outputs_exist(novelty_dir)
        if args.novelty_dir
        else True
    )
    mode = "merge_existing_outputs"
    if need_uniqueness or need_novelty:
        if not args.recompute_missing:
            if need_uniqueness:
                failures.append(
                    {
                        "kind": "missing_uniqueness_output",
                        "directory": str(uniqueness_dir),
                        "error_message": "Missing uniqueness outputs and --recompute-missing=False.",
                    }
                )
            if need_novelty:
                failures.append(
                    {
                        "kind": "missing_novelty_output",
                        "directory": str(novelty_dir),
                        "error_message": "Missing novelty outputs and --recompute-missing=False.",
                    }
                )
            return uniqueness_dir, novelty_dir, mode, failures
        mode = "computed_via_existing_cli_then_merged"

    if need_uniqueness and args.recompute_missing:
        ensure_output_dir(uniqueness_dir)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_uniqueness_from_cifs.py"),
            "--sample-glob",
            args.sample_glob,
            "--output-dir",
            str(uniqueness_dir),
            "--num-workers",
            str(args.num_workers),
            "--valid-only",
            str(bool(args.valid_only)),
            "--save-per-sample",
            "True",
        ]
        if args.sample_dir:
            cmd.extend(["--sample-dir", args.sample_dir])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        failure = run_helper(cmd, "uniqueness")
        if failure:
            failures.append(failure)
        elif not uniqueness_outputs_exist(uniqueness_dir):
            failures.append(
                {
                    "kind": "missing_uniqueness_output",
                    "directory": str(uniqueness_dir),
                    "error_message": "Uniqueness helper completed but required outputs are missing.",
                }
            )

    if need_novelty and args.recompute_missing:
        ensure_output_dir(novelty_dir)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_novelty_from_cifs.py"),
            "--sample-glob",
            args.sample_glob,
            "--mp20-root",
            args.mp20_root,
            "--processed-dir",
            args.processed_dir,
            "--output-dir",
            str(novelty_dir),
            "--num-workers",
            str(args.num_workers),
            "--valid-only",
            str(bool(args.valid_only)),
            "--save-per-sample",
            "True",
        ]
        if args.sample_dir:
            cmd.extend(["--sample-dir", args.sample_dir])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        failure = run_helper(cmd, "novelty")
        if failure:
            failures.append(failure)
        elif not novelty_outputs_exist(novelty_dir):
            failures.append(
                {
                    "kind": "missing_novelty_output",
                    "directory": str(novelty_dir),
                    "error_message": "Novelty helper completed but required outputs are missing.",
                }
            )

    return uniqueness_dir, novelty_dir, mode, failures


def parse_failure_rows(rows: List[Dict[str, str]], source: str) -> List[Dict[str, Any]]:
    failures = []
    for row in rows:
        if "parse_success" in row and not csv_bool(row.get("parse_success", "")):
            failures.append(
                {
                    "kind": "parse_failure",
                    "source": source,
                    "sample_index": row.get("sample_index", ""),
                    "cif_path": row.get("cif_path", ""),
                    "error_message": row.get("error_message", ""),
                }
            )
    return failures


def merge_rows(
    uniqueness_rows: List[Dict[str, str]],
    novelty_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    novelty_index = build_index(novelty_rows)
    used_novelty_rows = set()
    merged_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    strategy_counts: Counter[str] = Counter()
    unmatched_uniqueness = 0

    for uniqueness_idx, uniqueness_row in enumerate(
        tqdm(uniqueness_rows, total=len(uniqueness_rows), desc="Merge UN rows")
    ):
        matched_novelty_idx: Optional[int] = None
        matched_strategy = ""
        ambiguous: List[Dict[str, Any]] = []

        for strategy in MERGE_STRATEGIES:
            key = key_for_row(uniqueness_row, strategy)
            if not key:
                continue
            candidate_indices = [
                idx
                for idx in novelty_index[strategy].get(key, [])
                if idx not in used_novelty_rows
            ]
            if len(candidate_indices) == 1:
                matched_novelty_idx = candidate_indices[0]
                matched_strategy = strategy
                break
            if len(candidate_indices) > 1:
                ambiguous.append(
                    {
                        "strategy": strategy,
                        "key": key,
                        "candidate_indices": candidate_indices,
                    }
                )

        if matched_novelty_idx is None:
            unmatched_uniqueness += 1
            failures.append(
                {
                    "kind": "missing_novelty_row",
                    "uniqueness_row_index": uniqueness_idx,
                    "sample_index": uniqueness_row.get("sample_index", ""),
                    "cif_path": uniqueness_row.get("cif_path", ""),
                    "candidate_keys": row_keys(uniqueness_row),
                    "ambiguous_candidates": ambiguous,
                    "error_message": "No matching novelty row found for uniqueness row.",
                }
            )
            continue

        used_novelty_rows.add(matched_novelty_idx)
        novelty_row = novelty_rows[matched_novelty_idx]
        strategy_counts[matched_strategy] += 1
        merged_row = merge_one_row(uniqueness_row, novelty_row, matched_strategy)
        merged_rows.append(merged_row)

        if merged_row["evaluated_uniqueness"] != merged_row["evaluated_novelty"]:
            failures.append(
                {
                    "kind": "inconsistent_evaluated_flag",
                    "sample_index": merged_row["sample_index"],
                    "cif_path": merged_row["cif_path"],
                    "evaluated_uniqueness": merged_row["evaluated_uniqueness"],
                    "evaluated_novelty": merged_row["evaluated_novelty"],
                    "error_message": "Uniqueness and novelty evaluated flags differ.",
                }
            )

        formula_u = uniqueness_row.get("formula", "").strip()
        formula_n = novelty_row.get("formula", "").strip()
        reduced_u = uniqueness_row.get("reduced_formula", "").strip()
        reduced_n = novelty_row.get("reduced_formula", "").strip()
        sites_u = uniqueness_row.get("num_sites", "").strip()
        sites_n = novelty_row.get("num_sites", "").strip()
        if (formula_u and formula_n and formula_u != formula_n) or (
            reduced_u and reduced_n and reduced_u != reduced_n
        ) or (sites_u and sites_n and sites_u != sites_n):
            failures.append(
                {
                    "kind": "formula_mismatch",
                    "sample_index": merged_row["sample_index"],
                    "cif_path": merged_row["cif_path"],
                    "uniqueness_formula": formula_u,
                    "novelty_formula": formula_n,
                    "uniqueness_reduced_formula": reduced_u,
                    "novelty_reduced_formula": reduced_n,
                    "uniqueness_num_sites": sites_u,
                    "novelty_num_sites": sites_n,
                    "error_message": "Formula/reduced_formula/num_sites differ after merge.",
                }
            )

    unmatched_novelty = 0
    for novelty_idx, novelty_row in enumerate(novelty_rows):
        if novelty_idx in used_novelty_rows:
            continue
        unmatched_novelty += 1
        failures.append(
            {
                "kind": "missing_uniqueness_row",
                "novelty_row_index": novelty_idx,
                "sample_index": novelty_row.get("sample_index", ""),
                "cif_path": novelty_row.get("cif_path", ""),
                "candidate_keys": row_keys(novelty_row),
                "error_message": "No matching uniqueness row found for novelty row.",
            }
        )

    merge_info = {
        "strategy_order": MERGE_STRATEGIES,
        "matched_rows_by_strategy": dict(strategy_counts),
        "matched_rows": len(merged_rows),
        "unmatched_uniqueness_rows": unmatched_uniqueness,
        "unmatched_novelty_rows": unmatched_novelty,
    }
    return merged_rows, failures, merge_info


def merge_one_row(
    uniqueness_row: Dict[str, str],
    novelty_row: Dict[str, str],
    merge_strategy: str,
) -> Dict[str, Any]:
    evaluated_uniqueness = csv_bool(uniqueness_row.get("evaluated", ""))
    evaluated_novelty = csv_bool(novelty_row.get("evaluated", ""))
    evaluated = evaluated_uniqueness and evaluated_novelty
    unique_representative = csv_bool(uniqueness_row.get("unique_representative", ""))
    novel = csv_bool(novelty_row.get("novel", ""))
    unique_and_novel = bool(evaluated and unique_representative and novel)
    sample_index = choose_value(
        uniqueness_row.get("sample_index", ""),
        novelty_row.get("sample_index", ""),
    )

    return {
        "sample_index": safe_int(sample_index, sample_index),
        "cif_path": choose_value(
            uniqueness_row.get("cif_path", ""),
            novelty_row.get("cif_path", ""),
        ),
        "evaluated": evaluated,
        "evaluated_uniqueness": evaluated_uniqueness,
        "evaluated_novelty": evaluated_novelty,
        "unique_representative": unique_representative,
        "novel": novel,
        "unique_and_novel": unique_and_novel,
        "cluster_id": uniqueness_row.get("cluster_id", ""),
        "matched_representative_index": uniqueness_row.get("matched_representative_index", ""),
        "matched_representative_path": uniqueness_row.get("matched_representative_path", ""),
        "matched_reference_index": novelty_row.get("matched_reference_index", ""),
        "matched_reference_id": novelty_row.get("matched_reference_id", ""),
        "matched_reference_path": novelty_row.get("matched_reference_path", ""),
        "formula": choose_value(
            uniqueness_row.get("formula", ""),
            novelty_row.get("formula", ""),
        ),
        "reduced_formula": choose_value(
            uniqueness_row.get("reduced_formula", ""),
            novelty_row.get("reduced_formula", ""),
        ),
        "num_sites": choose_value(
            uniqueness_row.get("num_sites", ""),
            novelty_row.get("num_sites", ""),
        ),
        "merge_strategy": merge_strategy,
        "error_message": combine_errors(
            uniqueness_row.get("error_message", ""),
            novelty_row.get("error_message", ""),
        ),
    }


def write_per_sample_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "cif_path",
        "evaluated",
        "evaluated_uniqueness",
        "evaluated_novelty",
        "unique_representative",
        "novel",
        "unique_and_novel",
        "cluster_id",
        "matched_representative_index",
        "matched_representative_path",
        "matched_reference_index",
        "matched_reference_id",
        "matched_reference_path",
        "formula",
        "reduced_formula",
        "num_sites",
        "merge_strategy",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_un_structures(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    un_rows = []
    for row in rows:
        if not row.get("unique_and_novel"):
            continue
        un_rows.append(
            {
                "un_index": len(un_rows),
                "sample_index": row.get("sample_index", ""),
                "cif_path": row.get("cif_path", ""),
                "formula": row.get("formula", ""),
                "reduced_formula": row.get("reduced_formula", ""),
                "cluster_id": row.get("cluster_id", ""),
                "novelty_match_status": "novel"
                if row.get("novel")
                else "matched_reference",
            }
        )
    return un_rows


def safe_filename_piece(value: Any, default: str) -> str:
    text = str(value).strip() or default
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or default


def resolve_cif_source(path_text: str) -> Path:
    source = Path(path_text).expanduser()
    candidates = [source]
    if not source.is_absolute():
        candidates.append(REPO_ROOT / source)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source


def un_sample_filename(row: Dict[str, Any], source: Path) -> str:
    un_index = safe_int(row.get("un_index", ""), 0) or 0
    sample_index = safe_filename_piece(row.get("sample_index", ""), "unknown")
    stem = safe_filename_piece(source.stem, "sample")
    suffix = source.suffix if source.suffix else ".cif"
    return f"un_{un_index:06d}_sample_{sample_index}_{stem}{suffix}"


def write_un_sample_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "un_index",
        "sample_index",
        "cif_path",
        "resolved_cif_path",
        "exported_cif_path",
        "exported_filename",
        "export_status",
        "formula",
        "reduced_formula",
        "cluster_id",
        "novelty_match_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_un_samples(
    args: argparse.Namespace,
    output_dir: Path,
    un_structures: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not args.save_un_samples:
        return {
            "enabled": False,
            "requested_count": len(un_structures),
            "exported_count": 0,
            "failure_count": 0,
        }, []

    target_dir = (
        Path(args.un_sample_dir).expanduser()
        if args.un_sample_dir
        else output_dir / UN_SAMPLE_DIRNAME
    )
    ensure_output_dir(target_dir)
    manifest_path = target_dir / UN_SAMPLE_MANIFEST
    failures: List[Dict[str, Any]] = []
    exported_count = 0

    for row in tqdm(
        un_structures,
        total=len(un_structures),
        desc="Copy unique+novel CIFs",
    ):
        source_text = str(row.get("cif_path", "")).strip()
        if not source_text:
            row["export_status"] = "missing_source_path"
            failures.append(
                {
                    "kind": "missing_un_cif_path",
                    "un_index": row.get("un_index", ""),
                    "sample_index": row.get("sample_index", ""),
                    "error_message": "Unique and novel row has no cif_path.",
                }
            )
            continue

        source = resolve_cif_source(source_text)
        row["resolved_cif_path"] = str(source)
        if not source.exists():
            row["export_status"] = "missing_source_file"
            failures.append(
                {
                    "kind": "missing_un_cif_file",
                    "un_index": row.get("un_index", ""),
                    "sample_index": row.get("sample_index", ""),
                    "cif_path": source_text,
                    "resolved_cif_path": str(source),
                    "error_message": "Could not find source CIF for unique and novel sample.",
                }
            )
            continue

        target = target_dir / un_sample_filename(row, source)
        try:
            if source.resolve(strict=False) != target.resolve(strict=False):
                shutil.copy2(source, target)
        except OSError as exc:
            row["export_status"] = "copy_failed"
            failures.append(
                {
                    "kind": "copy_un_cif_failed",
                    "un_index": row.get("un_index", ""),
                    "sample_index": row.get("sample_index", ""),
                    "cif_path": source_text,
                    "resolved_cif_path": str(source),
                    "exported_cif_path": str(target),
                    "error_message": str(exc),
                }
            )
            continue

        row["export_status"] = "copied"
        row["exported_cif_path"] = str(target)
        row["exported_filename"] = target.name
        exported_count += 1

    write_un_sample_manifest(manifest_path, un_structures)
    return {
        "enabled": True,
        "directory": str(target_dir),
        "manifest": str(manifest_path),
        "requested_count": len(un_structures),
        "exported_count": exported_count,
        "failure_count": len(failures),
    }, failures


def sample_roots_from_summaries(
    uniqueness_summary: Dict[str, Any],
    novelty_summary: Dict[str, Any],
) -> List[str]:
    roots = []
    seen = set()
    for summary in (uniqueness_summary, novelty_summary):
        for root in summary.get("sample_roots", []) or []:
            root = str(root)
            if root in seen:
                continue
            seen.add(root)
            roots.append(root)
    return roots


def summary_count(
    key: str,
    uniqueness_summary: Dict[str, Any],
    novelty_summary: Dict[str, Any],
    fallback: int,
) -> int:
    for summary in (uniqueness_summary, novelty_summary):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return fallback


def build_summary(
    args: argparse.Namespace,
    start_time: float,
    mode: str,
    uniqueness_dir: Path,
    novelty_dir: Path,
    uniqueness_summary: Dict[str, Any],
    novelty_summary: Dict[str, Any],
    merged_rows: List[Dict[str, Any]],
    merge_info: Dict[str, Any],
    failures: List[Dict[str, Any]],
    un_sample_export: Dict[str, Any],
) -> Dict[str, Any]:
    evaluated_rows = [row for row in merged_rows if row.get("evaluated")]
    evaluated_count = len(evaluated_rows)
    unique_count = sum(1 for row in evaluated_rows if row.get("unique_representative"))
    novel_count = sum(1 for row in evaluated_rows if row.get("novel"))
    un_count = sum(1 for row in evaluated_rows if row.get("unique_and_novel"))
    duplicate_count = evaluated_count - unique_count
    non_novel_count = evaluated_count - novel_count

    return {
        "total_cif_files_found": summary_count(
            "total_cif_files_found",
            uniqueness_summary,
            novelty_summary,
            len(merged_rows),
        ),
        "evaluated_count": evaluated_count,
        "unique_count": unique_count,
        "novel_count": novel_count,
        "un_count": un_count,
        "un_sample_export": un_sample_export,
        "duplicate_count": duplicate_count,
        "non_novel_count": non_novel_count,
        "uniqueness_rate": rate(unique_count, evaluated_count),
        "novelty_rate": rate(novel_count, evaluated_count),
        "un_rate": rate(un_count, evaluated_count),
        "novel_among_unique_rate": rate(un_count, unique_count),
        "unique_among_novel_rate": rate(un_count, novel_count),
        "denominator_definition": (
            "matched per-sample generated structures with evaluated=True in both "
            "uniqueness_per_sample.csv and novelty_per_sample.csv; with "
            "--valid-only=True this is the valid evaluated generated sample set "
            "shared by both existing offline evaluators"
        ),
        "valid_only": bool(args.valid_only),
        "uniqueness_dir": str(uniqueness_dir),
        "novelty_dir": str(novelty_dir),
        "sample_roots": sample_roots_from_summaries(
            uniqueness_summary,
            novelty_summary,
        ),
        "merge_strategy": {
            **merge_info,
            "strict_merge": bool(args.strict_merge),
            "primary_strategy": "resolved_absolute_path",
        },
        "unmatched_uniqueness_rows": merge_info.get("unmatched_uniqueness_rows", 0),
        "unmatched_novelty_rows": merge_info.get("unmatched_novelty_rows", 0),
        "matcher_parameters_uniqueness": uniqueness_summary.get("matcher_parameters", {}),
        "matcher_parameters_novelty": novelty_summary.get("matcher_parameters", {}),
        "reference_structure_count": novelty_summary.get("reference_structure_count"),
        "reference_split": novelty_summary.get("reference_split"),
        "timestamp": now_iso(),
        "runtime_seconds": time.time() - start_time,
        "mode": mode,
        "input_row_counts": {
            "uniqueness_per_sample_rows": len(merged_rows)
            + int(merge_info.get("unmatched_uniqueness_rows", 0)),
            "novelty_per_sample_rows": len(merged_rows)
            + int(merge_info.get("unmatched_novelty_rows", 0)),
            "merged_rows": len(merged_rows),
        },
        "failure_count": len(failures),
        "inconsistent_evaluated_rows": sum(
            1 for failure in failures if failure.get("kind") == "inconsistent_evaluated_flag"
        ),
        "formula_mismatch_rows": sum(
            1 for failure in failures if failure.get("kind") == "formula_mismatch"
        ),
        "crystaldit_style_definition": (
            "UN Rate = count(unique_representative=True and novel=True) / "
            "evaluated_count. The auxiliary novel_among_unique_rate is reported "
            "separately and is not the primary UN Rate."
        ),
        "novelty_summary_rate_note": (
            "The existing novelty_summary.json reports novelty over unique groups. "
            "This UN script recomputes per-sample novelty_rate from merged "
            "novelty_per_sample.csv rows for the UN denominator."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CrystalDiT-style UN Rate from generated CIFs."
    )
    parser.add_argument("--sample-glob", default="./outputs/sample_LF_mp20_2026*")
    parser.add_argument("--sample-dir", default="")
    parser.add_argument("--mp20-root", default="./mp20")
    parser.add_argument("--processed-dir", default="./mp20/precessed")
    parser.add_argument("--uniqueness-dir", default="")
    parser.add_argument("--novelty-dir", default="")
    parser.add_argument("--output-dir", default="./outputs/un_rate_eval")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--valid-only", type=parse_bool, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-per-sample", type=parse_bool, default=True)
    parser.add_argument("--strict-merge", type=parse_bool, default=False)
    parser.add_argument("--recompute-missing", type=parse_bool, default=True)
    parser.add_argument("--tmp-dir", default="")
    parser.add_argument("--save-un-samples", type=parse_bool, default=True)
    parser.add_argument(
        "--un-sample-dir",
        default="",
        help=(
            "Directory for copied unique_and_novel CIF files. Defaults to "
            "<output-dir>/unique_novel_cifs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()
    output_dir = Path(args.output_dir).expanduser()
    ensure_output_dir(output_dir)

    uniqueness_dir, novelty_dir, mode, failures = resolve_eval_dirs(args, output_dir)
    if failures:
        summary = {
            "mode": mode,
            "uniqueness_dir": str(uniqueness_dir),
            "novelty_dir": str(novelty_dir),
            "evaluated_count": 0,
            "unique_count": 0,
            "novel_count": 0,
            "un_count": 0,
            "un_rate": None,
            "timestamp": now_iso(),
            "runtime_seconds": time.time() - start_time,
            "failure_count": len(failures),
        }
        write_json(output_dir / "un_rate_summary.json", summary)
        write_jsonl(output_dir / "un_rate_failures.jsonl", failures)
        return 1

    uniqueness_per_sample = uniqueness_dir / UNIQUE_PER_SAMPLE
    novelty_per_sample = novelty_dir / NOVELTY_PER_SAMPLE
    uniqueness_summary = load_json(uniqueness_dir / UNIQUE_SUMMARY)
    novelty_summary = load_json(novelty_dir / NOVELTY_SUMMARY)
    uniqueness_rows = read_csv_rows(uniqueness_per_sample)
    novelty_rows = read_csv_rows(novelty_per_sample)

    failures.extend(parse_failure_rows(uniqueness_rows, "uniqueness"))
    failures.extend(parse_failure_rows(novelty_rows, "novelty"))

    merged_rows, merge_failures, merge_info = merge_rows(uniqueness_rows, novelty_rows)
    failures.extend(merge_failures)

    un_structures = build_un_structures(merged_rows)
    un_sample_export, export_failures = export_un_samples(
        args,
        output_dir,
        un_structures,
    )
    failures.extend(export_failures)
    summary = build_summary(
        args,
        start_time,
        mode,
        uniqueness_dir,
        novelty_dir,
        uniqueness_summary,
        novelty_summary,
        merged_rows,
        merge_info,
        failures,
        un_sample_export,
    )

    write_json(output_dir / "un_rate_summary.json", summary)
    if args.save_per_sample:
        write_per_sample_csv(output_dir / "un_rate_per_sample.csv", merged_rows)
    write_jsonl(output_dir / "un_structures.jsonl", un_structures)
    write_jsonl(output_dir / "un_rate_failures.jsonl", failures)

    print(
        "UN Rate:",
        summary["un_rate"],
        f"({summary['un_count']}/{summary['evaluated_count']})",
    )
    print(f"Wrote results to {output_dir}")

    if args.strict_merge and (
        summary["unmatched_uniqueness_rows"] or summary["unmatched_novelty_rows"]
    ):
        print("Strict merge failed because unmatched rows were found.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
