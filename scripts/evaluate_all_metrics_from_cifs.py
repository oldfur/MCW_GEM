#!/usr/bin/env python
"""Unified offline evaluation for generated CIF directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp20.novelty import (  # noqa: E402
    DEFAULT_MATCHER_PARAMS,
    DEFAULT_SKIP_STRUCTURE_REDUCTION,
)
from evaluate_uniqueness_from_cifs import (  # noqa: E402
    clean_csv_row,
    cluster_candidate_structures,
    collect_sample_files,
    load_sample_rows,
    matcher_internal_options,
    parse_bool,
)
from evaluate_novelty_from_cifs import (  # noqa: E402
    bucket_reference_records,
    evaluate_group_novelty,
    load_reference_records,
    read_reference_inputs,
    resolve_processed_dir,
    resolve_reference_csv,
)
from evaluate_un_rate_from_cifs import (  # noqa: E402
    build_un_structures,
    export_un_samples,
)


SUMMARY_JSON = "all_metrics_summary.json"
PER_SAMPLE_CSV = "all_metrics_per_sample.csv"
CLUSTERS_JSONL = "unique_clusters.jsonl"
MATCHED_PAIRS_JSONL = "matched_pairs.jsonl"
UN_STRUCTURES_JSONL = "un_structures.jsonl"
FAILURES_JSONL = "all_metrics_failures.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def rate(count: Optional[int], total: Optional[int]) -> Optional[float]:
    if count is None or total is None or total == 0:
        return None
    return float(count / total)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=json_default) + "\n")


def row_bool(row: Dict[str, Any], key: str) -> bool:
    return bool(row.get("parse_success") and row.get(key))


def sample_rows_by_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["sample_index"]): row for row in rows if row.get("parse_success")}


def build_novelty_tasks(
    clusters: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int, Any]], List[Dict[str, Any]]]:
    rows_by_sample_index = sample_rows_by_index(sample_rows)
    tasks: List[Tuple[int, int, Any]] = []
    group_members: List[Dict[str, Any]] = []

    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        representative_sample_index = int(cluster["representative_index"])
        representative_row = rows_by_sample_index.get(representative_sample_index)
        if representative_row is None or "structure" not in representative_row:
            continue
        member_sample_indices = [int(value) for value in cluster.get("member_indices", [])]
        tasks.append(
            (
                cluster_id,
                representative_sample_index,
                representative_row["structure"],
            )
        )
        group_members.append(
            {
                "group_index": cluster_id,
                "representative_sample_index": representative_sample_index,
                "member_sample_indices": member_sample_indices,
            }
        )

    return tasks, group_members


def add_cluster_sizes_to_rows(
    sample_rows: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
) -> None:
    sizes_by_cluster = {
        int(cluster["cluster_id"]): int(cluster["cluster_size"])
        for cluster in clusters
    }
    for row in sample_rows:
        cluster_id = row.get("cluster_id", "")
        if cluster_id == "":
            row["unique_group_size"] = ""
            continue
        row["unique_group_size"] = sizes_by_cluster.get(int(cluster_id), "")


def add_novelty_to_clusters(
    clusters: List[Dict[str, Any]],
    group_results: List[Dict[str, Any]],
) -> None:
    results_by_cluster = {
        int(result["group_index"]): result for result in group_results
    }
    for cluster in clusters:
        result = results_by_cluster.get(int(cluster["cluster_id"]), {})
        cluster["novel"] = result.get("novel", "")
        cluster["matched_reference_index"] = result.get("matched_reference_index", "")
        cluster["matched_reference_id"] = result.get("matched_reference_id", "")
        cluster["matched_reference_path"] = result.get("matched_reference_path", "")
        cluster["reference_bucket_size"] = result.get("reference_bucket_size", "")
        cluster["novelty_error_message"] = result.get("error_message", "")


def apply_novelty_results_to_rows(
    sample_rows: List[Dict[str, Any]],
    group_members: List[Dict[str, Any]],
    group_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    members_by_group = {
        int(group["group_index"]): group["member_sample_indices"]
        for group in group_members
    }
    rows_by_sample_index = sample_rows_by_index(sample_rows)
    failures: List[Dict[str, Any]] = []
    matched_pairs: List[Dict[str, Any]] = []

    for result in sorted(group_results, key=lambda item: int(item["group_index"])):
        group_index = int(result["group_index"])
        member_indices = members_by_group.get(group_index, [])
        if result.get("error_message"):
            failure = dict(result)
            failure["kind"] = "novelty_matcher_failure"
            failures.append(failure)
        for sample_index in member_indices:
            row = rows_by_sample_index[sample_index]
            row["novelty_evaluated"] = not bool(result.get("error_message"))
            row["novel"] = result.get("novel", "")
            row["matched_reference_index"] = result.get("matched_reference_index", "")
            row["matched_reference_id"] = result.get("matched_reference_id", "")
            row["matched_reference_path"] = result.get("matched_reference_path", "")
            row["novelty_error_message"] = result.get("error_message", "")
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


def add_un_flags_to_rows(sample_rows: List[Dict[str, Any]]) -> None:
    for row in sample_rows:
        evaluated = bool(row.get("evaluated"))
        unique_representative = bool(row.get("unique_representative"))
        novel = row.get("novel")
        row["unique_and_novel"] = bool(
            evaluated and unique_representative and novel is True
        )


def build_unified_rows(sample_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in sample_rows:
        cleaned = clean_csv_row(row)
        rows.append(
            {
                "sample_index": cleaned.get("sample_index", ""),
                "cif_path": cleaned.get("cif_path", ""),
                "sample_root": cleaned.get("sample_root", ""),
                "relative_path": cleaned.get("relative_path", ""),
                "worker": cleaned.get("worker", ""),
                "filename": cleaned.get("filename", ""),
                "parse_success": cleaned.get("parse_success", ""),
                "comp_valid": cleaned.get("comp_valid", ""),
                "struct_valid": cleaned.get("struct_valid", ""),
                "valid": cleaned.get("valid", ""),
                "evaluated": cleaned.get("evaluated", ""),
                "unique_representative": cleaned.get("unique_representative", ""),
                "unique_and_novel": cleaned.get("unique_and_novel", ""),
                "cluster_id": cleaned.get("cluster_id", ""),
                "unique_group_size": cleaned.get("unique_group_size", ""),
                "novelty_evaluated": cleaned.get("novelty_evaluated", ""),
                "matched_representative_index": cleaned.get(
                    "matched_representative_index", ""
                ),
                "matched_representative_path": cleaned.get(
                    "matched_representative_path", ""
                ),
                "novel": cleaned.get("novel", ""),
                "matched_reference_index": cleaned.get("matched_reference_index", ""),
                "matched_reference_id": cleaned.get("matched_reference_id", ""),
                "matched_reference_path": cleaned.get("matched_reference_path", ""),
                "formula": cleaned.get("formula", ""),
                "reduced_formula": cleaned.get("reduced_formula", ""),
                "num_sites": cleaned.get("num_sites", ""),
                "error_message": " | ".join(
                    str(value).strip()
                    for value in (
                        cleaned.get("error_message", ""),
                        cleaned.get("novelty_error_message", ""),
                    )
                    if str(value).strip()
                ),
            }
        )
    return rows


def write_per_sample_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "cif_path",
        "sample_root",
        "relative_path",
        "worker",
        "filename",
        "parse_success",
        "comp_valid",
        "struct_valid",
        "valid",
        "evaluated",
        "unique_representative",
        "unique_and_novel",
        "cluster_id",
        "unique_group_size",
        "novelty_evaluated",
        "matched_representative_index",
        "matched_representative_path",
        "novel",
        "matched_reference_index",
        "matched_reference_id",
        "matched_reference_path",
        "formula",
        "reduced_formula",
        "num_sites",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_failures(sample_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = []
    for row in sample_rows:
        if row.get("parse_success"):
            continue
        failures.append(
            {
                "kind": "sample_parse_failure",
                **clean_csv_row(row),
                "traceback": row.get("traceback", ""),
            }
        )
    return failures


def validity_failures(sample_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = []
    for row in sample_rows:
        if not row.get("parse_success") or row.get("valid"):
            continue
        failures.append(
            {
                "kind": "validity_check_failure",
                **clean_csv_row(row),
            }
        )
    return failures


def load_references(args: argparse.Namespace, output_dir: Path) -> Tuple[
    List[Any],
    Dict[str, List[Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    mp20_root = Path(args.mp20_root).expanduser()
    processed_dir, processed_warnings = resolve_processed_dir(
        mp20_root,
        Path(args.processed_dir).expanduser(),
    )
    reference_csv = resolve_reference_csv(mp20_root, processed_dir)
    reference_inputs, reference_split, reference_split_uncertain, split_warnings = (
        read_reference_inputs(
            reference_csv,
            args.reference_split,
            args.train_seed,
            args.num_train,
        )
    )
    reference_records, reference_failures = load_reference_records(
        reference_inputs,
        max(1, int(args.num_workers)),
    )
    if not reference_records:
        write_json(
            output_dir / "reference_load_failure.json",
            {
                "reference_source": str(reference_csv),
                "reference_failures": reference_failures,
            },
        )
        raise RuntimeError("No reference structures could be parsed.")
    reference_buckets = bucket_reference_records(reference_records)
    info = {
        "reference_structure_count": len(reference_records),
        "reference_parse_failures": len(reference_failures),
        "reference_split": reference_split,
        "reference_split_uncertain": reference_split_uncertain,
        "reference_source": str(reference_csv),
        "processed_dir": str(processed_dir) if processed_dir else "",
        "reference_bucket_count": len(reference_buckets),
        "warnings": processed_warnings + split_warnings,
    }
    return reference_records, reference_buckets, reference_failures, info


def build_summary(
    args: argparse.Namespace,
    start: float,
    samples_found: int,
    sample_roots: List[Path],
    sample_rows: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    group_results: List[Dict[str, Any]],
    reference_info: Dict[str, Any],
    failures: List[Dict[str, Any]],
    un_sample_export: Dict[str, Any],
) -> Dict[str, Any]:
    parsed_count = sum(1 for row in sample_rows if row.get("parse_success"))
    comp_valid_count = sum(1 for row in sample_rows if row_bool(row, "comp_valid"))
    struct_valid_count = sum(1 for row in sample_rows if row_bool(row, "struct_valid"))
    total_valid_count = sum(1 for row in sample_rows if row_bool(row, "valid"))
    evaluated_count = sum(1 for row in sample_rows if row.get("evaluated"))
    unique_count = len(clusters)

    successful_group_results = [
        result for result in group_results if not result.get("error_message")
    ]
    novel_count = (
        sum(1 for result in successful_group_results if result.get("novel") is True)
        if bool(args.compute_novelty)
        else None
    )
    non_novel_count = (
        sum(1 for result in successful_group_results if result.get("novel") is False)
        if bool(args.compute_novelty)
        else None
    )
    novelty_evaluated_count = (
        len(successful_group_results) if bool(args.compute_novelty) else None
    )
    novel_valid_sample_count = (
        sum(1 for row in sample_rows if row.get("evaluated") and row.get("novel") is True)
        if bool(args.compute_novelty)
        else None
    )
    unique_and_novel_count = (
        sum(1 for row in sample_rows if row.get("unique_and_novel"))
        if bool(args.compute_novelty)
        else None
    )

    denominator_count = evaluated_count
    validity_denominator = samples_found

    return {
        "total_cif_files_found": samples_found,
        "total_cif_files_parsed": parsed_count,
        "parse_failure_count": samples_found - parsed_count,
        "composition_valid_count": comp_valid_count,
        "composition_valid_rate": rate(comp_valid_count, validity_denominator),
        "structural_valid_count": struct_valid_count,
        "structural_valid_rate": rate(struct_valid_count, validity_denominator),
        "total_valid_count": total_valid_count,
        "total_valid_rate": rate(total_valid_count, validity_denominator),
        "validity_rate_denominator": (
            "all .cif files found under --sample-dir; parse failures count as invalid"
        ),
        "evaluated_count": evaluated_count,
        "unique_count": unique_count,
        "duplicate_count": max(evaluated_count - unique_count, 0),
        "unique_rate": rate(unique_count, denominator_count),
        "novel_count": novel_count,
        "non_novel_count": non_novel_count,
        "novel_rate": rate(novel_count, novelty_evaluated_count),
        "novel_valid_sample_count": novel_valid_sample_count,
        "novel_sample_rate": rate(novel_valid_sample_count, denominator_count),
        "unique_and_novel_count": unique_and_novel_count,
        "UN_rate": rate(unique_and_novel_count, denominator_count),
        "un_rate": rate(unique_and_novel_count, denominator_count),
        "un_sample_export": un_sample_export,
        "metric_denominator_definition": (
            "unique_rate and UN_rate use evaluated_count: generated CIFs that "
            "parse successfully and satisfy mp20.crystal.array_dict_to_crystal(...).valid "
            "when --valid-only=True. novel_rate follows mp20.analyze_test and uses "
            "unique generated groups as denominator."
        ),
        "valid_only": bool(args.valid_only),
        "compute_novelty": bool(args.compute_novelty),
        "matcher_parameters": dict(DEFAULT_MATCHER_PARAMS),
        "matcher_internal_options": matcher_internal_options(dict(DEFAULT_MATCHER_PARAMS)),
        "matcher_fit_kwargs": {
            "skip_structure_reduction": DEFAULT_SKIP_STRUCTURE_REDUCTION,
        },
        "sample_dir": str(Path(args.sample_dir).expanduser()),
        "sample_roots": [str(root) for root in sample_roots],
        "max_samples": args.max_samples,
        "num_workers": max(1, int(args.num_workers)),
        "reference": reference_info,
        "failure_count": len(failures),
        "timestamp": now_iso(),
        "runtime_seconds": time.time() - start,
        "output_files": {
            "summary": SUMMARY_JSON,
            "per_sample": PER_SAMPLE_CSV if bool(args.save_per_sample) else "",
            "clusters": CLUSTERS_JSONL,
            "matched_pairs": MATCHED_PAIRS_JSONL if bool(args.compute_novelty) else "",
            "un_structures": UN_STRUCTURES_JSONL if bool(args.compute_novelty) else "",
            "failures": FAILURES_JSONL,
        },
        "existing_evaluator_audit": {
            "validity": (
                "CIFs are parsed with pymatgen and converted to the array_dict "
                "expected by mp20.crystal.array_dict_to_crystal. The reported "
                "comp_valid, struct_valid, and valid flags come from that Crystal object."
            ),
            "uniqueness": (
                "StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3).group_structures "
                "is applied to the evaluated generated structures."
            ),
            "novelty": (
                "One representative per unique generated group is compared with MP20 "
                "reference CIFs using matcher.fit(..., skip_structure_reduction=True)."
            ),
            "UN_rate": (
                "count(unique_representative=True and novel=True) / evaluated_count."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate struct valid, comp valid, total valid, Unique Rate, "
            "Novel Rate, and UN Rate from one generated CIF directory."
        )
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="Directory to scan recursively for generated .cif files.",
    )
    parser.add_argument("--mp20-root", default="./mp20")
    parser.add_argument("--processed-dir", default="./mp20/precessed")
    parser.add_argument("--output-dir", default="./outputs/all_metrics_eval")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--valid-only", type=parse_bool, default=True)
    parser.add_argument("--compute-novelty", type=parse_bool, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-per-sample", type=parse_bool, default=True)
    parser.add_argument("--save-un-samples", type=parse_bool, default=True)
    parser.add_argument(
        "--un-sample-dir",
        default="",
        help=(
            "Directory for copied unique_and_novel CIF files. Defaults to "
            "<output-dir>/unique_novel_cifs."
        ),
    )
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
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = max(1, int(args.num_workers))
    matcher_params = dict(DEFAULT_MATCHER_PARAMS)

    samples, sample_roots = collect_sample_files(
        sample_glob="",
        sample_dir=args.sample_dir,
        max_samples=args.max_samples,
    )
    print(f"Found {len(samples)} CIF files under {Path(args.sample_dir).expanduser()}.")
    print(
        "Using existing matcher: "
        f"StructureMatcher(stol={matcher_params['stol']}, "
        f"angle_tol={matcher_params['angle_tol']}, ltol={matcher_params['ltol']})"
    )

    sample_rows = load_sample_rows(samples, num_workers, check_validity=True)
    failures: List[Dict[str, Any]] = []
    failures.extend(parse_failures(sample_rows))
    failures.extend(validity_failures(sample_rows))

    clusters, cluster_failures = cluster_candidate_structures(
        sample_rows,
        valid_only=bool(args.valid_only),
        include_invalid_cifs=False,
        matcher_params=matcher_params,
    )
    failures.extend(cluster_failures)
    add_cluster_sizes_to_rows(sample_rows, clusters)
    print(f"Unique clusters: {len(clusters)}")

    group_results: List[Dict[str, Any]] = []
    matched_pairs: List[Dict[str, Any]] = []
    reference_info: Dict[str, Any] = {}

    if bool(args.compute_novelty):
        reference_records, reference_buckets, reference_failures, reference_info = (
            load_references(args, output_dir)
        )
        failures.extend(reference_failures)
        for warning in reference_info.get("warnings", []):
            print(f"Warning: {warning}")
        print(
            f"Loaded {len(reference_records)} reference structures in "
            f"{len(reference_buckets)} reduced-formula buckets."
        )

        tasks, group_members = build_novelty_tasks(clusters, sample_rows)
        group_results = evaluate_group_novelty(
            tasks,
            reference_buckets,
            matcher_params,
            DEFAULT_SKIP_STRUCTURE_REDUCTION,
            num_workers,
        )
        novelty_failures, matched_pairs = apply_novelty_results_to_rows(
            sample_rows,
            group_members,
            group_results,
        )
        failures.extend(novelty_failures)
        add_novelty_to_clusters(clusters, group_results)
    else:
        for row in sample_rows:
            row["novel"] = ""

    add_un_flags_to_rows(sample_rows)
    unified_rows = build_unified_rows(sample_rows)
    un_structures = build_un_structures(unified_rows) if bool(args.compute_novelty) else []
    if bool(args.compute_novelty):
        un_sample_export, export_failures = export_un_samples(
            args,
            output_dir,
            un_structures,
        )
        failures.extend(export_failures)
    else:
        un_sample_export = {
            "enabled": False,
            "requested_count": 0,
            "exported_count": 0,
            "failure_count": 0,
            "reason": "compute_novelty=False",
        }

    summary = build_summary(
        args,
        start,
        samples_found=len(samples),
        sample_roots=sample_roots,
        sample_rows=sample_rows,
        clusters=clusters,
        group_results=group_results,
        reference_info=reference_info,
        failures=failures,
        un_sample_export=un_sample_export,
    )

    write_json(output_dir / SUMMARY_JSON, summary)
    if bool(args.save_per_sample):
        write_per_sample_csv(output_dir / PER_SAMPLE_CSV, unified_rows)
    write_jsonl(output_dir / CLUSTERS_JSONL, clusters)
    if bool(args.compute_novelty):
        write_jsonl(output_dir / MATCHED_PAIRS_JSONL, matched_pairs)
        write_jsonl(output_dir / UN_STRUCTURES_JSONL, un_structures)
    write_jsonl(output_dir / FAILURES_JSONL, failures)

    print(
        "Validity:",
        f"struct={summary['structural_valid_rate']}",
        f"comp={summary['composition_valid_rate']}",
        f"total={summary['total_valid_rate']}",
    )
    print(
        "Rates:",
        f"unique={summary['unique_rate']}",
        f"novel={summary['novel_rate']}",
        f"UN={summary['UN_rate']}",
    )
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
