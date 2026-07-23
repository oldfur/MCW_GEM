#!/usr/bin/env python
"""Batch all-metrics evaluation and baseline comparison table generation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_all_metrics_from_cifs.py"
SUMMARY_NAME = "all_metrics_summary.json"
COMPARISON_CSV = "baseline_all_metrics_comparison.csv"
COMPARISON_JSON = "baseline_all_metrics_comparison.json"
MANIFEST_CSV = "baseline_all_metrics_manifest.csv"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=json_default)


def safe_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return text.strip("._-") or "model"


def pretty_model_name(directory_name: str) -> str:
    name = directory_name
    for suffix in ("_crystals_mp20", "_crystals", "-crystals"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def count_cifs(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*.cif") if path.is_file())


def discover_baseline_models(baseline_root: Path) -> List[Dict[str, Any]]:
    models = []
    for child in sorted(baseline_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        cif_count = count_cifs(child)
        if cif_count == 0:
            continue
        models.append(
            {
                "model": pretty_model_name(child.name),
                "source_dir": child,
                "source_dir_name": child.name,
                "cif_count": cif_count,
                "kind": "baseline",
            }
        )
    return models


def parse_model_spec(spec: str) -> Dict[str, Any]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected MODEL=DIR for --model, got {spec!r}"
        )
    name, path_text = spec.split("=", 1)
    path = Path(path_text).expanduser()
    return {
        "model": name.strip(),
        "source_dir": path,
        "source_dir_name": path.name,
        "cif_count": count_cifs(path) if path.exists() else 0,
        "kind": "extra",
    }


def model_output_dir(output_root: Path, model: str) -> Path:
    return output_root / safe_name(model)


def build_eval_command(
    args: argparse.Namespace,
    model: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    cmd = [
        args.python_exec,
        "-u",
        str(EVALUATOR),
        "--sample-dir",
        str(model["source_dir"]),
        "--mp20-root",
        args.mp20_root,
        "--processed-dir",
        args.processed_dir,
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(args.num_workers),
        "--valid-only",
        str(bool(args.valid_only)),
        "--compute-novelty",
        str(bool(args.compute_novelty)),
        "--save-per-sample",
        str(bool(args.save_per_sample)),
        "--save-un-samples",
        str(bool(args.save_un_samples)),
        "--reference-split",
        args.reference_split,
        "--train-seed",
        str(args.train_seed),
        "--num-train",
        str(args.num_train),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    return cmd


def run_one_model(
    args: argparse.Namespace,
    model: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    log_path = output_dir / "all_metrics.log"

    if summary_path.exists() and not args.force:
        return {
            "model": model["model"],
            "status": "skipped_existing",
            "returncode": 0,
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "runtime_seconds": 0.0,
        }

    cmd = build_eval_command(args, model, output_dir)
    if args.dry_run:
        return {
            "model": model["model"],
            "status": "dry_run",
            "returncode": 0,
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "command": cmd,
            "runtime_seconds": 0.0,
        }

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return {
        "model": model["model"],
        "status": "ok" if proc.returncode == 0 and summary_path.exists() else "failed",
        "returncode": proc.returncode,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "command": cmd,
        "runtime_seconds": time.time() - start,
    }


def load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_value(summary: Optional[Dict[str, Any]], key: str) -> Any:
    if not summary:
        return ""
    return summary.get(key, "")


def comparison_row(
    model: Dict[str, Any],
    run_info: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    summary = load_summary(output_dir / SUMMARY_NAME)
    return {
        "model": model["model"],
        "kind": model["kind"],
        "source_dir": str(model["source_dir"]),
        "source_cif_count": model["cif_count"],
        "eval_output_dir": str(output_dir),
        "status": run_info.get("status", ""),
        "returncode": run_info.get("returncode", ""),
        "total_cif_files_found": summary_value(summary, "total_cif_files_found"),
        "total_cif_files_parsed": summary_value(summary, "total_cif_files_parsed"),
        "parse_failure_count": summary_value(summary, "parse_failure_count"),
        "structural_valid_count": summary_value(summary, "structural_valid_count"),
        "structural_valid_rate": summary_value(summary, "structural_valid_rate"),
        "composition_valid_count": summary_value(summary, "composition_valid_count"),
        "composition_valid_rate": summary_value(summary, "composition_valid_rate"),
        "total_valid_count": summary_value(summary, "total_valid_count"),
        "total_valid_rate": summary_value(summary, "total_valid_rate"),
        "evaluated_count": summary_value(summary, "evaluated_count"),
        "unique_count": summary_value(summary, "unique_count"),
        "unique_rate": summary_value(summary, "unique_rate"),
        "novel_count": summary_value(summary, "novel_count"),
        "novel_rate": summary_value(summary, "novel_rate"),
        "novel_valid_sample_count": summary_value(summary, "novel_valid_sample_count"),
        "novel_sample_rate": summary_value(summary, "novel_sample_rate"),
        "unique_and_novel_count": summary_value(summary, "unique_and_novel_count"),
        "UN_rate": summary_value(summary, "UN_rate"),
        "failure_count": summary_value(summary, "failure_count"),
        "eval_runtime_seconds": summary_value(summary, "runtime_seconds"),
        "batch_runtime_seconds": run_info.get("runtime_seconds", ""),
        "summary_path": str(output_dir / SUMMARY_NAME),
        "log_path": run_info.get("log_path", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "model",
            "kind",
            "source_dir",
            "source_cif_count",
            "eval_output_dir",
            "status",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_models(args: argparse.Namespace) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    if args.discover_baselines:
        baseline_root = Path(args.baseline_root).expanduser()
        models.extend(discover_baseline_models(baseline_root))
    for spec in args.model:
        models.append(parse_model_spec(spec))
    if args.ours_dir:
        ours_dir = Path(args.ours_dir).expanduser()
        models.append(
            {
                "model": args.ours_name,
                "source_dir": ours_dir,
                "source_dir_name": ours_dir.name,
                "cif_count": count_cifs(ours_dir) if ours_dir.exists() else 0,
                "kind": "ours",
            }
        )

    seen = set()
    deduped = []
    for model in models:
        key = str(Path(model["source_dir"]).expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(model)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all_metrics for baseline generated CIF directories and "
            "write a model comparison table."
        )
    )
    parser.add_argument(
        "--baseline-root",
        default="~/下载/baseline_all_generate_crystals/Generate_crystals",
        help="Root containing one generated-CIF directory per baseline model.",
    )
    parser.add_argument(
        "--discover-baselines",
        type=parse_bool,
        default=True,
        help="Discover immediate child directories under --baseline-root.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Extra model in MODEL=DIR form. Can be repeated.",
    )
    parser.add_argument(
        "--ours-dir",
        default="./outputs/sample_LF_condL_epoch220_10000_interleaved_zbl_softZ",
        help="Generated CIF directory for our model. Empty string disables ours.",
    )
    parser.add_argument("--ours-name", default="ours")
    parser.add_argument(
        "--output-root",
        default="./outputs/baseline_all_metrics_eval",
        help="Root where per-model evaluations and comparison table are written.",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--mp20-root", default="./mp20")
    parser.add_argument("--processed-dir", default="./mp20/precessed")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--valid-only", type=parse_bool, default=True)
    parser.add_argument("--compute-novelty", type=parse_bool, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-per-sample", type=parse_bool, default=True)
    parser.add_argument("--save-un-samples", type=parse_bool, default=False)
    parser.add_argument(
        "--reference-split",
        default="existing_eval",
        choices=["existing_eval", "all", "train"],
    )
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--num-train", type=int, default=27138)
    parser.add_argument("--force", type=parse_bool, default=False)
    parser.add_argument("--dry-run", type=parse_bool, default=False)
    parser.add_argument("--fail-fast", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    models = build_models(args)
    if not models:
        raise RuntimeError("No model directories were found.")

    manifest_rows = []
    comparison_rows = []
    failures = 0
    print(f"Found {len(models)} model directory/directories.")

    for index, model in enumerate(models):
        source_dir = Path(model["source_dir"]).expanduser()
        output_dir = model_output_dir(output_root, str(model["model"]))
        model["source_dir"] = source_dir
        model["cif_count"] = count_cifs(source_dir) if source_dir.exists() else 0
        print(
            f"[{index + 1}/{len(models)}] {model['model']}: "
            f"{model['cif_count']} CIFs -> {output_dir}"
        )

        run_info = run_one_model(args, model, output_dir)
        if run_info["status"] == "failed":
            failures += 1
            print(f"  failed, see {run_info.get('log_path')}")
            if args.fail_fast:
                manifest_rows.append({**model, **run_info, "eval_output_dir": output_dir})
                break
        else:
            print(f"  {run_info['status']}")

        manifest_rows.append({**model, **run_info, "eval_output_dir": output_dir})
        comparison_rows.append(comparison_row(model, run_info, output_dir))

    write_csv(output_root / COMPARISON_CSV, comparison_rows)
    write_csv(output_root / MANIFEST_CSV, manifest_rows)
    write_json(
        output_root / COMPARISON_JSON,
        {
            "timestamp": now_iso(),
            "output_root": str(output_root),
            "comparison_csv": str(output_root / COMPARISON_CSV),
            "manifest_csv": str(output_root / MANIFEST_CSV),
            "model_count": len(models),
            "failure_count": failures,
            "rows": comparison_rows,
        },
    )

    print(f"Wrote comparison table to {output_root / COMPARISON_CSV}")
    print(f"Wrote manifest to {output_root / MANIFEST_CSV}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
