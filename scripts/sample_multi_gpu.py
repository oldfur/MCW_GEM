#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path


def parse_gpu_list(value):
    gpu_ids = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        gpu_ids.append(token)
    if not gpu_ids:
        raise argparse.ArgumentTypeError("At least one GPU id is required, e.g. --gpus 0,1,2,3")
    return gpu_ids


def distribute_rounds(total_rounds, num_workers):
    base = total_rounds // num_workers
    remainder = total_rounds % num_workers
    return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def get_arg_value(argv, flag, default=None):
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return default


def set_arg(argv, flag, value):
    value = str(value)
    updated = []
    consumed = False
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == flag:
            updated.extend([flag, value])
            consumed = True
            idx += 2
            continue
        if token.startswith(flag + "="):
            updated.append(f"{flag}={value}")
            consumed = True
            idx += 1
            continue
        updated.append(token)
        idx += 1
    if not consumed:
        updated.extend([flag, value])
    return updated


def append_worker_suffix(exp_name, worker_index):
    suffix = f"_worker_{worker_index:02d}"
    if exp_name.endswith(suffix):
        return exp_name
    return f"{exp_name}{suffix}"


def stream_output(pipe, prefix):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"{prefix}{line}", end="")
    finally:
        pipe.close()


def load_sampling_metrics(metrics_path):
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sum_int(metrics_list, field):
    values = [metrics.get(field) for metrics in metrics_list if metrics.get(field) is not None]
    if not values:
        return None
    return int(sum(int(value) for value in values))


def _rate(count, total):
    if count is None:
        return None
    return float(count / total) if total else 0.0


def _write_component_metrics_csv(base_save_dir, payload):
    flat_fields = [
        "config_name", "num_requested", "num_generated", "num_finalized_cifs",
        "structural_valid_count", "structural_valid_rate",
        "composition_valid_count", "composition_valid_rate",
        "total_valid_count", "total_valid_rate",
        "unique_count", "novel_count", "unique_and_novel_count", "UN_rate",
        "all_H_count", "all_H_rate", "single_element_count", "single_element_rate",
        "close_contact_fail_count", "close_contact_fail_rate",
        "invalid_lattice_count", "invalid_lattice_rate",
        "search_failure_count", "search_failure_rate",
        "raw_argmax_mode", "constrained_search_mode",
        "geometry_correction", "geometry_correction_mode",
        "zbl_correction_inner_steps", "zbl_correction_step_size",
        "zbl_correction_r_cut", "zbl_correction_force_clip",
        "zbl_correction_max_step",
        "interleaved_zbl_steps", "interleaved_zbl_step_size",
        "interleaved_zbl_r_cut", "interleaved_zbl_force_clip",
        "interleaved_zbl_max_step", "interleaved_zbl_atomic_number_mode",
        "atom_decode_mode",
        "pre_correction_structural_valid_count", "pre_correction_structural_valid_rate",
        "post_correction_structural_valid_count", "post_correction_structural_valid_rate",
    ]
    row = {field: payload.get(field) for field in flat_fields}
    for filename in ("summary.csv", "metrics.csv"):
        path = base_save_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_fields)
            writer.writeheader()
            writer.writerow(row)


def aggregate_component_metrics(base_save_dir, worker_dirs, manifest):
    """Aggregate per-worker component diagnostics without touching sampling logic."""
    worker_payloads = []
    for worker_dir in worker_dirs:
        metrics_path = worker_dir / "metrics.json"
        metrics = load_json(metrics_path)
        if metrics is None:
            continue
        worker_payloads.append(
            {
                "worker_dir": str(worker_dir),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )

    if not worker_payloads:
        return None

    metrics_list = [payload["metrics"] for payload in worker_payloads]
    first = metrics_list[0]
    num_generated = _sum_int(metrics_list, "num_generated") or 0
    num_requested = _sum_int(metrics_list, "num_requested") or 0
    constrained_mode = bool(first.get("constrained_search_mode"))
    raw_argmax_mode = bool(first.get("raw_argmax_mode"))

    payload = {
        "config_name": first.get("config_name") or base_save_dir.name,
        "num_requested": num_requested,
        "num_generated": num_generated,
        "num_finalized_cifs": _sum_int(metrics_list, "num_finalized_cifs") or 0,
        "structural_valid_count": _sum_int(metrics_list, "structural_valid_count") or 0,
        "composition_valid_count": _sum_int(metrics_list, "composition_valid_count") or 0,
        "total_valid_count": _sum_int(metrics_list, "total_valid_count") or 0,
        "unique_count": None,
        "novel_count": None,
        "unique_and_novel_count": None,
        "UN_rate": None,
        "all_H_count": _sum_int(metrics_list, "all_H_count") or 0,
        "single_element_count": _sum_int(metrics_list, "single_element_count") or 0,
        "close_contact_fail_count": _sum_int(metrics_list, "close_contact_fail_count") or 0,
        "invalid_lattice_count": _sum_int(metrics_list, "invalid_lattice_count") or 0,
        "search_failure_count": _sum_int(metrics_list, "search_failure_count") if constrained_mode else None,
        "raw_argmax_mode": raw_argmax_mode,
        "constrained_search_mode": constrained_mode,
        "geometry_correction": bool(first.get("geometry_correction")),
        "geometry_correction_mode": first.get("geometry_correction_mode", "default"),
        "zbl_correction_inner_steps": first.get("zbl_correction_inner_steps"),
        "zbl_correction_step_size": first.get("zbl_correction_step_size"),
        "zbl_correction_r_cut": first.get("zbl_correction_r_cut"),
        "zbl_correction_force_clip": first.get("zbl_correction_force_clip"),
        "zbl_correction_max_step": first.get("zbl_correction_max_step"),
        "interleaved_zbl_steps": first.get("interleaved_zbl_steps"),
        "interleaved_zbl_step_size": first.get("interleaved_zbl_step_size"),
        "interleaved_zbl_r_cut": first.get("interleaved_zbl_r_cut"),
        "interleaved_zbl_force_clip": first.get("interleaved_zbl_force_clip"),
        "interleaved_zbl_max_step": first.get("interleaved_zbl_max_step"),
        "interleaved_zbl_atomic_number_mode": first.get(
            "interleaved_zbl_atomic_number_mode",
            "masked_probability_expectation",
        ),
        "atom_decode_mode": first.get("atom_decode_mode"),
        "pre_correction_structural_valid_count": _sum_int(metrics_list, "pre_correction_structural_valid_count"),
        "post_correction_structural_valid_count": _sum_int(metrics_list, "post_correction_structural_valid_count") or 0,
        "unknown_class_masked": bool(first.get("unknown_class_masked", True)),
        "multi_gpu_worker_aggregate": True,
        "multi_gpu_note": (
            "Counts/rates are aggregated across workers. Global uniqueness/novelty/UN are "
            "not recomputed here; use the offline UN evaluation for final paper numbers."
        ),
        "worker_metrics": [
            {
                "worker_dir": payload["worker_dir"],
                "metrics_path": payload["metrics_path"],
                "num_generated": payload["metrics"].get("num_generated"),
                "num_finalized_cifs": payload["metrics"].get("num_finalized_cifs"),
            }
            for payload in worker_payloads
        ],
    }

    payload["structural_valid_rate"] = _rate(payload["structural_valid_count"], num_generated)
    payload["composition_valid_rate"] = _rate(payload["composition_valid_count"], num_generated)
    payload["total_valid_rate"] = _rate(payload["total_valid_count"], num_generated)
    payload["all_H_rate"] = _rate(payload["all_H_count"], num_generated)
    payload["single_element_rate"] = _rate(payload["single_element_count"], num_generated)
    payload["close_contact_fail_rate"] = _rate(payload["close_contact_fail_count"], num_generated)
    payload["invalid_lattice_rate"] = _rate(payload["invalid_lattice_count"], num_generated)
    payload["search_failure_rate"] = (
        _rate(payload["search_failure_count"], num_generated) if constrained_mode else None
    )
    payload["pre_correction_structural_valid_rate"] = _rate(
        payload["pre_correction_structural_valid_count"], num_generated
    )
    payload["post_correction_structural_valid_rate"] = _rate(
        payload["post_correction_structural_valid_count"], num_generated
    )

    guard_fields = [
        "search_failure_count",
        "search_guard_call_count",
        "raw_all_H_count_from_logits",
        "final_all_H_count_from_logits",
        "final_composition_valid_count_from_decode",
    ]
    payload["guard_summary"] = {
        field: int(sum(int((metrics.get("guard_summary") or {}).get(field, 0)) for metrics in metrics_list))
        for field in guard_fields
    }
    payload["geometry_pre_correction_summary_by_worker"] = [
        metrics.get("geometry_pre_correction_summary") for metrics in metrics_list
    ]

    run_config = dict(first.get("run_config") or {})
    run_config.update(
        {
            "config_name": payload["config_name"],
            "num_samples": num_requested,
            "num_rounds": int(manifest.get("num_rounds", 0)),
            "save_dir": str(base_save_dir),
            "multi_gpu": True,
            "gpus": manifest.get("gpus", []),
            "worker_dirs": [str(worker_dir) for worker_dir in worker_dirs],
            "command_line": " ".join(sys.argv),
        }
    )
    payload["run_config"] = run_config

    with (base_save_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, allow_nan=False)
    with (base_save_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=True, indent=2, allow_nan=False)
    _write_component_metrics_csv(base_save_dir, payload)
    return payload


def summarize_workers(base_save_dir, worker_dirs, exit_codes):
    from debug_all_h_samples import summarize_cifs, summarize_debug_dir

    total_samples = 0
    total_all_h = 0
    total_single_element = 0
    total_failures = 0
    total_debug_all_h_rows = 0
    element_frequency = Counter()
    all_h_files = []
    per_worker = []
    total_metric_samples = 0
    total_valid_count = 0
    total_comp_valid_count = 0
    total_struct_valid_count = 0
    total_metrics_workers = 0

    for worker_dir, exit_code in zip(worker_dirs, exit_codes):
        sample_dir = worker_dir / "epoch_0"
        debug_dir = worker_dir / "atom_type_debug"
        metrics_path = sample_dir / "sampling_metrics.json"
        cif_summary = summarize_cifs(sample_dir)
        debug_summary = summarize_debug_dir(debug_dir)
        sampling_metrics = load_sampling_metrics(metrics_path)
        worker_total = len(cif_summary["rows"])
        worker_all_h = sum(1 for row in cif_summary["rows"] if row["all_h"])
        worker_single_element = sum(1 for row in cif_summary["rows"] if row["single_element"])
        worker_failures = len(cif_summary["failures"])

        total_samples += worker_total
        total_all_h += worker_all_h
        total_single_element += worker_single_element
        total_failures += worker_failures
        total_debug_all_h_rows += len(debug_summary["all_h_rows"])
        element_frequency.update(cif_summary["element_frequency"])
        all_h_files.extend(cif_summary["all_h_files"])

        if sampling_metrics is not None:
            total_metrics_workers += 1
            total_metric_samples += int(sampling_metrics.get("total_samples", 0))
            total_valid_count += int(sampling_metrics.get("valid_count", 0))
            total_comp_valid_count += int(sampling_metrics.get("comp_valid_count", 0))
            total_struct_valid_count += int(sampling_metrics.get("struct_valid_count", 0))

        worker_payload = {
            "worker_dir": str(worker_dir),
            "exit_code": int(exit_code),
            "sample_dir": str(sample_dir),
            "debug_dir": str(debug_dir),
            "metrics_path": str(metrics_path),
            "total_samples": worker_total,
            "all_h_samples": worker_all_h,
            "single_element_samples": worker_single_element,
            "cif_parse_failures": worker_failures,
            "debug_all_h_rows": len(debug_summary["all_h_rows"]),
        }
        if sampling_metrics is not None:
            worker_payload.update(
                {
                    "metric_total_samples": int(sampling_metrics.get("total_samples", 0)),
                    "valid_samples": int(sampling_metrics.get("valid_count", 0)),
                    "valid_rate": sampling_metrics.get("valid_rate_mean"),
                    "comp_valid_samples": int(sampling_metrics.get("comp_valid_count", 0)),
                    "comp_valid_rate": sampling_metrics.get("comp_valid_rate_mean"),
                    "struct_valid_samples": int(sampling_metrics.get("struct_valid_count", 0)),
                    "struct_valid_rate": sampling_metrics.get("struct_valid_rate_mean"),
                    "unique_rate": sampling_metrics.get("unique_rate"),
                    "novel_rate": sampling_metrics.get("novel_rate"),
                }
            )
        per_worker.append(worker_payload)

    aggregate_valid_rate = (
        float(total_valid_count / total_metric_samples) if total_metric_samples > 0 else None
    )
    aggregate_comp_valid_rate = (
        float(total_comp_valid_count / total_metric_samples) if total_metric_samples > 0 else None
    )
    aggregate_struct_valid_rate = (
        float(total_struct_valid_count / total_metric_samples) if total_metric_samples > 0 else None
    )

    summary = {
        "base_save_dir": str(base_save_dir),
        "total_samples": total_samples,
        "all_h_samples": total_all_h,
        "single_element_samples": total_single_element,
        "cif_parse_failures": total_failures,
        "debug_all_h_rows": total_debug_all_h_rows,
        "element_frequency": dict(sorted(element_frequency.items())),
        "all_h_files": all_h_files,
        "metric_total_samples": total_metric_samples,
        "valid_samples": total_valid_count,
        "valid_rate": aggregate_valid_rate,
        "comp_valid_samples": total_comp_valid_count,
        "comp_valid_rate": aggregate_comp_valid_rate,
        "struct_valid_samples": total_struct_valid_count,
        "struct_valid_rate": aggregate_struct_valid_rate,
        "workers_with_sampling_metrics": total_metrics_workers,
        "workers": per_worker,
    }

    print("\n=== Multi-GPU Sampling Summary ===")
    print(f"Base save dir: {base_save_dir}")
    print(f"Total CIF samples: {total_samples}")
    print(f"All-H CIF samples: {total_all_h}")
    print(f"Single-element CIF samples: {total_single_element}")
    print(f"CIF parse failures: {total_failures}")
    print(f"Debug all-H rows: {total_debug_all_h_rows}")
    if total_metric_samples > 0:
        print(
            "Validity summary: "
            f"valid={total_valid_count}/{total_metric_samples} ({aggregate_valid_rate:.4f}), "
            f"comp={total_comp_valid_count}/{total_metric_samples} ({aggregate_comp_valid_rate:.4f}), "
            f"struct={total_struct_valid_count}/{total_metric_samples} ({aggregate_struct_valid_rate:.4f})"
        )
    for worker in per_worker:
        worker_line = (
            f"  - {Path(worker['worker_dir']).name}: exit={worker['exit_code']} "
            f"samples={worker['total_samples']} all_H={worker['all_h_samples']} "
            f"single_element={worker['single_element_samples']}"
        )
        if "valid_samples" in worker:
            worker_line += (
                f" valid={worker['valid_samples']}/{worker['metric_total_samples']}"
                f" comp={worker['comp_valid_samples']}/{worker['metric_total_samples']}"
                f" struct={worker['struct_valid_samples']}/{worker['metric_total_samples']}"
            )
        print(worker_line)
    return summary


def build_worker_command(
    child_args,
    worker_index,
    assigned_rounds,
    round_offset,
    base_seed,
    worker_save_dir,
):
    worker_args = list(child_args)
    base_exp_name = get_arg_value(worker_args, "--exp_name", worker_save_dir.parent.name)
    worker_exp_name = append_worker_suffix(base_exp_name, worker_index)
    worker_seed = base_seed + round_offset
    worker_debug_dir = worker_save_dir / "atom_type_debug"

    worker_args = set_arg(worker_args, "--exp_name", worker_exp_name)
    worker_args = set_arg(worker_args, "--num_rounds", assigned_rounds)
    worker_args = set_arg(worker_args, "--sample_seed", worker_seed)
    worker_args = set_arg(worker_args, "--save_dir", str(worker_save_dir))
    worker_args = set_arg(worker_args, "--debug-atom-dir", str(worker_debug_dir))
    return worker_args


def main():
    parser = argparse.ArgumentParser(
        description="Run LF_wrap sampling on multiple GPUs by sharding num_rounds across worker processes."
    )
    parser.add_argument("--gpus", type=parse_gpu_list, required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--num-rounds", type=int, required=True, help="Total num_rounds to distribute across workers.")
    parser.add_argument("--sample-seed", type=int, required=True, help="Base sample seed used for round-offset sharding.")
    parser.add_argument("--save-dir", type=Path, required=True, help="Base save dir. Each worker writes to save_dir/worker_XX.")
    parser.add_argument(
        "--main-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "main_LF_sample.py",
        help="Path to main_LF_sample.py",
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable for worker processes. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--launch-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between worker launches to reduce simultaneous I/O spikes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print worker commands without launching them.",
    )
    parser.add_argument(
        "child_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to main_LF_sample.py. Put them after '--'.",
    )
    args = parser.parse_args()

    child_args = list(args.child_args)
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    if not child_args:
        raise SystemExit("Please pass main_LF_sample.py arguments after '--'.")

    total_rounds = int(args.num_rounds)
    if total_rounds <= 0:
        raise SystemExit("--num-rounds must be positive.")

    base_save_dir = args.save_dir.resolve()
    base_save_dir.mkdir(parents=True, exist_ok=True)
    rounds_per_worker = distribute_rounds(total_rounds, len(args.gpus))

    worker_specs = []
    round_offset = 0
    for worker_index, (gpu_id, assigned_rounds) in enumerate(zip(args.gpus, rounds_per_worker)):
        if assigned_rounds <= 0:
            continue
        worker_save_dir = base_save_dir / f"worker_{worker_index:02d}"
        worker_save_dir.mkdir(parents=True, exist_ok=True)
        worker_args = build_worker_command(
            child_args=child_args,
            worker_index=worker_index,
            assigned_rounds=assigned_rounds,
            round_offset=round_offset,
            base_seed=args.sample_seed,
            worker_save_dir=worker_save_dir,
        )
        worker_specs.append(
            {
                "worker_index": worker_index,
                "gpu_id": gpu_id,
                "assigned_rounds": assigned_rounds,
                "round_offset": round_offset,
                "sample_seed": args.sample_seed + round_offset,
                "worker_save_dir": str(worker_save_dir),
                "command": [args.python_exec, str(args.main_script)] + worker_args,
            }
        )
        round_offset += assigned_rounds

    manifest = {
        "main_script": str(args.main_script.resolve()),
        "python_exec": args.python_exec,
        "gpus": args.gpus,
        "num_rounds": total_rounds,
        "sample_seed": int(args.sample_seed),
        "workers": worker_specs,
    }
    manifest_path = base_save_dir / "multi_gpu_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"Saved launch manifest to {manifest_path}")
    for spec in worker_specs:
        printable_cmd = " ".join(shlex.quote(token) for token in spec["command"])
        print(
            f"[worker {spec['worker_index']:02d} | gpu {spec['gpu_id']}] "
            f"rounds={spec['assigned_rounds']} seed={spec['sample_seed']} save_dir={spec['worker_save_dir']}"
        )
        print(f"  {printable_cmd}")

    if args.dry_run:
        return

    processes = []
    stream_threads = []
    try:
        for spec in worker_specs:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_id"])
            process = subprocess.Popen(
                spec["command"],
                cwd=os.getcwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            prefix = f"[worker {spec['worker_index']:02d} | gpu {spec['gpu_id']}] "
            thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, prefix),
                daemon=True,
            )
            thread.start()
            processes.append(process)
            stream_threads.append(thread)
            if args.launch_delay_seconds > 0:
                time.sleep(args.launch_delay_seconds)

        exit_codes = [process.wait() for process in processes]
        for thread in stream_threads:
            thread.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating worker processes...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    summary = summarize_workers(
        base_save_dir=base_save_dir,
        worker_dirs=[Path(spec["worker_save_dir"]) for spec in worker_specs],
        exit_codes=exit_codes,
    )
    summary_path = base_save_dir / "multi_gpu_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    print(f"Saved aggregate summary to {summary_path}")

    component_metrics = aggregate_component_metrics(
        base_save_dir=base_save_dir,
        worker_dirs=[Path(spec["worker_save_dir"]) for spec in worker_specs],
        manifest=manifest,
    )
    if component_metrics is not None:
        print(f"Saved aggregate component diagnostics to {base_save_dir / 'metrics.json'}")

    if any(code != 0 for code in exit_codes):
        raise SystemExit(max(code for code in exit_codes if code != 0))


if __name__ == "__main__":
    main()
