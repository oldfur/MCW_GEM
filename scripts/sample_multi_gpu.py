#!/usr/bin/env python3
import argparse
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

    if any(code != 0 for code in exit_codes):
        raise SystemExit(max(code for code in exit_codes if code != 0))


if __name__ == "__main__":
    main()
