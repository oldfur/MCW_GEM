#!/usr/bin/env python3
"""Measure how much final-window geometry correction moves MP-20 samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIGS = ("full_pipeline", "corrected_geometry_raw_logits")
SAMPLE_ID_RE = re.compile(r"(epoch_\d+_sample_\d+)")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def finite_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_stat(values: list[float] | np.ndarray, fn: str) -> float | None:
    arr = finite_array(values if isinstance(values, list) else values.tolist())
    if arr.size == 0:
        return None
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "p95":
        return float(np.percentile(arr, 95))
    if fn == "max":
        return float(np.max(arr))
    if fn == "min":
        return float(np.min(arr))
    if fn == "rms":
        return float(np.sqrt(np.mean(arr ** 2)))
    raise ValueError(f"unknown stat {fn}")


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    # Sampler cells are column-wise: Cartesian row vector = frac @ cell.T.
    return np.asarray(frac, dtype=float) @ np.asarray(lattice, dtype=float).T


def lattice_metric_from_sampler_cell(lattice: np.ndarray) -> np.ndarray:
    cell = np.asarray(lattice, dtype=float)
    return cell.T @ cell


def lattice_metric_from_pymatgen_matrix(lattice: np.ndarray) -> np.ndarray:
    matrix = np.asarray(lattice, dtype=float)
    return matrix @ matrix.T


def lengths_angles_from_metric(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    metric = np.asarray(metric, dtype=float)
    lengths = np.sqrt(np.clip(np.diag(metric), 0.0, None))
    angles = []
    for i, j, k in ((1, 2, 0), (0, 2, 1), (0, 1, 2)):
        denom = max(float(lengths[i] * lengths[j]), 1e-12)
        cos_value = float(np.clip(metric[i, j] / denom, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_value))))
    return lengths, np.asarray(angles, dtype=float)


def minimum_image_delta(frac_after: np.ndarray, frac_before: np.ndarray) -> np.ndarray:
    delta = np.asarray(frac_after, dtype=float) - np.asarray(frac_before, dtype=float)
    return delta - np.round(delta)


def pair_min_distance(frac_coords: np.ndarray, lattice: np.ndarray) -> float | None:
    frac_coords = np.asarray(frac_coords, dtype=float)
    n = int(frac_coords.shape[0])
    if n < 2:
        return None
    delta = frac_coords[:, None, :] - frac_coords[None, :, :]
    delta = delta - np.round(delta)
    cart_delta = frac_to_cart(delta, lattice)
    dist = np.linalg.norm(cart_delta, axis=-1)
    dist[np.eye(n, dtype=bool)] = np.inf
    value = float(np.min(dist))
    return value if math.isfinite(value) else None


def index_final_cifs(epoch_dir: Path) -> dict[str, Path]:
    cifs: dict[str, Path] = {}
    if not epoch_dir.exists():
        return cifs
    for path in epoch_dir.rglob("*.cif"):
        match = SAMPLE_ID_RE.search(path.name)
        if not match:
            continue
        sample_id = match.group(1)
        # Prefer the directly saved CIF if duplicate names appear in invalid subdirs.
        if sample_id not in cifs or path.parent == epoch_dir:
            cifs[sample_id] = path
    return cifs


def load_structure(path: Path):
    from pymatgen.core import Structure

    return Structure.from_file(str(path))


def analyze_worker(
    worker_dir: Path,
    d_min: float,
    move_tol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    npz_path = worker_dir / "epoch_0" / "geometry_pre_correction" / "geometry_pre_correction_raw.npz"
    epoch_dir = worker_dir / "epoch_0"
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if not npz_path.exists():
        failures.append({"worker_dir": str(worker_dir), "reason": "missing_pre_correction_npz", "path": str(npz_path)})
        return rows, failures

    final_cifs = index_final_cifs(epoch_dir)
    data = np.load(npz_path, allow_pickle=True)
    lattices = np.asarray(data["lattices"], dtype=float)
    frac_coords = np.asarray(data["frac_coords"], dtype=float)
    num_atoms = np.asarray(data["num_atoms"], dtype=int)
    sample_ids = [str(v) for v in data["sample_ids"].tolist()]

    for idx, sample_id in enumerate(sample_ids):
        cif_path = final_cifs.get(sample_id)
        if cif_path is None:
            failures.append(
                {
                    "worker_dir": str(worker_dir),
                    "sample_id": sample_id,
                    "reason": "missing_final_cif",
                }
            )
            continue

        n = int(num_atoms[idx])
        pre_lattice = lattices[idx]
        pre_frac = np.mod(frac_coords[idx, :n, :], 1.0)
        try:
            structure = load_structure(cif_path)
        except Exception as exc:
            failures.append(
                {
                    "worker_dir": str(worker_dir),
                    "sample_id": sample_id,
                    "reason": "cif_parse_error",
                    "path": str(cif_path),
                    "error": str(exc),
                }
            )
            continue

        post_frac = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
        post_lattice = np.asarray(structure.lattice.matrix, dtype=float)
        if int(post_frac.shape[0]) != n:
            failures.append(
                {
                    "worker_dir": str(worker_dir),
                    "sample_id": sample_id,
                    "reason": "atom_count_mismatch",
                    "pre_num_atoms": n,
                    "post_num_atoms": int(post_frac.shape[0]),
                    "path": str(cif_path),
                }
            )
            continue

        delta_frac = minimum_image_delta(post_frac, pre_frac)
        delta_cart = frac_to_cart(delta_frac, pre_lattice)
        displacement = np.linalg.norm(delta_cart, axis=-1)
        moved = displacement > float(move_tol)
        rmsd = float(np.sqrt(np.mean(displacement ** 2))) if displacement.size else 0.0
        pre_min_distance = pair_min_distance(pre_frac, pre_lattice)
        post_min_distance = pair_min_distance(post_frac, pre_lattice)
        pre_metric = lattice_metric_from_sampler_cell(pre_lattice)
        post_metric = lattice_metric_from_pymatgen_matrix(post_lattice)
        pre_lengths, pre_angles = lengths_angles_from_metric(pre_metric)
        post_lengths, post_angles = lengths_angles_from_metric(post_metric)
        lattice_metric_delta = post_metric - pre_metric
        lattice_length_delta = post_lengths - pre_lengths
        lattice_angle_delta = post_angles - pre_angles

        rows.append(
            {
                "worker_dir": str(worker_dir),
                "sample_id": sample_id,
                "final_cif": str(cif_path),
                "num_atoms": n,
                "num_atoms_moved": int(np.sum(moved)),
                "any_correction": bool(np.any(moved)),
                "mean_displacement": float(np.mean(displacement)) if displacement.size else 0.0,
                "median_displacement": float(np.median(displacement)) if displacement.size else 0.0,
                "max_displacement": float(np.max(displacement)) if displacement.size else 0.0,
                "rmsd": rmsd,
                "pre_min_distance": pre_min_distance,
                "post_min_distance": post_min_distance,
                "pre_contact_fail": bool(pre_min_distance is not None and pre_min_distance < d_min),
                "post_contact_fail": bool(post_min_distance is not None and post_min_distance < d_min),
                "lattice_metric_frobenius_change": float(np.linalg.norm(lattice_metric_delta)),
                "lattice_metric_max_abs_change": float(np.max(np.abs(lattice_metric_delta))),
                "lattice_length_max_abs_change": float(np.max(np.abs(lattice_length_delta))),
                "lattice_angle_max_abs_change": float(np.max(np.abs(lattice_angle_delta))),
                "displacements": displacement,
                "moved_displacements": displacement[moved],
            }
        )
    return rows, failures


def discover_workers(root: Path, configs: list[str]) -> list[Path]:
    workers: list[Path] = []
    for config in configs:
        config_dir = root / config
        if not config_dir.exists():
            continue
        workers.extend(sorted(p for p in config_dir.glob("worker_*") if p.is_dir()))
        if (config_dir / "epoch_0" / "geometry_pre_correction" / "geometry_pre_correction_raw.npz").exists():
            workers.append(config_dir)
    return workers


def analyze_correction_log_worker(worker_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    log_path = worker_dir / "atom_type_debug" / "geometry_correction_displacement.jsonl"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not log_path.exists():
        failures.append({"worker_dir": str(worker_dir), "reason": "missing_correction_displacement_log", "path": str(log_path)})
        return rows, failures
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "worker_dir": str(worker_dir),
                        "path": str(log_path),
                        "line_number": line_number,
                        "reason": "json_decode_error",
                        "error": str(exc),
                    }
                )
                continue
            for sample in payload.get("samples", []):
                displacement = np.asarray(sample.get("displacements", []), dtype=float)
                moved_displacement = np.asarray(sample.get("moved_displacements", []), dtype=float)
                pre_min_distance = sample.get("pre_min_distance")
                post_min_distance = sample.get("post_min_distance")
                rows.append(
                    {
                        "worker_dir": str(worker_dir),
                        "sample_id": f"round_{payload.get('round_index')}_sample_{sample.get('sample_local_index')}",
                        "num_atoms": int(sample.get("num_atoms", displacement.size)),
                        "num_atoms_moved": int(sample.get("num_atoms_moved", moved_displacement.size)),
                        "any_correction": bool(sample.get("any_correction", moved_displacement.size > 0)),
                        "mean_displacement": sample.get("mean_displacement"),
                        "median_displacement": sample.get("median_displacement"),
                        "max_displacement": sample.get("max_displacement"),
                        "rmsd": sample.get("rmsd"),
                        "pre_min_distance": pre_min_distance,
                        "post_min_distance": post_min_distance,
                        "pre_contact_fail": bool(sample.get("pre_contact_fail", False)),
                        "post_contact_fail": bool(sample.get("post_contact_fail", False)),
                        "lattice_metric_frobenius_change": 0.0,
                        "lattice_metric_max_abs_change": 0.0,
                        "lattice_length_max_abs_change": 0.0,
                        "lattice_angle_max_abs_change": 0.0,
                        "displacements": displacement,
                        "moved_displacements": moved_displacement,
                    }
                )
    return rows, failures


def build_metrics(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    root: Path,
    configs: list[str],
    d_min: float,
    move_tol: float,
) -> dict[str, Any]:
    num_samples = len(rows)
    num_atoms_total = int(sum(int(row["num_atoms"]) for row in rows))
    num_samples_with_any_correction = int(sum(bool(row["any_correction"]) for row in rows))
    num_atoms_moved = int(sum(int(row["num_atoms_moved"]) for row in rows))

    all_disp = np.concatenate([row["displacements"] for row in rows if len(row["displacements"])]) if rows else np.asarray([])
    moved_disp = (
        np.concatenate([row["moved_displacements"] for row in rows if len(row["moved_displacements"])])
        if rows else np.asarray([])
    )
    corrected_rmsd = [float(row["rmsd"]) for row in rows if bool(row["any_correction"])]
    pre_min = [row["pre_min_distance"] for row in rows if row["pre_min_distance"] is not None]
    post_min = [row["post_min_distance"] for row in rows if row["post_min_distance"] is not None]
    lattice_metric_frob = [float(row["lattice_metric_frobenius_change"]) for row in rows]
    lattice_metric_max_abs = [float(row["lattice_metric_max_abs_change"]) for row in rows]
    lattice_length_max_abs = [float(row["lattice_length_max_abs_change"]) for row in rows]
    lattice_angle_max_abs = [float(row["lattice_angle_max_abs_change"]) for row in rows]

    metrics = {
        "root": str(root),
        "configs_analyzed": configs,
        "d_min_angstrom": float(d_min),
        "move_tolerance_angstrom": float(move_tol),
        "num_samples": num_samples,
        "num_failures": len(failures),
        "num_samples_with_any_correction": num_samples_with_any_correction,
        "fraction_samples_corrected": rate(num_samples_with_any_correction, num_samples),
        "num_atoms_total": num_atoms_total,
        "num_atoms_moved": num_atoms_moved,
        "fraction_atoms_moved": rate(num_atoms_moved, num_atoms_total),
        "mean_cartesian_displacement_per_atom": safe_stat(all_disp, "mean"),
        "median_cartesian_displacement_per_atom": safe_stat(all_disp, "median"),
        "mean_cartesian_displacement_moved_only": safe_stat(moved_disp, "mean"),
        "median_cartesian_displacement_moved_only": safe_stat(moved_disp, "median"),
        "p95_cartesian_displacement_moved_only": safe_stat(moved_disp, "p95"),
        "max_cartesian_displacement_moved_only": safe_stat(moved_disp, "max"),
        "rmsd_all_atoms": safe_stat(all_disp, "rms"),
        "mean_rmsd_per_sample": safe_stat([float(row["rmsd"]) for row in rows], "mean"),
        "median_rmsd_per_sample": safe_stat([float(row["rmsd"]) for row in rows], "median"),
        "mean_rmsd_per_corrected_sample": safe_stat(corrected_rmsd, "mean"),
        "median_rmsd_per_corrected_sample": safe_stat(corrected_rmsd, "median"),
        "mean_nearest_neighbor_distance_before_correction": safe_stat(pre_min, "mean"),
        "median_nearest_neighbor_distance_before_correction": safe_stat(pre_min, "median"),
        "min_nearest_neighbor_distance_before_correction": safe_stat(pre_min, "min"),
        "mean_nearest_neighbor_distance_after_correction": safe_stat(post_min, "mean"),
        "median_nearest_neighbor_distance_after_correction": safe_stat(post_min, "median"),
        "min_nearest_neighbor_distance_after_correction": safe_stat(post_min, "min"),
        "num_samples_below_d_min_before_correction": int(sum(bool(row["pre_contact_fail"]) for row in rows)),
        "num_samples_below_d_min_after_correction": int(sum(bool(row["post_contact_fail"]) for row in rows)),
        "fraction_samples_below_d_min_before_correction": rate(
            int(sum(bool(row["pre_contact_fail"]) for row in rows)), num_samples
        ),
        "fraction_samples_below_d_min_after_correction": rate(
            int(sum(bool(row["post_contact_fail"]) for row in rows)), num_samples
        ),
        "mean_lattice_metric_frobenius_change": safe_stat(lattice_metric_frob, "mean"),
        "max_lattice_metric_frobenius_change": safe_stat(lattice_metric_frob, "max"),
        "mean_lattice_metric_max_abs_change": safe_stat(lattice_metric_max_abs, "mean"),
        "max_lattice_metric_max_abs_change": safe_stat(lattice_metric_max_abs, "max"),
        "mean_lattice_length_max_abs_change": safe_stat(lattice_length_max_abs, "mean"),
        "max_lattice_length_max_abs_change": safe_stat(lattice_length_max_abs, "max"),
        "mean_lattice_angle_max_abs_change": safe_stat(lattice_angle_max_abs, "mean"),
        "max_lattice_angle_max_abs_change": safe_stat(lattice_angle_max_abs, "max"),
        "lattice_unchanged_within_1e_5": (
            bool(max(lattice_length_max_abs) <= 1e-5 and max(lattice_angle_max_abs) <= 1e-5)
            if lattice_length_max_abs and lattice_angle_max_abs else None
        ),
        "pairing_note": (
            "Correction-only metrics are computed from atom_type_debug/geometry_correction_displacement.jsonl "
            "when available. The legacy pre-NPZ/final-CIF proxy is opt-in because it includes normal "
            "reverse-SDE motion after the first correction point."
        ),
        "failures_preview": failures[:20],
    }
    return metrics


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if key == "failures_preview":
                continue
            writer.writerow([key, json.dumps(jsonable(value), ensure_ascii=False) if isinstance(value, (list, dict)) else value])


def format_value(value: Any, percent: bool = False, digits: int = 4) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{100.0 * float(value):.2f}\\%"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_latex_table(path: Path, metrics: dict[str, Any]) -> None:
    rows = [
        ("Samples corrected (\\%)", format_value(metrics["fraction_samples_corrected"], percent=True)),
        ("Atoms moved (\\%)", format_value(metrics["fraction_atoms_moved"], percent=True)),
        ("Mean displacement among moved atoms (\\AA)", format_value(metrics["mean_cartesian_displacement_moved_only"])),
        ("Median displacement among moved atoms (\\AA)", format_value(metrics["median_cartesian_displacement_moved_only"])),
        ("95th percentile displacement (\\AA)", format_value(metrics["p95_cartesian_displacement_moved_only"])),
        ("Max displacement (\\AA)", format_value(metrics["max_cartesian_displacement_moved_only"])),
        ("Mean RMSD per corrected sample (\\AA)", format_value(metrics["mean_rmsd_per_corrected_sample"])),
        ("Pre-correction contact-fail rate (\\%)", format_value(metrics["fraction_samples_below_d_min_before_correction"], percent=True)),
        ("Post-correction contact-fail rate (\\%)", format_value(metrics["fraction_samples_below_d_min_after_correction"], percent=True)),
    ]
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    for metric, value in rows:
        lines.append(f"{tex_escape(metric)} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_plots(root: Path, rows: list[dict[str, Any]], d_min: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable; skipping histograms ({exc})", file=sys.stderr)
        return

    all_disp = np.concatenate([row["displacements"] for row in rows if len(row["displacements"])]) if rows else np.asarray([])
    moved_disp = (
        np.concatenate([row["moved_displacements"] for row in rows if len(row["moved_displacements"])])
        if rows else np.asarray([])
    )
    if all_disp.size:
        plt.figure(figsize=(6, 4))
        plt.hist(all_disp, bins=80, alpha=0.65, label="all atoms")
        if moved_disp.size:
            plt.hist(moved_disp, bins=80, alpha=0.65, label="moved atoms")
        plt.xlabel("Cartesian displacement (Angstrom)")
        plt.ylabel("Atom count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / "displacement_histogram.png", dpi=200)
        plt.close()

    pre_min = finite_array([row["pre_min_distance"] for row in rows if row["pre_min_distance"] is not None])
    post_min = finite_array([row["post_min_distance"] for row in rows if row["post_min_distance"] is not None])
    if pre_min.size or post_min.size:
        plt.figure(figsize=(6, 4))
        if pre_min.size:
            plt.hist(pre_min, bins=80, alpha=0.6, label="pre")
        if post_min.size:
            plt.hist(post_min, bins=80, alpha=0.6, label="post")
        plt.axvline(d_min, color="black", linestyle="--", linewidth=1.0, label=f"d_min={d_min:g}")
        plt.xlabel("Nearest-neighbor distance (Angstrom)")
        plt.ylabel("Sample count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(root / "min_distance_pre_post_histogram.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ablation_component_diagnostics"))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--d-min", type=float, default=0.5)
    parser.add_argument("--move-tol", type=float, default=1e-4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--allow-pre-final-proxy",
        action="store_true",
        help="Use pre-correction NPZ plus final CIFs if correction-only JSONL logs are missing. "
             "This is a proxy and includes reverse-SDE motion, so it is not recommended for paper metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    workers = discover_workers(root, list(args.configs))
    if not workers:
        raise SystemExit(f"No geometry-correction workers found under {root} for configs={args.configs}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for worker_dir in workers:
        worker_rows, worker_failures = analyze_correction_log_worker(worker_dir)
        rows.extend(worker_rows)
        failures.extend(worker_failures)

    if not rows:
        if not args.allow_pre_final_proxy:
            raise SystemExit(
                "No correction-only displacement logs were found. Rerun geometry-correction-enabled "
                "sampling with --debug-atom-types True after this patch, then rerun this script. "
                "Use --allow-pre-final-proxy only for debugging; it is not correction-only."
            )
        rows = []
        failures = []
        for worker_dir in workers:
            worker_rows, worker_failures = analyze_worker(worker_dir, d_min=args.d_min, move_tol=args.move_tol)
            rows.extend(worker_rows)
            failures.extend(worker_failures)

    metrics = build_metrics(rows, failures, root=root, configs=list(args.configs), d_min=args.d_min, move_tol=args.move_tol)
    json_path = root / "geometry_correction_displacement_metrics.json"
    csv_path = root / "geometry_correction_displacement_metrics.csv"
    tex_path = root / "geometry_correction_displacement_table.tex"
    json_path.write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_metrics_csv(csv_path, metrics)
    write_latex_table(tex_path, metrics)
    if not args.no_plots:
        maybe_write_plots(root, rows, d_min=args.d_min)

    print(f"Analyzed {metrics['num_samples']} matched samples across {len(workers)} worker dirs")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    if failures:
        print(f"warning: {len(failures)} samples/files could not be paired or parsed; see failures_preview in JSON", file=sys.stderr)


if __name__ == "__main__":
    main()
