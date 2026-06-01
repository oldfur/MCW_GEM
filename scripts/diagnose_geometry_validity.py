#!/usr/bin/env python3
"""Diagnose PBC geometry validity for generated crystal CIFs.

The LF_wrap model currently builds a column-wise lattice matrix in
compute_lattice_matrix(). This script compares the current internal metric
(`frac @ L`) against the corrected column-wise metric (`frac @ L.T`) and
pymatgen's CIF/evaluation metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure


SAMPLE_RE = re.compile(r"sample_(\d+)")


def compute_lattice_matrix_columns(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Match en_diffusion_LF_wrap.compute_lattice_matrix: columns are v1,v2,v3."""
    a, b, c = [float(v) for v in lengths]
    alpha, beta, gamma = np.deg2rad(angles.astype(float))
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        return np.full((3, 3), np.nan, dtype=float)

    v1 = np.array([a, 0.0, 0.0], dtype=float)
    v2 = np.array([b * cos_g, b * sin_g, 0.0], dtype=float)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz2 = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz2, 0.0))
    v3 = np.array([cx, cy, cz], dtype=float)
    return np.stack([v1, v2, v3], axis=1)


def pairwise_min_dist_from_frac(
    frac: np.ndarray,
    lattice: np.ndarray,
    *,
    transpose_lattice: bool,
    cutoff: float,
) -> tuple[float | None, int]:
    """Compute all-pair minimum-image distances by enumerating 27 neighbor cells."""
    frac = np.asarray(frac, dtype=float)
    if frac.shape[0] < 2:
        return None, 0

    transform = lattice.T if transpose_lattice else lattice
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    distances = []
    close_pairs = 0
    for i in range(frac.shape[0]):
        for j in range(i + 1, frac.shape[0]):
            dfrac = frac[i] - frac[j]
            images = (dfrac[None, :] + shifts) @ transform
            dist = float(np.linalg.norm(images, axis=1).min())
            distances.append(dist)
            if dist < cutoff:
                close_pairs += 1
    return min(distances), close_pairs


def pymatgen_min_dist_and_count(structure: Structure, cutoff: float) -> tuple[float | None, int]:
    if len(structure) < 2:
        return None, 0
    dist_mat = np.array(structure.distance_matrix, dtype=float)
    iu = np.triu_indices(len(structure), k=1)
    pair_dist = dist_mat[iu]
    return float(pair_dist.min()), int((pair_dist < cutoff).sum())


def shortest_cell_translation(matrix: np.ndarray) -> float | None:
    """Shortest non-zero translation among small integer lattice combinations."""
    vectors = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                vectors.append(np.array([i, j, k], dtype=float) @ matrix)
    norms = np.linalg.norm(np.stack(vectors, axis=0), axis=1)
    if norms.size == 0:
        return None
    return float(norms.min())


def parse_sample_index(path: Path) -> int | None:
    match = SAMPLE_RE.search(path.name)
    return int(match.group(1)) if match else None


def find_worker_dir(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    for part in rel.parts:
        if re.fullmatch(r"worker_\d+", part):
            return part
    return "."


def category_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "struct_invalid" in parts:
        return "struct_invalid"
    if "comp_invalid" in parts:
        return "comp_invalid"
    return "valid_dir"


def load_sample_metadata(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load internal per-sample metadata when debug JSONL files are available."""
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for jsonl_path in root.rglob("analyze_test_samples.jsonl"):
        worker = find_worker_dir(jsonl_path, root)
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") != "sample_pre_cif_save":
                    continue
                sample_idx = row.get("sample_global_index")
                if sample_idx is None:
                    continue
                metadata[(worker, int(sample_idx))] = row
    return metadata


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    clean = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "p05": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def summarize_rows(rows: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "geometry_valid": [r for r in rows if r["geometry_valid"]],
        "geometry_invalid": [r for r in rows if not r["geometry_valid"]],
    }
    feature_names = [
        "pymatgen_min_dist",
        "current_min_dist_frac_at_L",
        "corrected_min_dist_frac_at_LT",
        "volume_per_atom",
        "shortest_lattice_vector",
        "shortest_cell_translation",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "cell_condition_number",
        "pymatgen_close_pairs",
    ]

    by_group = {}
    for group_name, group_rows in groups.items():
        by_group[group_name] = {
            "count": len(group_rows),
            "features": {
                feature: summarize_values([finite_float(row.get(feature)) for row in group_rows])
                for feature in feature_names
            },
        }

    current_diffs = [
        abs(r["current_min_dist_frac_at_L"] - r["pymatgen_min_dist"])
        for r in rows
        if r["current_min_dist_frac_at_L"] is not None and r["pymatgen_min_dist"] is not None
    ]
    corrected_diffs = [
        abs(r["corrected_min_dist_frac_at_LT"] - r["pymatgen_min_dist"])
        for r in rows
        if r["corrected_min_dist_frac_at_LT"] is not None and r["pymatgen_min_dist"] is not None
    ]
    current_valid_agree = sum(
        bool(r["current_min_dist_frac_at_L"] is not None and r["current_min_dist_frac_at_L"] >= cutoff)
        == bool(r["geometry_valid"])
        for r in rows
    )
    corrected_valid_agree = sum(
        bool(r["corrected_min_dist_frac_at_LT"] is not None and r["corrected_min_dist_frac_at_LT"] >= cutoff)
        == bool(r["geometry_valid"])
        for r in rows
    )

    invalid_rows = [r for r in rows if not r["geometry_valid"]]
    invalid_small_vn = [
        r for r in invalid_rows
        if r["volume_per_atom"] is not None and r["volume_per_atom"] < 5.0
    ]
    invalid_bad_shape = [
        r for r in invalid_rows
        if r["cell_condition_number"] is not None and r["cell_condition_number"] > 10.0
    ]

    return {
        "total_samples": len(rows),
        "cutoff_angstrom": float(cutoff),
        "geometry_valid_count": len(groups["geometry_valid"]),
        "geometry_invalid_count": len(groups["geometry_invalid"]),
        "geometry_valid_rate": float(len(groups["geometry_valid"]) / len(rows)) if rows else None,
        "metric_comparison": {
            "current_abs_diff_to_pymatgen": summarize_values(current_diffs),
            "corrected_abs_diff_to_pymatgen": summarize_values(corrected_diffs),
            "current_validity_agreement_count": int(current_valid_agree),
            "corrected_validity_agreement_count": int(corrected_valid_agree),
            "current_validity_agreement_rate": float(current_valid_agree / len(rows)) if rows else None,
            "corrected_validity_agreement_rate": float(corrected_valid_agree / len(rows)) if rows else None,
        },
        "valid_vs_invalid": by_group,
        "invalid_lattice_flags": {
            "invalid_with_volume_per_atom_lt_5": len(invalid_small_vn),
            "invalid_with_condition_number_gt_10": len(invalid_bad_shape),
        },
    }


def collect_rows(root: Path, cutoff: float, max_samples: int | None) -> list[dict[str, Any]]:
    metadata = load_sample_metadata(root)
    cif_paths = sorted(root.rglob("*.cif"))
    if max_samples is not None:
        cif_paths = cif_paths[:max_samples]

    rows: list[dict[str, Any]] = []
    for cif_path in cif_paths:
        worker = find_worker_dir(cif_path, root)
        sample_idx = parse_sample_index(cif_path)
        row: dict[str, Any] = {
            "path": str(cif_path),
            "worker": worker,
            "sample_idx": sample_idx,
            "category": category_from_path(cif_path),
            "read_error": None,
        }
        try:
            structure = Structure.from_file(str(cif_path))
        except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
            row["read_error"] = repr(exc)
            row["geometry_valid"] = False
            rows.append(row)
            continue

        frac = np.asarray(structure.frac_coords, dtype=float) % 1.0
        lengths = np.asarray(structure.lattice.abc, dtype=float)
        angles = np.asarray(structure.lattice.angles, dtype=float)
        meta = metadata.get((worker, sample_idx)) if sample_idx is not None else None
        if meta is not None:
            meta_lengths = np.asarray(meta.get("lengths", lengths), dtype=float)
            meta_angles = np.asarray(meta.get("angles", angles), dtype=float)
            row["metadata_found"] = True
            row["max_abs_length_delta_metadata_vs_cif"] = float(np.max(np.abs(meta_lengths - lengths)))
            row["max_abs_angle_delta_metadata_vs_cif"] = float(np.max(np.abs(meta_angles - angles)))
            # The historical debug JSONL stores generated lengths/angles and atom types,
            # but not fractional coordinates. Use CIF fractional coords for distance checks.
            lengths_for_internal_metric = meta_lengths
            angles_for_internal_metric = meta_angles
        else:
            row["metadata_found"] = False
            row["max_abs_length_delta_metadata_vs_cif"] = None
            row["max_abs_angle_delta_metadata_vs_cif"] = None
            lengths_for_internal_metric = lengths
            angles_for_internal_metric = angles

        column_lattice = compute_lattice_matrix_columns(lengths_for_internal_metric, angles_for_internal_metric)
        row_lattice = np.asarray(structure.lattice.matrix, dtype=float)
        pymatgen_min, pymatgen_close = pymatgen_min_dist_and_count(structure, cutoff)
        current_min, current_close = pairwise_min_dist_from_frac(
            frac, column_lattice, transpose_lattice=False, cutoff=cutoff
        )
        corrected_min, corrected_close = pairwise_min_dist_from_frac(
            frac, column_lattice, transpose_lattice=True, cutoff=cutoff
        )

        row.update(
            {
                "num_atoms": int(len(structure)),
                "pymatgen_min_dist": pymatgen_min,
                "current_min_dist_frac_at_L": current_min,
                "corrected_min_dist_frac_at_LT": corrected_min,
                "pymatgen_close_pairs": int(pymatgen_close),
                "current_close_pairs": int(current_close),
                "corrected_close_pairs": int(corrected_close),
                "geometry_valid": bool(
                    pymatgen_min is None or (pymatgen_min >= cutoff and structure.volume >= 0.1)
                ),
                "volume": float(structure.volume),
                "volume_per_atom": float(structure.volume / max(len(structure), 1)),
                "shortest_lattice_vector": float(min(lengths)),
                "shortest_cell_translation": shortest_cell_translation(row_lattice),
                "a": float(lengths[0]),
                "b": float(lengths[1]),
                "c": float(lengths[2]),
                "alpha": float(angles[0]),
                "beta": float(angles[1]),
                "gamma": float(angles[2]),
                "cell_condition_number": float(np.linalg.cond(row_lattice)),
                "current_abs_diff_to_pymatgen": None
                if current_min is None or pymatgen_min is None
                else float(abs(current_min - pymatgen_min)),
                "corrected_abs_diff_to_pymatgen": None
                if corrected_min is None or pymatgen_min is None
                else float(abs(corrected_min - pymatgen_min)),
            }
        )
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "path",
        "worker",
        "sample_idx",
        "category",
        "read_error",
        "metadata_found",
        "num_atoms",
        "geometry_valid",
        "pymatgen_min_dist",
        "current_min_dist_frac_at_L",
        "corrected_min_dist_frac_at_LT",
        "current_abs_diff_to_pymatgen",
        "corrected_abs_diff_to_pymatgen",
        "pymatgen_close_pairs",
        "current_close_pairs",
        "corrected_close_pairs",
        "volume",
        "volume_per_atom",
        "shortest_lattice_vector",
        "shortest_cell_translation",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "cell_condition_number",
        "max_abs_length_delta_metadata_vs_cif",
        "max_abs_angle_delta_metadata_vs_cif",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path, help="Sample output directory containing CIF files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Minimum valid PBC distance in Angstrom.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick diagnostics.")
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else sample_dir)
    rows = collect_rows(sample_dir, cutoff=args.cutoff, max_samples=args.max_samples)
    summary = summarize_rows(rows, cutoff=args.cutoff)

    csv_path = output_dir / "geometry_diagnostics.csv"
    summary_path = output_dir / "geometry_diagnostics_summary.json"
    write_csv(rows, csv_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    print(json.dumps(summary["metric_comparison"], ensure_ascii=True, indent=2))
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
