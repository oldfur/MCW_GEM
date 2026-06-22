"""Reusable diagnostics for raw crystal geometry before correction/relaxation.

Lattices accepted by this module are column-wise: ``L[:, i]`` is lattice
vector i and Cartesian row vectors are computed as ``frac @ L.T``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PAIR_THRESHOLDS = (0.5, 0.55, 0.7, 0.8, 1.0)
AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660


def lattice_from_lengths_angles(lengths: Sequence[float], angles: Sequence[float]) -> np.ndarray:
    """Construct a column-wise lattice from a,b,c and alpha,beta,gamma."""
    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.deg2rad(np.asarray(angles, dtype=float))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("invalid lattice: sin(gamma) is zero")
    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * math.cos(gamma), b * sin_gamma, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < -1e-8:
        raise ValueError(f"invalid lattice: negative c_z^2={cz_sq}")
    v3 = np.array([cx, cy, math.sqrt(max(cz_sq, 0.0))])
    return np.stack([v1, v2, v3], axis=1)


def lattice_lengths_angles(lattice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lattice = np.asarray(lattice, dtype=float)
    vectors = lattice.T
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        raise ValueError(f"non-positive lattice length: {lengths.tolist()}")

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))

    # alpha=(b,c), beta=(a,c), gamma=(a,b)
    angles = np.array(
        [angle(vectors[1], vectors[2]), angle(vectors[0], vectors[2]), angle(vectors[0], vectors[1])]
    )
    return lengths, angles


def pbc_pair_distances(frac_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Return unique-pair PBC distances using the column-wise lattice convention.

    The base minimum-image conversion is exactly ``delta -= round(delta)`` and
    ``delta_cart = delta @ lattice.T``.  The 27 neighboring images are checked
    as well, which remains correct for skewed triclinic cells.
    """
    frac = np.asarray(frac_coords, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError(f"frac_coords must be [N,3], got {frac.shape}")
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must be [3,3], got {lattice.shape}")
    if len(frac) < 2:
        return np.empty(0, dtype=float)

    delta_frac = frac[:, None, :] - frac[None, :, :]
    delta_frac = delta_frac - np.round(delta_frac)
    shifts = np.asarray(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    candidate_frac = delta_frac[..., None, :] + shifts
    # Correct convention: cart = frac @ L.T.
    candidate_cart = candidate_frac @ lattice.T
    distances = np.linalg.norm(candidate_cart, axis=-1).min(axis=-1)
    upper = np.triu_indices(len(frac), k=1)
    return distances[upper]


def _mass_density(atom_types: Optional[Sequence[int]], volume: float) -> float:
    if atom_types is None:
        return float("nan")
    atomic_numbers = np.asarray(atom_types, dtype=int).reshape(-1)
    if atomic_numbers.size == 0 or np.any(atomic_numbers < 1):
        return float("nan")
    from pymatgen.core import Element

    total_mass_amu = sum(float(Element.from_Z(int(z)).atomic_mass) for z in atomic_numbers)
    return float(total_mass_amu * AMU_PER_ANGSTROM3_TO_G_PER_CM3 / volume)


def diagnose_geometry_sample(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    num_atoms: Optional[int] = None,
    atom_types: Optional[Sequence[int]] = None,
    sample_id: Optional[str] = None,
    source: Optional[str] = None,
    train_vpa_stats: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Compute one sample's raw, pre-correction geometry metrics."""
    lattice = np.asarray(lattice, dtype=float)
    frac_coords = np.asarray(frac_coords, dtype=float)
    n = int(num_atoms if num_atoms is not None else len(frac_coords))
    if n < 1:
        raise ValueError(f"num_atoms must be positive, got {n}")
    if len(frac_coords) < n:
        raise ValueError(f"frac_coords has {len(frac_coords)} rows but num_atoms={n}")
    frac_coords = frac_coords[:n]
    if atom_types is not None:
        atom_types = np.asarray(atom_types, dtype=int).reshape(-1)[:n]
        if len(atom_types) != n:
            raise ValueError(f"atom_types has {len(atom_types)} entries but num_atoms={n}")
    if not np.isfinite(lattice).all() or not np.isfinite(frac_coords).all():
        raise ValueError("lattice or fractional coordinates contain NaN/Inf")

    volume = float(abs(np.linalg.det(lattice)))
    if not np.isfinite(volume) or volume <= 1e-10:
        raise ValueError(f"singular/invalid lattice volume: {volume}")
    lengths, angles = lattice_lengths_angles(lattice)
    pair_distances = pbc_pair_distances(frac_coords, lattice)
    d_min = float(np.min(pair_distances)) if pair_distances.size else float("nan")

    row: Dict[str, Any] = {
        "sample_id": sample_id or "",
        "source": source or "",
        "geometry_stage": "before_first_geometry_correction",
        "num_atoms": n,
        "lattice_volume": volume,
        "volume_per_atom": volume / n,
        "atom_number_density": n / volume,
        "lattice_a": float(lengths[0]),
        "lattice_b": float(lengths[1]),
        "lattice_c": float(lengths[2]),
        "lattice_alpha": float(angles[0]),
        "lattice_beta": float(angles[1]),
        "lattice_gamma": float(angles[2]),
        "lattice_condition_number": float(np.linalg.cond(lattice)),
        "d_min": d_min,
    }
    for threshold in PAIR_THRESHOLDS:
        label = str(threshold)
        row[f"num_pairs_lt_{label}"] = int(np.count_nonzero(pair_distances < threshold))
    row.update(
        {
            "has_dmin_lt_0.5": bool(np.isfinite(d_min) and d_min < 0.5),
            "has_dmin_lt_0.55": bool(np.isfinite(d_min) and d_min < 0.55),
            "has_dmin_lt_0.7": bool(np.isfinite(d_min) and d_min < 0.7),
            "has_dmin_lt_0.8": bool(np.isfinite(d_min) and d_min < 0.8),
            "num_pairs_lt_0.7_ge_1": bool(row["num_pairs_lt_0.7"] >= 1),
            "num_pairs_lt_0.7_ge_2": bool(row["num_pairs_lt_0.7"] >= 2),
        }
    )
    mass_density = _mass_density(atom_types, volume)
    row["mass_density_g_cm3"] = mass_density

    if train_vpa_stats:
        stats = train_vpa_stats.get(n)
        row["train_vpa_p05_for_n"] = stats["p05"] if stats else float("nan")
        row["train_vpa_median_for_n"] = stats["median"] if stats else float("nan")
        row["vpa_below_train_p05"] = bool(
            stats and row["volume_per_atom"] < stats["p05"]
        )
        if stats and stats["std"] > 0:
            row["vpa_zscore_for_n"] = (
                row["volume_per_atom"] - stats["mean"]
            ) / stats["std"]
        else:
            row["vpa_zscore_for_n"] = float("nan")
    return row


def load_train_vpa_stats(train_csv: str) -> Dict[int, Dict[str, float]]:
    """Compute MP-20 V/N conditional quantiles from CSV CIFs or volume columns."""
    import pandas as pd

    frame = pd.read_csv(train_csv)
    grouped: Dict[int, List[float]] = {}
    if "cif" in frame.columns:
        from pymatgen.core import Structure

        for cif_text in frame["cif"].dropna():
            try:
                structure = Structure.from_str(str(cif_text), fmt="cif")
                n = len(structure)
                grouped.setdefault(n, []).append(float(structure.volume / n))
            except Exception:
                continue
    elif {"volume", "num_atoms"}.issubset(frame.columns):
        for volume, n_value in zip(frame["volume"], frame["num_atoms"]):
            try:
                n = int(n_value)
                grouped.setdefault(n, []).append(float(volume) / n)
            except Exception:
                continue
    else:
        raise ValueError("train CSV needs a 'cif' column or volume+num_atoms columns")

    output = {}
    for n, values in grouped.items():
        array = np.asarray(values, dtype=float)
        output[n] = {
            "count": int(array.size),
            "p01": float(np.quantile(array, 0.01)),
            "p05": float(np.quantile(array, 0.05)),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "p99": float(np.quantile(array, 0.99)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }
    return output


def _finite(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, float("nan")) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _distribution_summary(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], key: str, name: str) -> None:
    values = _finite(rows, key)
    for prefix, function in (
        ("mean", np.mean), ("median", np.median),
        ("p05", lambda x: np.quantile(x, 0.05)),
        ("p95", lambda x: np.quantile(x, 0.95)),
    ):
        summary[f"{prefix}_{name}"] = float(function(values)) if values.size else None


def summarize_geometry_rows(
    rows: Sequence[Dict[str, Any]], total_samples: int, failed_samples: int
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "geometry_stage": "before_first_geometry_correction",
        "total_samples": int(total_samples),
        "parsed_samples": int(len(rows)),
        "failed_samples": int(failed_samples),
    }
    _distribution_summary(summary, rows, "atom_number_density", "atom_number_density")
    _distribution_summary(summary, rows, "volume_per_atom", "volume_per_atom")
    _distribution_summary(summary, rows, "d_min", "d_min")

    denominator = len(rows)
    for threshold in (0.5, 0.55, 0.7, 0.8):
        key = f"has_dmin_lt_{threshold}"
        summary[f"ratio_dmin_lt_{threshold}"] = (
            float(sum(bool(row.get(key, False)) for row in rows) / denominator)
            if denominator else None
        )
    for count in (1, 2):
        key = f"num_pairs_lt_0.7_ge_{count}"
        summary[f"ratio_pairs_lt_0.7_ge_{count}"] = (
            float(sum(bool(row.get(key, False)) for row in rows) / denominator)
            if denominator else None
        )
    pair_counts = _finite(rows, "num_pairs_lt_0.7")
    summary["mean_num_pairs_lt_0.7"] = float(np.mean(pair_counts)) if pair_counts.size else None
    summary["median_num_pairs_lt_0.7"] = float(np.median(pair_counts)) if pair_counts.size else None

    mass_density = _finite(rows, "mass_density_g_cm3")
    if mass_density.size:
        summary["mass_density_samples"] = int(mass_density.size)
        summary["ratio_mass_density_gt_8"] = float(np.mean(mass_density > 8.0))
        summary["ratio_mass_density_gt_10"] = float(np.mean(mass_density > 10.0))
    return summary


def save_raw_geometry_npz(records: Sequence[Dict[str, Any]], path: str) -> str:
    """Save sampler records in a padded, non-object NPZ representation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(records)
    max_atoms = max((int(record["num_atoms"]) for record in records), default=0)
    lattices = np.empty((count, 3, 3), dtype=np.float64)
    frac_coords = np.zeros((count, max_atoms, 3), dtype=np.float64)
    atom_types = np.zeros((count, max_atoms), dtype=np.int64)
    num_atoms = np.empty(count, dtype=np.int64)
    sample_ids = np.empty(count, dtype=f"U{max(1, max((len(str(r.get('sample_id', ''))) for r in records), default=1))}")
    has_atom_types = False
    for index, record in enumerate(records):
        n = int(record["num_atoms"])
        lattices[index] = np.asarray(record["lattice"], dtype=float)
        frac_coords[index, :n] = np.asarray(record["frac_coords"], dtype=float)[:n]
        num_atoms[index] = n
        sample_ids[index] = str(record.get("sample_id", index))
        if record.get("atom_types") is not None:
            atom_types[index, :n] = np.asarray(record["atom_types"], dtype=int)[:n]
            has_atom_types = True
    payload = {
        "lattices": lattices,
        "frac_coords": frac_coords,
        "num_atoms": num_atoms,
        "sample_ids": sample_ids,
        "lattice_convention": np.asarray("columns_cart_equals_frac_matmul_L_T"),
        "geometry_stage": np.asarray("before_first_geometry_correction"),
    }
    if has_atom_types:
        payload["atom_types"] = atom_types
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def _write_plots(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = (
        ("atom_number_density", "hist_atom_number_density.png"),
        ("volume_per_atom", "hist_volume_per_atom.png"),
        ("d_min", "hist_dmin.png"),
    )
    for key, filename in plots:
        values = _finite(rows, key)
        if not values.size:
            continue
        plt.figure()
        plt.hist(values, bins=min(50, max(10, int(math.sqrt(values.size)))))
        plt.xlabel(key)
        plt.ylabel("samples")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()
    density = _finite(rows, "atom_number_density")
    d_min = np.asarray([row.get("d_min", np.nan) for row in rows], dtype=float)
    all_density = np.asarray([row.get("atom_number_density", np.nan) for row in rows], dtype=float)
    valid = np.isfinite(d_min) & np.isfinite(all_density)
    if density.size and valid.any():
        plt.figure()
        plt.scatter(all_density[valid], d_min[valid], s=10, alpha=0.6)
        plt.xlabel("atom_number_density")
        plt.ylabel("d_min (Angstrom)")
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_density_vs_dmin.png", dpi=160)
        plt.close()


def diagnose_geometry_records(
    records: Sequence[Dict[str, Any]],
    output_dir: str,
    failures: Optional[List[Dict[str, Any]]] = None,
    total_samples: Optional[int] = None,
    train_csv: str = "",
    make_plots: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Diagnose records and write the required CSV/JSON/JSONL artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failure_rows = list(failures or [])
    train_stats = load_train_vpa_stats(train_csv) if train_csv else None
    rows = []
    for index, record in enumerate(records):
        try:
            rows.append(
                diagnose_geometry_sample(
                    lattice=record["lattice"],
                    frac_coords=record["frac_coords"],
                    num_atoms=record.get("num_atoms"),
                    atom_types=record.get("atom_types"),
                    sample_id=str(record.get("sample_id", index)),
                    source=str(record.get("source", "raw_arrays")),
                    train_vpa_stats=train_stats,
                )
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "sample_id": str(record.get("sample_id", index)),
                    "source": str(record.get("source", "raw_arrays")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    all_fieldnames = []
    for row in rows:
        for key in row:
            if key not in all_fieldnames:
                all_fieldnames.append(key)
    with (output / "geometry_pre_correction_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fieldnames)
        if all_fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    with (output / "geometry_pre_correction_failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failure_rows:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    effective_total = int(total_samples if total_samples is not None else len(records) + len(failures or []))
    summary = summarize_geometry_rows(rows, effective_total, len(failure_rows))
    if train_stats is not None:
        summary["train_vpa_stats_by_num_atoms"] = {str(k): v for k, v in train_stats.items()}
    with (output / "geometry_pre_correction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if make_plots:
        _write_plots(rows, output)
    return rows, summary, failure_rows


def records_from_npz(path: str) -> List[Dict[str, Any]]:
    """Read padded or ragged raw geometry arrays from NPZ."""
    records = []
    with np.load(path, allow_pickle=True) as arrays:
        lattice_key = next((key for key in ("lattices", "lattice", "cell") if key in arrays), None)
        if lattice_key is None:
            if "lengths" not in arrays or "angles" not in arrays:
                raise ValueError("NPZ needs lattices/lattice/cell or lengths+angles")
            lattices = np.stack(
                [lattice_from_lengths_angles(l, a) for l, a in zip(arrays["lengths"], arrays["angles"])]
            )
        else:
            lattices = np.asarray(arrays[lattice_key])
        convention = str(arrays["lattice_convention"].item()) if "lattice_convention" in arrays else "columns"
        if convention.lower().startswith("row"):
            lattices = np.swapaxes(lattices, -1, -2)
        if "frac_coords" not in arrays:
            raise ValueError("NPZ needs frac_coords")
        frac_coords = arrays["frac_coords"]
        if "num_atoms" in arrays:
            num_atoms = np.asarray(arrays["num_atoms"], dtype=int).reshape(-1)
        else:
            num_atoms = np.asarray([len(value) for value in frac_coords], dtype=int)
        atom_types = arrays["atom_types"] if "atom_types" in arrays else None
        sample_ids = arrays["sample_ids"] if "sample_ids" in arrays else np.arange(len(num_atoms)).astype(str)
        for index, n in enumerate(num_atoms):
            records.append(
                {
                    "sample_id": str(sample_ids[index]),
                    "source": str(path),
                    "lattice": np.asarray(lattices[index], dtype=float),
                    "frac_coords": np.asarray(frac_coords[index], dtype=float)[:n],
                    "num_atoms": int(n),
                    "atom_types": None if atom_types is None else np.asarray(atom_types[index], dtype=int)[:n],
                }
            )
    return records
