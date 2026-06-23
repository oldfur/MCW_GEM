#!/usr/bin/env python3
"""Compare unconditional p(L) and atom-count-conditioned p(L|n).

This script calls the project's existing ``VariationalDiffusion_L.sample``
method directly.  It samples only lengths/angles and never enters fractional
coordinate generation or atom-type decoding.

Example:
    conda run -n mpgem python scripts/compare_lattice_volume_by_n.py \
      --old-lattice-ckpt outputs/train_LatticeGen_mp20/diffusion_L/generative_model_ema.npy \
      --new-lattice-ckpt outputs/lattice_cond_n/diffusion_L/generative_model_ema_epoch220.npy \
      --config configs/lattice_train_cond_n.yaml \
      --mp20-root mp20 \
      --n-min 1 --n-max 20 --samples-per-n 1000 --batch-size 128 \
      --seed 42 --device cuda \
      --output-dir outputs/lattice_cond_n_diagnostics/compare_old_vs_new

``vpa_zscore_vs_train_n`` is calculated in log(V/N) space using the MP-20
conditional mean/std.  This is less dominated by the long upper tail than a
raw V/N z-score.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_diffusion.en_diffusion_LF_wrap import lattice_volume  # noqa: E402
from mp20.get_model import get_Lattice_model  # noqa: E402


MODEL_NAMES = ("old_uncond", "new_cond_n")
LOW_TRAIN_COUNT_THRESHOLD = 50
ANGLE_MIN_DEGREES = 50.0
ANGLE_MAX_DEGREES = 130.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare p(L) and p(L|n) against MP-20 V/N conditional distributions."
    )
    parser.add_argument("--old-lattice-ckpt", required=True, type=Path)
    parser.add_argument("--new-lattice-ckpt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument(
        "--mp20-root",
        type=Path,
        help="MP-20 root containing processed/mp20.pt; reproduces the training split.",
    )
    train_source.add_argument(
        "--train-csv",
        type=Path,
        help="CSV representing the training set; it must contain a cif column.",
    )
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--samples-per-n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lattice_cond_n_diagnostics/compare_old_vs_new"),
    )
    parser.add_argument(
        "--condition-lattice-on-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass the current n to the new conditional checkpoint (default: enabled).",
    )
    parser.add_argument(
        "--old-model-ignores-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass n through the old model interface while its condition branch is disabled.",
    )
    parser.add_argument(
        "--show-sampler-output",
        action="store_true",
        help="Do not suppress the existing sampler's per-batch print statements.",
    )
    return parser


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading --config requires PyYAML") from exc
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def extract_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load raw/nested/DataParallel checkpoints into a normalized state dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Lattice checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a state-dict mapping: {path}")
    for key in ("state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            normalized_key = key[7:] if str(key).startswith("module.") else str(key)
            state_dict[normalized_key] = value
    if not state_dict:
        raise ValueError(f"No tensor state-dict entries found in checkpoint: {path}")
    return state_dict


def checkpoint_is_conditional(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return "num_atom_embedding.weight" in state_dict


def make_lattice_model_args(
    config: Mapping[str, Any],
    conditional_architecture: bool,
    state_dict: Mapping[str, torch.Tensor],
) -> argparse.Namespace:
    """Build the subset of training args required by get_Lattice_model."""
    defaults: Dict[str, Any] = {
        "LatticeGenModel": "diffusion_L",
        "include_charges": True,
        "diffusion_steps": 1000,
        "property_pred": False,
        "freeze_gradient": False,
        "target_property": None,
        "diffusion_noise_schedule": "polynomial_2",
        "diffusion_noise_precision": 1e-5,
        "diffusion_loss_type": "l2",
        "normalize_factors": [1, 4, 16, 3, 18],
        "normalize_biases": [0, 0, 0, 0, 0],
        "prediction_threshold_t": 10,
        "use_prop_pred": 1,
        "unnormal_time_step": False,
        "only_noisy_node": False,
        "half_noisy_node": False,
        "sep_noisy_node": True,
        "atom_type_pred": False,
        "bfn_schedule": False,
        "bond_pred": False,
        "bfn_str": False,
        "optimal_sampling": False,
        "str_loss_type": "denoise_loss",
        "str_sigma_x": 0.01,
        "str_sigma_h": 0.01,
        "str_schedule_norm": False,
        "temp_index": 0,
        "lambda_l": 1.0,
        "lambda_a": 1.0,
        "condition_lattice_on_n": conditional_architecture,
        "num_atom_embed_dim": 32,
        "max_num_atoms": 20,
    }
    defaults.update(config)
    defaults["LatticeGenModel"] = config.get("LatticeGenModel", "diffusion_L")
    defaults["condition_lattice_on_n"] = bool(conditional_architecture)
    if conditional_architecture:
        embedding = state_dict.get("num_atom_embedding.weight")
        if embedding is None or embedding.ndim != 2:
            raise ValueError(
                "Conditional checkpoint is missing a 2-D num_atom_embedding.weight"
            )
        defaults["max_num_atoms"] = int(embedding.shape[0] - 1)
        defaults["num_atom_embed_dim"] = int(embedding.shape[1])
    return argparse.Namespace(**defaults)


def load_lattice_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
    expected_conditional: bool,
    label: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    state_dict = extract_state_dict(checkpoint_path)
    detected_conditional = checkpoint_is_conditional(state_dict)
    if detected_conditional != expected_conditional:
        expected = "conditional p(L|n)" if expected_conditional else "unconditional p(L)"
        detected = "conditional" if detected_conditional else "unconditional"
        raise ValueError(
            f"{label} checkpoint type mismatch: expected {expected}, detected {detected}. "
            f"Checkpoint: {checkpoint_path}"
        )
    model_args = make_lattice_model_args(
        config=config,
        conditional_architecture=expected_conditional,
        state_dict=state_dict,
    )
    if model_args.LatticeGenModel != "diffusion_L":
        raise ValueError(
            "This comparison supports LatticeGenModel=diffusion_L because that is "
            "the current p(L|n) implementation. "
            f"Got {model_args.LatticeGenModel!r}."
        )
    # diffusion_L does not use atom features, but the factory expects dataset metadata.
    dataset_info = {
        "atom_encoder": {"MAX_ATOMIC_INDEX": 94},
        "atom_decoder": ["PAD"] * 95,
        "max_n_nodes": int(model_args.max_num_atoms),
    }
    model = get_Lattice_model(model_args, device, dataset_info)
    if model is None:
        raise RuntimeError(f"Failed to construct lattice model for {label}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Strict checkpoint load failed for {label} ({checkpoint_path}). "
            "Check diffusion_steps, noise schedule, normalization values, and model type "
            f"in --config. Original error:\n{exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    metadata = {
        "label": label,
        "checkpoint": str(checkpoint_path.resolve()),
        "conditional_architecture": bool(expected_conditional),
        "max_num_atoms": int(model_args.max_num_atoms),
        "num_atom_embed_dim": (
            int(model_args.num_atom_embed_dim) if expected_conditional else None
        ),
        "diffusion_steps": int(model_args.diffusion_steps),
    }
    return model, metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gram_volume_term_np(angles: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(angles, dtype=float))
    alpha, beta, gamma = radians[:, 0], radians[:, 1], radians[:, 2]
    return (
        1
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
    )


def column_lattice_matrices_np(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of LF_wrap.compute_lattice_matrix (columns=a,b,c)."""
    lengths = np.asarray(lengths, dtype=float)
    angles_rad = np.deg2rad(np.asarray(angles, dtype=float))
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2]
    sin_gamma = np.sin(gamma)
    matrices = np.full((len(lengths), 3, 3), np.nan, dtype=float)
    safe = np.isfinite(lengths).all(axis=1) & np.isfinite(angles_rad).all(axis=1)
    safe &= np.abs(sin_gamma) > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
        cz_sq = c**2 - cx**2 - cy**2
        safe &= cz_sq > 1e-12
        matrices[:, 0, 0] = a
        matrices[:, 1, 0] = 0.0
        matrices[:, 2, 0] = 0.0
        matrices[:, 0, 1] = b * np.cos(gamma)
        matrices[:, 1, 1] = b * sin_gamma
        matrices[:, 2, 1] = 0.0
        matrices[:, 0, 2] = cx
        matrices[:, 1, 2] = cy
        matrices[:, 2, 2] = np.sqrt(np.maximum(cz_sq, 0.0))
    matrices[~safe] = np.nan
    return matrices


def metrics_from_lattice_batch(
    model_name: str,
    n: int,
    sample_offset: int,
    lengths_tensor: torch.Tensor,
    angles_tensor: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Convert one sampled lattice batch into per-sample diagnostic rows."""
    # Reuse the volume formula used by LF_wrap during actual crystal sampling.
    volumes_tensor = lattice_volume(lengths_tensor, angles_tensor)
    lengths = lengths_tensor.detach().cpu().double().numpy()
    angles = angles_tensor.detach().cpu().double().numpy()
    volumes = volumes_tensor.detach().cpu().double().numpy()
    gram_term = gram_volume_term_np(angles)
    finite = np.isfinite(lengths).all(axis=1) & np.isfinite(angles).all(axis=1)
    positive_lengths = (lengths > 0.0).all(axis=1)
    reasonable_angles = (
        (angles >= ANGLE_MIN_DEGREES) & (angles <= ANGLE_MAX_DEGREES)
    ).all(axis=1)
    valid = finite & positive_lengths & reasonable_angles
    valid &= np.isfinite(volumes) & (volumes > 0.1) & (gram_term > 1e-6)

    matrices = column_lattice_matrices_np(lengths, angles)
    condition_numbers = np.full(len(lengths), np.nan, dtype=float)
    matrix_valid = np.isfinite(matrices).all(axis=(1, 2))
    if matrix_valid.any():
        valid_indices = np.flatnonzero(matrix_valid)
        try:
            condition_numbers[valid_indices] = np.linalg.cond(matrices[valid_indices])
        except np.linalg.LinAlgError:
            # One pathological lattice must not abort a large comparison run.
            for index in valid_indices:
                try:
                    condition_numbers[index] = float(np.linalg.cond(matrices[index]))
                except np.linalg.LinAlgError:
                    condition_numbers[index] = float("nan")

    rows: List[Dict[str, Any]] = []
    for index in range(len(lengths)):
        volume = float(volumes[index])
        positive_volume = math.isfinite(volume) and volume > 0.0
        vpa = volume / n if positive_volume else float("nan")
        density = n / volume if positive_volume else float("nan")
        rows.append(
            {
                "model_name": model_name,
                "n": int(n),
                "sample_idx": int(sample_offset + index),
                "a": float(lengths[index, 0]),
                "b": float(lengths[index, 1]),
                "c": float(lengths[index, 2]),
                "alpha": float(angles[index, 0]),
                "beta": float(angles[index, 1]),
                "gamma": float(angles[index, 2]),
                "volume": volume,
                "log_volume": math.log(volume) if positive_volume else float("nan"),
                "volume_per_atom": vpa,
                "log_volume_per_atom": math.log(vpa) if vpa > 0 else float("nan"),
                "atom_number_density": density,
                "condition_number": float(condition_numbers[index]),
                "valid_lattice": bool(valid[index]),
            }
        )
    return rows


@torch.no_grad()
def sample_lattices_for_n(
    model: torch.nn.Module,
    model_name: str,
    n: int,
    samples_per_n: int,
    batch_size: int,
    device: torch.device,
    pass_num_atoms: bool,
    seed: int,
    show_sampler_output: bool,
) -> List[Dict[str, Any]]:
    """Use the existing lattice diffusion sampler in bounded batches."""
    seed_everything(seed)
    rows: List[Dict[str, Any]] = []
    sampled = 0
    while sampled < samples_per_n:
        current_batch = min(batch_size, samples_per_n - sampled)
        sample_kwargs: Dict[str, Any] = {"fix_noise": False}
        if pass_num_atoms:
            sample_kwargs["num_atoms"] = torch.full(
                (current_batch,), n, device=device, dtype=torch.long
            )
        output_context = contextlib.nullcontext()
        if not show_sampler_output:
            output_context = contextlib.redirect_stdout(io.StringIO())
        with output_context:
            lengths, angles = model.sample(
                current_batch, device=device, **sample_kwargs
            )
        rows.extend(
            metrics_from_lattice_batch(
                model_name=model_name,
                n=n,
                sample_offset=sampled,
                lengths_tensor=lengths,
                angles_tensor=angles,
            )
        )
        sampled += current_batch
    return rows


def volume_from_lengths_angles_np(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    lengths_tensor = torch.as_tensor(lengths, dtype=torch.float64)
    angles_tensor = torch.as_tensor(angles, dtype=torch.float64)
    with torch.no_grad():
        return lattice_volume(lengths_tensor, angles_tensor).cpu().numpy()


def load_train_from_processed(
    mp20_root: Path,
    config: Mapping[str, Any],
    n_min: int,
    n_max: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the same deterministic MP-20 train split used by main_mp20.py."""
    processed_path = mp20_root / "processed" / "mp20.pt"
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed MP-20 data not found: {processed_path}. "
            "Use --train-csv or prepare MP20(root)/processed/mp20.pt."
        )
    from mp20.mp20 import MP20

    dataset = MP20(root=str(mp20_root))
    split_seed = int(config.get("seed", 1))
    requested_num_train = int(config.get("num_train", 27138))
    indices = np.arange(len(dataset))
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)
    train_indices = indices[: min(requested_num_train, len(indices))]
    n_values: List[int] = []
    lengths_values: List[np.ndarray] = []
    angles_values: List[np.ndarray] = []
    for index in tqdm(train_indices, desc="Reading MP-20 processed train split"):
        data = dataset[int(index)]
        n = int(data.num_atoms.reshape(-1)[0].item())
        if n_min <= n <= n_max:
            n_values.append(n)
            lengths_values.append(data.lengths.reshape(-1, 3)[0].cpu().numpy())
            angles_values.append(data.angles.reshape(-1, 3)[0].cpu().numpy())
    if not n_values:
        raise ValueError(
            f"No processed MP-20 training samples found for n in [{n_min}, {n_max}]"
        )
    lengths = np.asarray(lengths_values, dtype=float)
    angles = np.asarray(angles_values, dtype=float)
    volumes = volume_from_lengths_angles_np(lengths, angles)
    frame = pd.DataFrame({"n": n_values, "volume": volumes})
    frame["volume_per_atom"] = frame["volume"] / frame["n"]
    frame["atom_number_density"] = frame["n"] / frame["volume"]
    metadata = {
        "source_type": "mp20_processed_train_split",
        "source": str(mp20_root.resolve()),
        "dataset_size": int(len(dataset)),
        "train_split_size": int(len(train_indices)),
        "split_seed": split_seed,
        "requested_num_train": requested_num_train,
        "selected_n_range_count": int(len(frame)),
        "parse_failures": 0,
    }
    return frame, metadata


def load_train_from_csv(
    train_csv: Path,
    n_min: int,
    n_max: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse a CSV that explicitly represents the MP-20 training split."""
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    source = pd.read_csv(train_csv)
    if "cif" not in source.columns:
        raise ValueError(
            f"Training CSV must contain a 'cif' column. Found columns: {source.columns.tolist()}"
        )
    from pymatgen.core import Structure

    rows: List[Dict[str, Any]] = []
    failures = 0
    for row_index, cif_text in tqdm(
        source["cif"].items(), total=len(source), desc="Parsing MP-20 train CIFs"
    ):
        try:
            if pd.isna(cif_text):
                raise ValueError("empty CIF")
            structure = Structure.from_str(str(cif_text), fmt="cif")
            n = int(len(structure))
            if n_min <= n <= n_max:
                volume = float(structure.volume)
                rows.append(
                    {
                        "n": n,
                        "volume": volume,
                        "volume_per_atom": volume / n,
                        "atom_number_density": n / volume,
                    }
                )
        except Exception:
            failures += 1
    if not rows:
        raise ValueError(
            f"No valid CIF training samples found for n in [{n_min}, {n_max}] in {train_csv}"
        )
    metadata = {
        "source_type": "train_csv",
        "source": str(train_csv.resolve()),
        "csv_rows": int(len(source)),
        "selected_n_range_count": int(len(rows)),
        "parse_failures": int(failures),
        "note": "The supplied CSV is treated as the training set without an additional split.",
    }
    return pd.DataFrame(rows), metadata


def distribution_stats(values: Iterable[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p05": float(np.quantile(array, 0.05)),
        f"{prefix}_p95": float(np.quantile(array, 0.95)),
    }


def build_train_stats(
    train_frame: pd.DataFrame, n_min: int, n_max: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for n in range(n_min, n_max + 1):
        group = train_frame[train_frame["n"] == n]
        row: Dict[str, Any] = {
            "n": int(n),
            "train_count": int(len(group)),
            "low_train_count": bool(len(group) < LOW_TRAIN_COUNT_THRESHOLD),
        }
        row.update(distribution_stats(group["volume"], "train_volume"))
        row.update(distribution_stats(group["volume_per_atom"], "train_vpa"))
        row.update(
            distribution_stats(group["atom_number_density"], "train_atom_density")
        )
        positive_vpa = group.loc[group["volume_per_atom"] > 0, "volume_per_atom"]
        log_vpa = np.log(positive_vpa.to_numpy(dtype=float))
        row["train_log_vpa_mean"] = (
            float(np.mean(log_vpa)) if log_vpa.size else float("nan")
        )
        row["train_log_vpa_std"] = (
            float(np.std(log_vpa)) if log_vpa.size else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_samples_with_train_reference(
    samples: pd.DataFrame, train_stats: pd.DataFrame
) -> pd.DataFrame:
    reference = train_stats.set_index("n")
    output = samples.copy()
    below: List[Any] = []
    above: List[Any] = []
    zscores: List[float] = []
    for row in output.itertuples(index=False):
        stats = reference.loc[int(row.n)]
        has_reference = int(stats["train_count"]) > 0
        finite_vpa = math.isfinite(float(row.volume_per_atom)) and row.volume_per_atom > 0
        if bool(row.valid_lattice) and has_reference and finite_vpa:
            below.append(bool(row.volume_per_atom < stats["train_vpa_p05"]))
            above.append(bool(row.volume_per_atom > stats["train_vpa_p95"]))
            log_std = float(stats["train_log_vpa_std"])
            if math.isfinite(log_std) and log_std > 0:
                zscores.append(
                    float(
                        (row.log_volume_per_atom - stats["train_log_vpa_mean"])
                        / log_std
                    )
                )
            else:
                zscores.append(float("nan"))
        else:
            below.append(None)
            above.append(None)
            zscores.append(float("nan"))
    output["below_train_vpa_p05"] = below
    output["above_train_vpa_p95"] = above
    output["vpa_zscore_vs_train_n"] = zscores
    return output


def build_model_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_name, n), group in samples.groupby(["model_name", "n"], sort=True):
        valid = group[group["valid_lattice"]]
        row: Dict[str, Any] = {
            "model_name": str(model_name),
            "n": int(n),
            "count": int(len(group)),
            "valid_lattice_rate": float(group["valid_lattice"].mean()),
        }
        row.update(distribution_stats(valid["volume"], "volume"))
        row.update(distribution_stats(valid["volume_per_atom"], "vpa"))
        row.update(
            distribution_stats(valid["atom_number_density"], "atom_density")
        )
        below = valid["below_train_vpa_p05"].dropna().astype(bool)
        above = valid["above_train_vpa_p95"].dropna().astype(bool)
        zscores = valid["vpa_zscore_vs_train_n"].dropna().to_numpy(dtype=float)
        row["ratio_below_train_vpa_p05"] = (
            float(below.mean()) if len(below) else float("nan")
        )
        row["ratio_above_train_vpa_p95"] = (
            float(above.mean()) if len(above) else float("nan")
        )
        row["mean_vpa_zscore"] = (
            float(np.mean(zscores)) if zscores.size else float("nan")
        )
        row["median_vpa_zscore"] = (
            float(np.median(zscores)) if zscores.size else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 2:
        return float("nan")
    x_rank = pd.Series(x_array[valid]).rank(method="average").to_numpy()
    y_rank = pd.Series(y_array[valid]).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def model_trend_metrics(
    model_name: str,
    model_summary: pd.DataFrame,
    train_stats: pd.DataFrame,
) -> Dict[str, float]:
    model = model_summary[model_summary["model_name"] == model_name]
    merged = model.merge(train_stats, on="n", how="inner")
    merged = merged[merged["train_count"] > 0]
    positive = (
        (merged["volume_median"] > 0)
        & (merged["vpa_median"] > 0)
        & (merged["train_volume_median"] > 0)
        & (merged["train_vpa_median"] > 0)
    )
    merged = merged[positive]
    return {
        "n_vs_median_volume_spearman": rank_correlation(
            merged["n"], merged["volume_median"]
        ),
        "n_vs_median_vpa_spearman": rank_correlation(
            merged["n"], merged["vpa_median"]
        ),
        "train_n_vs_median_volume_spearman": rank_correlation(
            merged["n"], merged["train_volume_median"]
        ),
        "median_log_volume_mae_vs_train": finite_mean(
            np.abs(
                np.log(merged["volume_median"].to_numpy())
                - np.log(merged["train_volume_median"].to_numpy())
            )
        ),
        "median_log_vpa_mae_vs_train": finite_mean(
            np.abs(
                np.log(merged["vpa_median"].to_numpy())
                - np.log(merged["train_vpa_median"].to_numpy())
            )
        ),
    }


def subset_comparison(
    samples: pd.DataFrame,
    model_summary: pd.DataFrame,
    train_stats: pd.DataFrame,
    n_threshold: int,
) -> Dict[str, Any]:
    subset_samples = samples[samples["n"] >= n_threshold]
    subset_summary = model_summary[model_summary["n"] >= n_threshold]
    result: Dict[str, Any] = {"n_threshold": int(n_threshold)}
    per_model: Dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        model_samples = subset_samples[
            (subset_samples["model_name"] == model_name)
            & subset_samples["valid_lattice"]
        ]
        model_by_n = subset_summary[subset_summary["model_name"] == model_name]
        merged = model_by_n.merge(train_stats, on="n", how="inner")
        merged = merged[
            (merged["train_count"] > 0)
            & (merged["volume_median"] > 0)
            & (merged["train_volume_median"] > 0)
            & (merged["vpa_median"] > 0)
            & (merged["train_vpa_median"] > 0)
        ]
        below = model_samples["below_train_vpa_p05"].dropna().astype(bool)
        above = model_samples["above_train_vpa_p95"].dropna().astype(bool)
        per_model[model_name] = {
            "sample_count": int(len(model_samples)),
            "mean_ratio_below_train_vpa_p05": (
                float(below.mean()) if len(below) else None
            ),
            "mean_ratio_above_train_vpa_p95": (
                float(above.mean()) if len(above) else None
            ),
            "median_log_vpa_mae_vs_train": (
                finite_mean(
                    np.abs(
                        np.log(merged["vpa_median"].to_numpy())
                        - np.log(merged["train_vpa_median"].to_numpy())
                    )
                )
                if len(merged)
                else None
            ),
            "median_log_volume_mae_vs_train": (
                finite_mean(
                    np.abs(
                        np.log(merged["volume_median"].to_numpy())
                        - np.log(merged["train_volume_median"].to_numpy())
                    )
                )
                if len(merged)
                else None
            ),
        }
    result["models"] = per_model
    old = per_model["old_uncond"]
    new = per_model["new_cond_n"]
    result["new_minus_old_ratio_below_p05"] = _optional_difference(
        new["mean_ratio_below_train_vpa_p05"],
        old["mean_ratio_below_train_vpa_p05"],
    )
    result["new_minus_old_ratio_above_p95"] = _optional_difference(
        new["mean_ratio_above_train_vpa_p95"],
        old["mean_ratio_above_train_vpa_p95"],
    )
    result["new_minus_old_log_vpa_mae"] = _optional_difference(
        new["median_log_vpa_mae_vs_train"], old["median_log_vpa_mae_vs_train"]
    )
    result["new_minus_old_log_volume_mae"] = _optional_difference(
        new["median_log_volume_mae_vs_train"],
        old["median_log_volume_mae_vs_train"],
    )
    return result


def _optional_difference(first: Any, second: Any) -> Any:
    if first is None or second is None:
        return None
    if not (math.isfinite(float(first)) and math.isfinite(float(second))):
        return None
    return float(first - second)


def aggregate_model_metrics(samples: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    group = samples[(samples["model_name"] == model_name) & samples["valid_lattice"]]
    below = group["below_train_vpa_p05"].dropna().astype(bool)
    above = group["above_train_vpa_p95"].dropna().astype(bool)
    return {
        "count": int((samples["model_name"] == model_name).sum()),
        "valid_count": int(len(group)),
        "valid_lattice_rate": float(
            samples.loc[samples["model_name"] == model_name, "valid_lattice"].mean()
        ),
        "median_volume": float(group["volume"].median()) if len(group) else None,
        "median_vpa": float(group["volume_per_atom"].median()) if len(group) else None,
        "median_atom_density": (
            float(group["atom_number_density"].median()) if len(group) else None
        ),
        "ratio_below_train_vpa_p05": float(below.mean()) if len(below) else None,
        "ratio_above_train_vpa_p95": float(above.mean()) if len(above) else None,
    }


def build_comparison_summary(
    args: argparse.Namespace,
    samples: pd.DataFrame,
    model_summary: pd.DataFrame,
    train_stats: pd.DataFrame,
    checkpoint_metadata: Mapping[str, Any],
    train_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    trends = {
        model_name: model_trend_metrics(model_name, model_summary, train_stats)
        for model_name in MODEL_NAMES
    }
    subsets = {
        "n_ge_12": subset_comparison(samples, model_summary, train_stats, 12),
        "n_ge_16": subset_comparison(samples, model_summary, train_stats, 16),
    }
    old_trend = trends["old_uncond"]
    new_trend = trends["new_cond_n"]
    large = subsets["n_ge_12"]
    large_below_delta = large["new_minus_old_ratio_below_p05"]
    large_mae_delta = large["new_minus_old_log_vpa_mae"]
    old_mismatch = bool(
        math.isfinite(old_trend["n_vs_median_volume_spearman"])
        and math.isfinite(old_trend["n_vs_median_vpa_spearman"])
        and abs(old_trend["n_vs_median_volume_spearman"]) < 0.35
        and old_trend["n_vs_median_vpa_spearman"] < -0.5
    )
    new_volume_trend_improved = bool(
        math.isfinite(new_trend["n_vs_median_volume_spearman"])
        and math.isfinite(old_trend["n_vs_median_volume_spearman"])
        and new_trend["n_vs_median_volume_spearman"]
        > old_trend["n_vs_median_volume_spearman"] + 0.2
    )
    new_vpa_closer = bool(
        math.isfinite(new_trend["median_log_vpa_mae_vs_train"])
        and math.isfinite(old_trend["median_log_vpa_mae_vs_train"])
        and new_trend["median_log_vpa_mae_vs_train"]
        < old_trend["median_log_vpa_mae_vs_train"]
    )
    large_n_below_improved = bool(
        large_below_delta is not None and large_below_delta <= -0.05
    )
    repair_effective = bool(
        new_volume_trend_improved and new_vpa_closer and large_n_below_improved
    )
    new_above_overall = aggregate_model_metrics(samples, "new_cond_n")[
        "ratio_above_train_vpa_p95"
    ]
    new_above_large = large["models"]["new_cond_n"][
        "mean_ratio_above_train_vpa_p95"
    ]
    overexpansion = bool(
        (new_above_overall is not None and new_above_overall >= 0.20)
        or (new_above_large is not None and new_above_large >= 0.20)
    )
    return {
        "analysis": "old unconditional p(L) vs new conditional p(L|n)",
        "parameters": {
            "n_min": int(args.n_min),
            "n_max": int(args.n_max),
            "samples_per_n": int(args.samples_per_n),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "condition_lattice_on_n": bool(args.condition_lattice_on_n),
            "old_model_ignores_n": bool(args.old_model_ignores_n),
            "vpa_zscore_definition": "z-score of log(V/N) within train samples at the same n",
        },
        "checkpoints": dict(checkpoint_metadata),
        "train_data": dict(train_metadata),
        "overall": {
            model_name: aggregate_model_metrics(samples, model_name)
            for model_name in MODEL_NAMES
        },
        "trend_by_n": trends,
        "large_n_subsets": subsets,
        "judgments": {
            "thresholds_are_heuristic_not_hypothesis_tests": True,
            "old_n_lattice_mismatch_detected": old_mismatch,
            "new_volume_trend_improved": new_volume_trend_improved,
            "new_vpa_closer_to_train": new_vpa_closer,
            "large_n_below_train_p05_improved_by_at_least_0.05": large_n_below_improved,
            "conditional_lattice_repair_effective": repair_effective,
            "new_model_overexpansion_risk": overexpansion,
            "overexpansion_rule": "new ratio above train V/N p95 >= 0.20 overall or for n>=12",
            "notes": {
                "old_mismatch": (
                    "Old median volume is nearly n-independent while median V/N falls with n."
                    if old_mismatch
                    else "The configured heuristic did not detect the canonical old-model mismatch."
                ),
                "repair": (
                    "New p(L|n) improves volume trend, V/N agreement, and large-n low-V/N failures."
                    if repair_effective
                    else "At least one repair criterion was not met; inspect per-n curves and ratios."
                ),
                "overexpansion": (
                    "New model often exceeds train V/N p95; possible lattice overexpansion."
                    if overexpansion
                    else "No high-frequency V/N-over-train-p95 signal under the configured rule."
                ),
            },
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_plots(
    samples: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_stats: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"train": "black", "old_uncond": "tab:orange", "new_cond_n": "tab:blue"}

    def line_plot(train_key: str, model_key: str, ylabel: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        train_valid = train_stats[train_stats["train_count"] > 0]
        axis.plot(
            train_valid["n"], train_valid[train_key], marker="o", label="train", color=colors["train"]
        )
        for model_name in MODEL_NAMES:
            group = model_summary[model_summary["model_name"] == model_name]
            axis.plot(
                group["n"], group[model_key], marker="o", label=model_name, color=colors[model_name]
            )
        axis.set_xlabel("num atoms n")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    line_plot("train_volume_median", "volume_median", "median volume (A^3)", "volume_median_by_n.png")
    line_plot("train_vpa_median", "vpa_median", "median V/N (A^3/atom)", "vpa_median_by_n.png")
    line_plot(
        "train_atom_density_median",
        "atom_density_median",
        "median atom density (atom/A^3)",
        "atom_density_median_by_n.png",
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name in MODEL_NAMES:
        group = model_summary[model_summary["model_name"] == model_name]
        axis.plot(
            group["n"],
            group["ratio_below_train_vpa_p05"],
            marker="o",
            label=model_name,
            color=colors[model_name],
        )
    axis.set_xlabel("num atoms n")
    axis.set_ylabel("ratio V/N below train p05")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_below_train_vpa_p05_by_n.png", dpi=180)
    plt.close(fig)

    def scatter_plot(y_key: str, ylabel: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        rng = np.random.RandomState(0)
        train_x = train_frame["n"].to_numpy(dtype=float) + rng.uniform(-0.12, 0.12, len(train_frame))
        axis.scatter(
            train_x,
            train_frame[y_key],
            s=4,
            alpha=0.07,
            label="train",
            color=colors["train"],
            rasterized=True,
        )
        for model_name in MODEL_NAMES:
            group = samples[
                (samples["model_name"] == model_name) & samples["valid_lattice"]
            ]
            x = group["n"].to_numpy(dtype=float) + rng.uniform(-0.12, 0.12, len(group))
            axis.scatter(
                x,
                group[y_key],
                s=4,
                alpha=0.06,
                label=model_name,
                color=colors[model_name],
                rasterized=True,
            )
        axis.set_xlabel("num atoms n")
        axis.set_ylabel(ylabel)
        axis.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    scatter_plot("volume", "volume (A^3)", "scatter_n_vs_volume.png")
    scatter_plot("volume_per_atom", "V/N (A^3/atom)", "scatter_n_vs_vpa.png")


def print_core_judgments(summary: Mapping[str, Any]) -> None:
    trends = summary["trend_by_n"]
    subsets = summary["large_n_subsets"]
    judgments = summary["judgments"]
    print("\n=== Lattice p(L) vs p(L|n) comparison ===")
    for model_name in MODEL_NAMES:
        trend = trends[model_name]
        print(
            f"{model_name}: Spearman(n, median V)={trend['n_vs_median_volume_spearman']:.4f}, "
            f"Spearman(n, median V/N)={trend['n_vs_median_vpa_spearman']:.4f}, "
            f"median-log-V/N MAE vs train={trend['median_log_vpa_mae_vs_train']:.4f}"
        )
    for subset_name in ("n_ge_12", "n_ge_16"):
        subset = subsets[subset_name]
        print(
            f"{subset_name}: new-old below-p05 ratio="
            f"{subset['new_minus_old_ratio_below_p05']}, "
            f"new-old log-V/N MAE={subset['new_minus_old_log_vpa_mae']}"
        )
    print(f"old n-lattice mismatch: {judgments['old_n_lattice_mismatch_detected']}")
    print(f"conditional repair effective: {judgments['conditional_lattice_repair_effective']}")
    print(f"new-model overexpansion risk: {judgments['new_model_overexpansion_risk']}")
    print(f"repair assessment: {judgments['notes']['repair']}")


def validate_cli_args(args: argparse.Namespace) -> torch.device:
    if args.n_min < 1 or args.n_max < args.n_min:
        raise ValueError(f"Invalid n range: [{args.n_min}, {args.n_max}]")
    if args.samples_per_n < 1 or args.batch_size < 1:
        raise ValueError("--samples-per-n and --batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device is CUDA but torch.cuda.is_available() is False")
    return device


def main() -> None:
    args = build_parser().parse_args()
    device = validate_cli_args(args)
    config = load_yaml_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    old_model, old_metadata = load_lattice_model(
        checkpoint_path=args.old_lattice_ckpt,
        config=config,
        device=device,
        expected_conditional=False,
        label="old_uncond",
    )
    new_model, new_metadata = load_lattice_model(
        checkpoint_path=args.new_lattice_ckpt,
        config=config,
        device=device,
        expected_conditional=True,
        label="new_cond_n",
    )
    if args.n_max > int(new_metadata["max_num_atoms"]):
        raise ValueError(
            f"--n-max={args.n_max} exceeds conditional embedding max "
            f"{new_metadata['max_num_atoms']}"
        )

    if args.mp20_root is not None:
        train_frame, train_metadata = load_train_from_processed(
            args.mp20_root, config, args.n_min, args.n_max
        )
    else:
        train_frame, train_metadata = load_train_from_csv(
            args.train_csv, args.n_min, args.n_max
        )
    train_stats = build_train_stats(train_frame, args.n_min, args.n_max)
    train_stats.to_csv(output_dir / "train_lattice_stats_by_n.csv", index=False)

    all_rows: List[Dict[str, Any]] = []
    for n in tqdm(range(args.n_min, args.n_max + 1), desc="Sampling by atom count"):
        paired_seed = int(args.seed + n * 100_003)
        all_rows.extend(
            sample_lattices_for_n(
                old_model,
                "old_uncond",
                n,
                args.samples_per_n,
                args.batch_size,
                device,
                pass_num_atoms=bool(args.old_model_ignores_n),
                seed=paired_seed,
                show_sampler_output=args.show_sampler_output,
            )
        )
        # Reset to the same per-n seed so old/new use paired diffusion noise.
        all_rows.extend(
            sample_lattices_for_n(
                new_model,
                "new_cond_n",
                n,
                args.samples_per_n,
                args.batch_size,
                device,
                pass_num_atoms=bool(args.condition_lattice_on_n),
                seed=paired_seed,
                show_sampler_output=args.show_sampler_output,
            )
        )

    samples = annotate_samples_with_train_reference(pd.DataFrame(all_rows), train_stats)
    samples.to_csv(output_dir / "lattice_samples_per_n.csv", index=False)
    model_summary = build_model_summary(samples)
    model_summary.to_csv(output_dir / "model_summary_by_n.csv", index=False)

    comparison_summary = build_comparison_summary(
        args=args,
        samples=samples,
        model_summary=model_summary,
        train_stats=train_stats,
        checkpoint_metadata={"old_uncond": old_metadata, "new_cond_n": new_metadata},
        train_metadata=train_metadata,
    )
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(comparison_summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_plots(samples, train_frame, train_stats, model_summary, output_dir)
    print_core_judgments(comparison_summary)
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
