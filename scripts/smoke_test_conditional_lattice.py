#!/usr/bin/env python3
"""Fast CPU smoke test for p(L|n), legacy loading, and geometry diagnostics."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from equivariant_diffusion.diffusion_L import VariationalDiffusion_L  # noqa: E402
from mp20.geometry_diagnostics import (  # noqa: E402
    diagnose_geometry_records,
    records_from_npz,
    save_raw_geometry_npz,
)


def main() -> None:
    common = dict(
        in_node_nf=95,
        n_dims=3,
        timesteps=4,
        noise_schedule="polynomial_2",
        noise_precision=1e-5,
        loss_type="l2",
        norm_values=[1, 4, 16, 3, 18],
        norm_biases=[0, 0, 0, 0, 0],
        device="cpu",
        lambda_l=1,
        lambda_a=1,
    )
    model = VariationalDiffusion_L(
        **common,
        condition_lattice_on_n=True,
        num_atom_embed_dim=16,
        max_num_atoms=20,
    )
    model.train()
    lengths = torch.tensor([[4.0, 4.2, 4.4], [7.0, 7.2, 7.4]])
    angles = torch.tensor([[90.0, 90.0, 90.0], [80.0, 95.0, 105.0]])
    num_atoms = torch.tensor([2, 18])
    loss, _ = model(lengths, angles, num_atoms=num_atoms)
    assert loss.shape == (2,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.num_atom_embedding.weight.grad is not None
    model.eval()
    sampled_lengths, sampled_angles = model.sample(
        2, "cpu", fix_noise=True, num_atoms=num_atoms
    )
    assert sampled_lengths.shape == (2, 3)
    assert sampled_angles.shape == (2, 3)

    legacy = VariationalDiffusion_L(**common, condition_lattice_on_n=False)
    legacy_clone = VariationalDiffusion_L(**common, condition_lattice_on_n=False)
    legacy_clone.load_state_dict(legacy.state_dict(), strict=True)
    assert not any("num_atom_embedding" in key for key in legacy.state_dict())

    record = {
        "sample_id": "synthetic",
        "source": "smoke",
        "lattice": np.diag([10.0, 10.0, 10.0]),
        "frac_coords": np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0]]),
        "num_atoms": 2,
        "atom_types": np.array([1, 8]),
    }
    bad_record = {
        "sample_id": "singular",
        "source": "smoke",
        "lattice": np.zeros((3, 3)),
        "frac_coords": np.zeros((2, 3)),
        "num_atoms": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        rows, summary, failures = diagnose_geometry_records(
            [record, bad_record], tmp, make_plots=False
        )
        assert len(failures) == 1 and failures[0]["sample_id"] == "singular"
        failure_lines = (Path(tmp) / "geometry_pre_correction_failures.jsonl").read_text().splitlines()
        assert len(failure_lines) == 1
        assert abs(rows[0]["d_min"] - 0.6) < 1e-8
        assert rows[0]["num_pairs_lt_0.7"] == 1
        assert summary["ratio_dmin_lt_0.7"] == 1.0
        path = save_raw_geometry_npz([record], str(Path(tmp) / "raw.npz"))
        loaded = records_from_npz(path)
        assert len(loaded) == 1 and loaded[0]["num_atoms"] == 2

    print(
        json.dumps(
            {
                "status": "passed",
                "conditional_loss": float(loss.mean()),
                "sampled_n": num_atoms.tolist(),
                "diagnostic_d_min": rows[0]["d_min"],
                "ratio_dmin_lt_0.7": summary["ratio_dmin_lt_0.7"],
                "legacy_strict_load": True,
            }
        )
    )


if __name__ == "__main__":
    main()
