#!/usr/bin/env python3
import os

import numpy as np
import torch

from equivariant_diffusion.en_diffusion_LF_wrap import EquiTransVariationalDiffusion_LF_wrap
from mp20.analyze_test import _atom_type_symbols
from mp20.crystal import array_dict_to_crystal


def build_stub_model():
    model = EquiTransVariationalDiffusion_LF_wrap.__new__(EquiTransVariationalDiffusion_LF_wrap)
    model.num_classes = 12
    model.unknown_atom_type_idx = 0
    model.h_class_idx = 1
    model.atom_decoder = ["PAD", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na"]
    model.known_atom_class_ids = [1, 6, 7, 8, 11]
    model.disable_all_h_guard_arg = False
    model.all_h_guard_enabled = True
    model.all_h_guard_topk = 4
    model.all_h_guard_min_non_h = 1
    model.atom_type_repair_topk = 4
    model.debug_atom_types = False
    model.debug_atom_dir = ""
    model._last_prepare_inputs_debug = {}
    model._last_dynamics_atom_debug_info = {}
    model._atom_type_all_h_guard_summary = {}
    model._reset_atom_type_all_h_guard_summary()
    return model


def main():
    os.environ["MCW_ALL_H_GUARD_ENABLED"] = "1"
    os.environ["MCW_ALL_H_GUARD_DISABLED_ARG"] = "0"

    model = build_stub_model()
    logits = torch.full((1, 3, model.num_classes), -8.0, dtype=torch.float32)
    logits[0, :, model.h_class_idx] = 5.0
    logits[0, 0, 8] = 4.95
    logits[0, 1, 6] = 4.85
    logits[0, 2, 7] = 4.75
    node_mask = torch.ones((1, 3, 1), dtype=torch.float32)

    one_hot = model._finalize_atom_type_logits(
        logits,
        node_mask,
        round_index=0,
        source_tag="all_h_guard_smoke_test",
        apply_repair=True,
    )

    one_hot_valid = one_hot[0][node_mask[0].squeeze(-1).bool()].detach().cpu().numpy()
    atom_types = np.argmax(one_hot_valid, axis=-1)
    species_symbols = _atom_type_symbols(atom_types.tolist())
    assert not np.all(atom_types == 1), (
        "all-H guard failed: analyze_test-style argmax still decodes to all H "
        f"({species_symbols})"
    )

    crystal = array_dict_to_crystal(
        {
            "frac_coords": np.array(
                [[0.0, 0.0, 0.0], [0.3, 0.4, 0.5], [0.6, 0.2, 0.8]],
                dtype=np.float64,
            ),
            "atom_types": atom_types,
            "lengths": np.array([5.0, 5.0, 5.0], dtype=np.float64),
            "angles": np.array([90.0, 90.0, 90.0], dtype=np.float64),
            "sample_idx": "all_h_guard_smoke_test",
        },
        save=False,
        save_dir_name="",
    )

    final_atom_types = np.array(crystal.atom_types, dtype=int)
    assert not np.all(final_atom_types == 1), (
        "all-H guard failed: Crystal conversion still yields all H "
        f"({final_atom_types.tolist()})"
    )
    if getattr(crystal, "constructed", False) and hasattr(crystal, "structure"):
        structure_atom_types = [int(v) for v in crystal.structure.atomic_numbers]
        assert not all(v == 1 for v in structure_atom_types), (
            "all-H guard failed: final Structure species are still all H "
            f"({structure_atom_types})"
        )

    print("all-H guard end-to-end smoke test passed.")
    print("Decoded atom types:", atom_types.tolist())
    print("Decoded species:", species_symbols)


if __name__ == "__main__":
    main()
