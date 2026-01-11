#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended StructureData supporting property head targets

Adds:
    targets["prop"] = multitask material properties
"""

import torch
import numpy as np

from chgnet.data.dataset import StructureData, TORCH_DTYPE
from chgnet.graph import CrystalGraph

class PropertyStructureData(StructureData):
    """
    Extends CHGNet StructureData to include property_targets

    property_targets shape:
        [N, prop_dim]
        NaN values are allowed (Masked loss will ignore)
    """

    def __init__(self, *args, property_targets=None, prop_dim=4, **kwargs):

        super().__init__(*args, **kwargs)

        if property_targets is None:
            raise ValueError("property_targets must be provided")

        if len(property_targets) != len(self.structures):
            raise RuntimeError(
                f"Inconsistent lengths: structures={len(self.structures)}, "
                f"property_targets={len(property_targets)}"
            )

        self.property_targets = np.asarray(property_targets, dtype=float)
        self.prop_dim = prop_dim

        print(
            f"{type(self).__name__} imported property targets "
            f"shape={self.property_targets.shape}"
        )

    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict]:

        crystal_graph, targets = super().__getitem__(idx)

        prop_vec = self.property_targets[self.keys[idx]]

        targets["prop"] = torch.tensor(
            prop_vec,
            dtype=TORCH_DTYPE,
        )

        return crystal_graph, targets