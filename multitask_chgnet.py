#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-task CHGNet:
Extend CHGNet with an additional crystal-level property head for
    band_gap / bulk_modulus / shear_modulus / efermi

Base CHGNet outputs remain unchanged:
    energy / forces / stress / magmom

Property head consumes graph embedding and produces
    [N, prop_dim] regression outputs.
"""

import torch
import torch.nn as nn
from chgnet.model.model import CHGNet


# ================================================================
# ---- Masked MSE loss (ignores NaN labels)
# ================================================================

class MaskedMSELoss(nn.Module):
    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred[mask] - target[mask]) ** 2).mean()


# ================================================================
# ---- Property prediction head (graph-level regression)
# ================================================================

class PropertyHead(nn.Module):
    """
    Input : graph embedding tensor [N, H]
    Output: multi-property regression vector [N, prop_dim]
    """

    def __init__(self, hidden_dim: int, prop_dim: int = 4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, prop_dim),
        )

    def forward(self, g_emb):
        return self.mlp(g_emb)


# ================================================================
# ---- Multi-task extension of CHGNet
# ================================================================

class MultiTaskCHGNet(CHGNet):
    """
    Subclass CHGNet but add:
        self.property_head

    forward() returns:
        base_outputs (energy/forces/...)
        property_pred (multi-property vector)
    """

    def __init__(
        self,
        prop_dim: int = 4,
        init_property_head: bool = True,
        *args,
        **kwargs,
    ):
        """
        prop_dim = number of material properties to predict
            [ band_gap , bulk_modulus , shear_modulus , efermi ]

        hidden_dim = size of CHGNet graph representation
            (default 384 for official model)
        """

        super().__init__(*args, **kwargs)

        self.prop_dim = prop_dim

        self.hidden_dim = self.atom_fea_dim  # CHGNet graph embedding dim = atom_fea_dim

        self.property_head = (
            PropertyHead(hidden_dim=self.hidden_dim, prop_dim=prop_dim)
            if init_property_head
            else None
        )

        self.masked_mse = MaskedMSELoss()

    # ------------------------------------------------------------
    # utility: extract graph embedding robustly
    # ------------------------------------------------------------
    def _get_graph_embedding(self, model_out):
        """
        Robust graph-level embedding extractor
        """

        # preferred: CHGNet pooled structure representation
        if "crystal_fea" in model_out:
            return model_out["crystal_fea"]

        # fallback: average pool atom features
        if "atom_fea" in model_out and "atoms_per_graph" in model_out:
            atom_fea = torch.cat(model_out["atom_fea"], dim=0)
            counts = model_out["atoms_per_graph"]

            owners = torch.repeat_interleave(
                torch.arange(len(counts), device=atom_fea.device),
                counts,
            )

            return torch.zeros(len(counts), atom_fea.shape[-1], device=atom_fea.device)\
                .index_add_(0, owners, atom_fea)

        raise RuntimeError(
            f"No usable graph embedding found. "
            f"Model outputs={list(model_out.keys())}"
        )

    # ------------------------------------------------------------
    # override forward()
    # ------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """
        Returns:
            base_out      = CHGNet normal outputs (energy/forces/...)
            property_pred = [N, prop_dim]
        """

        base_out = super().forward(*args, return_crystal_feas=True, **kwargs)
        

        if self.property_head is None:
            return base_out

        g_emb = self._get_graph_embedding(base_out)

        property_pred = self.property_head(g_emb)

        base_out["property_pred"] = property_pred

        return base_out


# ================================================================
# ---- convenience constructor from pretrained CHGNet
# ================================================================

def load_multitask_chgnet(prop_dim: int = 4, hidden_dim: int = 384):
    """
    Load official pretrained CHGNet weights,
    then attach a fresh property head.
    """

    base = CHGNet.load()

    mt = MultiTaskCHGNet(
        prop_dim=prop_dim,
        hidden_dim=hidden_dim,
    )

    # copy pretrained base weights
    mt.load_state_dict(base.state_dict(), strict=False)

    return mt


# ================================================================
# ---- Example usage (for reference)
# ================================================================

if __name__ == "__main__":

    print("Loading multitask CHGNet...")
    model = load_multitask_chgnet()

    print("Model ready. You may now attach MaskedMSELoss for property head.")
