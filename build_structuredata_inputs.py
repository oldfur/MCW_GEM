#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert JSON dataset → CHGNet StructureData inputs

Outputs:
    list_of_structures : list[pymatgen.Structure]
    list_of_energies   : list[float]
    list_of_forces     : list[n_atoms, 3]
    list_of_stresses   : None (no stress labels)
    list_of_magmoms    : None (no magmom labels)

These objects can be passed directly to:

    StructureData(
        structures,
        energies,
        forces,
        stresses,
        magmoms
    )

Author: your project pipeline
"""

import json
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer import Trainer
from chgnet.model.model import CHGNet
import torch
import torch.nn as nn


# ================================================================
# 配置：json 数据路径
# ================================================================
JSON_PATH = "../crystal_data.json"


def load_dataframe(json_path: str) -> pd.DataFrame:
    """Load dataset json into pandas DataFrame"""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Total records: {len(df)}")

    # 保证 structure_dict 存在
    df = df[df["structure_dict"].notna()].reset_index(drop=True)
    print(f"Valid structure entries: {len(df)}")

    return df

# ======================================================================
# ----------------------- helper functions -----------------------------
# ======================================================================

def extract_vrh(x):
    """
    bulk_modulus / shear_modulus are dicts:
        {"voigt": , "reuss": , "vrh": }
    We select vrh (standard isotropic aggregate).
    """
    if isinstance(x, dict) and "vrh" in x:
        return float(x["vrh"])
    return np.nan


# ================================================================
# 构建 StructureData 输入
# ================================================================
def build_structuredata_inputs(df: pd.DataFrame):
    """
    Convert dataframe → StructureData compatible inputs
    """

    print("Converting structure_dict → pymatgen.Structure ...")

    list_of_structures = [
        Structure.from_dict(d)
        for d in df["structure_dict"]
    ]

    print("Building energy targets (using energy_above_hull) ...")

    list_of_energies = (
        df["energy_above_hull"]
        .astype(float)
        .apply(lambda x: [x])   # ← 包成 1D 列向量
        .tolist()
    ) # [N,1]


    print("Allocating zero-force placeholders ...")

    # CHGNet expects [n_atoms, 3]
    list_of_forces = [
        np.zeros((len(struct.sites), 3)).tolist()
        for struct in list_of_structures
    ]

    # 当前数据集中不存在 → 不参与训练
    list_of_stresses = None
    list_of_magmoms = None

    print("Done.")
    print(f"Structures : {len(list_of_structures)}")
    print(f"Energies   : {len(list_of_energies)}")
    print(f"Forces     : {len(list_of_forces)}")

    return (
        list_of_structures,
        list_of_energies,
        list_of_forces,
        list_of_stresses,
        list_of_magmoms,
    )


def build_multitask_property_targets(df: pd.DataFrame):
    """
    Build multitask material property labels.

    Output matrix: [N, 4]
        0 = band_gap
        1 = bulk_modulus_vrh
        2 = shear_modulus_vrh
        3 = efermi
    """

    print("[INFO] Building multi-property targets")

    band_gap = (
        df["band_gap"]
        .astype(float)
        .tolist()
    )

    bulk_modulus = (
        df["bulk_modulus"]
        .apply(extract_vrh)
        .tolist()
    )

    shear_modulus = (
        df["shear_modulus"]
        .apply(extract_vrh)
        .tolist()
    )

    efermi = (
        df["efermi"]
        .astype(float)
        .tolist()
    )

    property_targets = np.stack(
        [
            np.array(band_gap, dtype=float),
            np.array(bulk_modulus, dtype=float),
            np.array(shear_modulus, dtype=float),
            np.array(efermi, dtype=float),
        ],
        axis=1,   # -> [N, 4]
    )

    print("[INFO] Property target coverage:")
    print("  band_gap       :", np.isfinite(property_targets[:, 0]).sum())
    print("  bulk_modulus   :", np.isfinite(property_targets[:, 1]).sum())
    print("  shear_modulus  :", np.isfinite(property_targets[:, 2]).sum())
    print("  efermi         :", np.isfinite(property_targets[:, 3]).sum())
    #   band_gap       : 80943
    #   bulk_modulus   : 12316
    #   shear_modulus  : 12316
    #   efermi         : 47481

    return property_targets


# ================================================================
# 入口（可单独运行测试）
# ================================================================
if __name__ == "__main__":

    df = load_dataframe(JSON_PATH)

    (
        list_of_structures,
        list_of_energies,
        list_of_forces,
        list_of_stresses,
        list_of_magmoms,
    ) = build_structuredata_inputs(df)

    print("\nSanity check — first entry:")
    print("  atoms 0 =", len(list_of_structures[0].sites))
    print("  energy 0 =", list_of_energies[0])
    print("  energy shape =", np.array(list_of_energies).shape)
    print("  forces 0 shape =", np.array(list_of_forces[0]).shape)

    # ---- Multi-property targets ----
    property_targets = build_multitask_property_targets(df)

    print("  property_targets shape:", property_targets.shape)
    print("  first property row    :", property_targets[0])
    # first property row    : [ 0.         23.574      13.226       6.20071114]

    print("\nNow pass these into CHGNet StructureData()...")

    dataset = StructureData(
        structures=list_of_structures,
        energies=list_of_energies,
        forces=list_of_forces,
        stresses=list_of_stresses,
        magmoms=list_of_magmoms,
    )
    
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )

    print("DataLoaders ready.")
    
    chgnet = CHGNet.load()
    
    trainer = Trainer(
        model=chgnet,
        # targets="efsm",
        targets="e",
        optimizer="Adam",
        criterion="MSE",
        learning_rate=1e-2,
        epochs=50,
        use_device="cuda",
    )

    trainer.train(train_loader, val_loader, test_loader)

    print("StructureData and training setup complete.")