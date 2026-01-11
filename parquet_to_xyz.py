#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from ase import Atoms
from ase.io import write

# ================================================================
# -------------------- CONFIG ------------------------------------
# ================================================================

DATA_ROOT = "../mptrj_dataset/data"
OUT_ROOT = "../mptrj_xyz"
SPLITS = ["train", "val", "test"]

# ================================================================
# -------------------- ROW → ASE ATOMS ----------------------------
# ================================================================

def row_to_atoms(row):
    """
    Convert one MPTrj parquet row to ASE Atoms in Extended XYZ format
    """

    # 1. 基础数据解析 - 增加 .tolist() 以确保 NumPy 能正确解析嵌套结构
    # 这里的 row["numbers"] 等如果是对象类型，直接 np.array 可能会报错
    numbers = np.array(row["numbers"], dtype=int)
    
    # 针对坐标和晶胞，先转成 list 再转 numpy，非常稳健
    positions = np.array(row["positions"].tolist(), dtype=float).reshape(-1, 3)
    cell = np.array(row["cell"].tolist(), dtype=float).reshape(3, 3)
    
    # 2. 创建 Atoms 对象
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=[True, True, True], 
    )

    # 3. 写入元数据
    atoms.info["material_id"] = row.get("mp_id")
    
    if "ionic_step" in row:
        atoms.info["ionic_step"] = int(row["ionic_step"])

    if "energy" in row:
        atoms.info["REF_energy"] = float(row["energy"])

    # 针对 stress 和 forces 同样使用 .tolist()
    if "stress" in row and row["stress"] is not None:
        stress_data = row["stress"]
        if hasattr(stress_data, 'tolist'):
            stress_data = stress_data.tolist()
        atoms.info["REF_stress"] = np.array(stress_data, dtype=float).flatten()

    if "forces" in row and row["forces"] is not None:
        forces_data = row["forces"]
        if hasattr(forces_data, 'tolist'):
            forces_data = forces_data.tolist()
        atoms.arrays["REF_forces"] = np.array(forces_data, dtype=float).reshape(-1, 3)

    return atoms

# ================================================================
# -------------------- MAIN LOGIC --------------------------------
# ================================================================

def convert_split(split):
    parquet_files = sorted(
        glob.glob(os.path.join(DATA_ROOT, f"{split}-*.parquet"))
    )

    if not parquet_files:
        print(f"[WARN] No parquet files for split={split}")
        return

    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Processing split='{split}', files={len(parquet_files)}")

    seen = set()

    for pq in parquet_files:
        df = pd.read_parquet(pq)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {os.path.basename(pq)}"):
            mp_id = row.get("mp_id")
            if mp_id is None:
                continue

            # 使用 mp_id 和 ionic_step 组合作为唯一标识，防止覆盖同 ID 的不同步骤
            step = row.get("ionic_step", 0)
            unique_key = f"{mp_id}_{step}"
            
            if unique_key in seen:
                continue
            seen.add(unique_key)

            atoms = row_to_atoms(row)

            # 文件名：split_mp-id_step.xyz
            out_path = os.path.join(
                out_dir,
                f"{split}_{mp_id}_s{step}.xyz"
            )

            # 关键：必须使用 format="extxyz" 才能生成附件中的 Lattice 和 Properties 格式 
            write(out_path, atoms, format="extxyz")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for split in SPLITS:
        convert_split(split)

    print("\n[OK] MPTrj parquet → Extended XYZ conversion finished.")


if __name__ == "__main__":
    main()