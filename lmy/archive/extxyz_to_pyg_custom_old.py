import torch
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from tqdm import tqdm

# --- 常量定义 ---
EV_A3_TO_GPA = 160.21766208  # 1 eV/A^3 = 160.2 GPa

def extxyz_to_pyg_custom(xyz_file_path, cutoff=6.0):
    print(f"Reading {xyz_file_path} ...")
    frames = read(xyz_file_path, index=':')
    data_list = []
    
    for atoms in tqdm(frames):
        # 基础检查：保留结构类型检查，防止混入不需要的数据结构
        # if atoms.info.get("structure_type", "single_component") != "single_component": 
        #     continue

        # 保留 Z 值一致性检查 (可选，视你的数据集质量而定)
        # z_val = atoms.info.get("Z_value", None)
        # z_csd = atoms.info.get("Z_in_csd", None)
        
        # if z_val is not None and z_csd is not None:
        #     if z_val != z_csd:
        #         continue
        
        # [修改 1] 移除了对 'mol_id' 的存在性检查和最大值校验
        # 这样即使没有拓扑信息的普通 XYZ 文件也能运行

        # --- A. 基础张量 ---
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
        
        # 晶胞: [1, 3, 3] 必须保持 float32
        cell_matrix = atoms.get_cell().array 
        cell = torch.tensor(cell_matrix, dtype=torch.float).unsqueeze(0) 
        vol_value = np.abs(np.linalg.det(cell_matrix))
        volume = torch.tensor(vol_value, dtype=torch.float)
        
        # [修改 2] 彻底移除了 mol_id 的获取与重排代码

        # --- B. 标签 (Energy, Force, Stress) ---
        y = torch.tensor(atoms.info.get('REF_energy', 0.0), dtype=torch.float)
        forces = torch.tensor(atoms.arrays.get('REF_forces', np.zeros_like(pos)), dtype=torch.float)
        
        stress_info = atoms.info.get('REF_stress', None)
        stress_tensor = torch.zeros((1, 3, 3), dtype=torch.float)
        if stress_info is not None:
            if isinstance(stress_info, str):
                s_list = [float(x) for x in stress_info.split()]
            else:
                s_list = list(stress_info)
            
            # ASE / VASP Voigt Order 处理
            if len(s_list) == 9:
                s_mat = torch.tensor(s_list, dtype=torch.float).view(3, 3)
            elif len(s_list) == 6:
                s_mat = torch.tensor([
                    [s_list[0], s_list[5], s_list[4]],
                    [s_list[5], s_list[1], s_list[3]],
                    [s_list[4], s_list[3], s_list[2]]
                ], dtype=torch.float)
            else:
                s_mat = torch.zeros((3, 3), dtype=torch.float)
            stress_tensor = s_mat.unsqueeze(0)

        # --- C. 纯几何图构建 ---
        # 只根据 cutoff 构建邻居列表，不考虑化学成键
        i_idx, j_idx, d_val, S_integers = neighbor_list('ijdS', atoms, cutoff)
        
        edge_index = torch.stack([torch.tensor(i_idx), torch.tensor(j_idx)], dim=0).long()
        distances = torch.tensor(d_val, dtype=torch.float)
        
        # 保留 Periodic Shifts 以支持 PBC 下的微分
        shifts_int = torch.tensor(S_integers, dtype=torch.float)
        
        # --- D. 边类型 (Edge Type) ---
        # [修改 3] 移除了基于半径和分子 ID 的判定逻辑
        # 所有在 cutoff 内的边都被视为同一种类型 (0)
        num_edges = edge_index.shape[1]
        edge_types = torch.zeros(num_edges, dtype=torch.long)
        
        data = Data(
            z=z, pos=pos, cell=cell, volume=volume,
            edge_index=edge_index, 
            edge_type=edge_types,          # 全为 0
            edge_attr=distances.unsqueeze(-1),
            shifts_int=shifts_int, 
            y=y, force=forces, stress=stress_tensor, 
            # [修改 4] 移除了 mol_id, is_intra, num_molecules 等拓扑字段
            batch=torch.zeros(z.size(0), dtype=torch.long)
        )
        data_list.append(data)
        
    return data_list
