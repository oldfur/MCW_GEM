import torch
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from tqdm import tqdm

def extxyz_to_pyg_custom(xyz_file_path, cutoff=6.0):
    # print(f"Reading {xyz_file_path} ...") # 注释掉防止刷屏
    try:
        frames = read(xyz_file_path, index=':')
    except Exception as e:
        print(f"❌ 读取错误 {xyz_file_path}: {e}")
        return []

    data_list = []
    
    for atoms in frames:
        # --- A. 基础张量 (压缩，有些不) ---
        
        # 1. 原子序数: int64 -> int8 (最大支持到 127 号元素)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int8)

        # 2. 坐标: 保持 float64
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        
        # 3. 晶胞: float64
        cell_matrix = atoms.get_cell().array 
        cell = torch.tensor(cell_matrix, dtype=torch.float64).unsqueeze(0) 

        # --- B. 标签 (Energy, Force, Stress) ---
        y = torch.tensor(atoms.info.get('REF_energy', 0.0), dtype=torch.float64)
        forces = torch.tensor(atoms.arrays.get('REF_forces', np.zeros_like(pos)), dtype=torch.float64)
        
        stress_info = atoms.info.get('REF_stress', None)
        stress_tensor = None # 默认为 None，节省空间，如果有才存
        
        if stress_info is not None:
            if isinstance(stress_info, str):
                s_list = [float(x) for x in stress_info.split()]
            else:
                s_list = list(stress_info)
            
            # Voigt Order 处理
            if len(s_list) == 9:
                s_mat = torch.tensor(s_list, dtype=torch.float32).view(3, 3)
            elif len(s_list) == 6:
                s_mat = torch.tensor([
                    [s_list[0], s_list[5], s_list[4]],
                    [s_list[5], s_list[1], s_list[3]],
                    [s_list[4], s_list[3], s_list[2]]
                ], dtype=torch.float32)
            else:
                s_mat = torch.zeros((3, 3), dtype=torch.float32)
            stress_tensor = s_mat.unsqueeze(0)
        else:
            continue  # 如果没有应力信息，就跳过该结构，节省空间

        # --- C. 纯几何图构建 ---
        i_idx, j_idx, d_val, S_integers = neighbor_list('ijdS', atoms, cutoff)
        
        # [关键压缩 1] edge_index: int64 -> int32
        # int32 足够索引 20 亿个原子，绝对安全
        edge_index = torch.stack([torch.tensor(i_idx), torch.tensor(j_idx)], dim=0).to(torch.int32)
        
        # [关键压缩 2] shifts_int: float32 -> int8
        # S_integers 通常是 -1, 0, 1。int8 范围 -128~127，足够了。
        # 注意：模型读取时可能需要转回 float，但这在 Dataset 读取时做，不占硬盘。
        shifts_int = torch.tensor(S_integers, dtype=torch.int32)
        
        # 距离保持 float32
        # distances = torch.tensor(d_val, dtype=torch.float32) 
        # 实际上 distances 也可以不存，训练时用 pos 和 shifts 算一下很快，
        # 不过为了这里保持一致性，还是留着，或者像你之前那样放在 edge_attr 里
        
        # --- D. 边类型 ---
        # [关键压缩 3] edge_type: int64 -> int8
        num_edges = edge_index.shape[1]
        edge_types = torch.zeros(num_edges, dtype=torch.int8)
        
        data = Data(
            z=z, 
            pos=pos, 
            cell=cell, 
            edge_index=edge_index, 
            shifts_int=shifts_int, 
            y=y, 
            force=forces
        )
        
        if stress_tensor is not None:
            data.stress = stress_tensor
            
        data_list.append(data)
        
    return data_list
