import os
# 限制底层库线程，防止多进程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import random
import multiprocessing
import gc
import h5py
import numpy as np
from tqdm.auto import tqdm
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

# ==========================================
# 1. 核心转换函数：适配你的 REF_ 字段
# ==========================================
def extxyz_to_pyg_custom(fpath, cutoff=6.0):
    """
    读取你的 .xyz 文件并转换为 PyG Data 对象
    """
    try:
        atoms = read(fpath, format="extxyz")
        
        # 基础数据
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        
        # 邻居列表计算 (生成边索引和周期性位移)
        i, j, S = neighbor_list('ijS', atoms, cutoff)
        edge_index = torch.tensor(np.stack([i, j], axis=0), dtype=torch.long)
        shifts = torch.tensor(S, dtype=torch.float32)
        
        # 提取你的特定属性 (匹配 AAXTHP.xyz)
        # 能量
        energy = torch.tensor([atoms.info.get('REF_energy', 0.0)], dtype=torch.float32)
        # 力 (REF_forces)
        force = torch.tensor(atoms.arrays.get('REF_forces', np.zeros_like(atoms.positions)), dtype=torch.float32)
        # 应力 (REF_stress 是 9 维向量，转为 3x3)
        stress_raw = atoms.info.get('REF_stress', np.zeros(9))
        stress = torch.tensor(stress_raw, dtype=torch.float32).view(1, 3, 3)
        
        cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32).view(1, 3, 3)

        data = Data(
            z=z, pos=pos, edge_index=edge_index, shifts_int=shifts,
            y=energy, force=force, stress=stress, cell=cell,
            num_nodes=len(z)
        )
        return [data]
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
        return []

# ==========================================
# 2. H5 写入函数 (保持原逻辑，确保高性能)
# ==========================================
def save_chunk_to_h5(data_list, save_path):
    n_graphs = len(data_list)
    if n_graphs == 0: return [], 0

    n_atoms_list = [d.num_nodes for d in data_list]
    n_edges_list = [d.num_edges for d in data_list]
    atom_ptr = np.cumsum([0] + n_atoms_list, dtype=np.int64)
    edge_ptr = np.cumsum([0] + n_edges_list, dtype=np.int64)
    
    with h5py.File(save_path, 'w') as f:
        comp = 'lzf' 
        f.create_dataset('atom_ptr', data=atom_ptr)
        f.create_dataset('edge_ptr', data=edge_ptr)
        f.create_dataset('z', data=np.concatenate([d.z.numpy() for d in data_list]), compression=comp)
        f.create_dataset('pos', data=np.concatenate([d.pos.numpy() for d in data_list]), compression=comp)
        f.create_dataset('force', data=np.concatenate([d.force.numpy() for d in data_list]), compression=comp)
        f.create_dataset('edge_index', data=np.concatenate([d.edge_index.numpy() for d in data_list], axis=1), compression=comp)
        f.create_dataset('shifts_int', data=np.concatenate([d.shifts_int.numpy() for d in data_list]), compression=comp)
        f.create_dataset('y', data=np.stack([d.y.numpy() for d in data_list]))
        f.create_dataset('cell', data=np.stack([d.cell.numpy() for d in data_list]))
        f.create_dataset('stress', data=np.stack([d.stress.numpy() for d in data_list]))
        f.attrs['has_stress'] = True
            
    return [(n_atoms_list[i], n_edges_list[i]) for i in range(n_graphs)], n_graphs

# ==========================================
# 3. Worker & Manager (适配新路径逻辑)
# ==========================================
def worker_task(args):
    (worker_id, file_paths, save_dir, prefix, cutoff, chunk_size) = args
    buffer, save_counter, generated_info = [], 0, []
    for fpath in file_paths:
        data_list = extxyz_to_pyg_custom(fpath, cutoff=cutoff)
        for data in data_list:
            buffer.append(data)
            if len(buffer) >= chunk_size:
                fname = f"{prefix}_w{worker_id}_p{save_counter}.h5"
                stats, count = save_chunk_to_h5(buffer, os.path.join(save_dir, fname))
                generated_info.append({'file_name': fname, 'count': count, 'stats': stats})
                buffer, save_counter = [], save_counter + 1
                gc.collect()
    if buffer:
        fname = f"{prefix}_w{worker_id}_p{save_counter}.h5"
        stats, count = save_chunk_to_h5(buffer, os.path.join(save_dir, fname))
        generated_info.append({'file_name': fname, 'count': count, 'stats': stats})
    return generated_info

def process_manager(file_list, save_dir, prefix, num_workers, chunk_size, cutoff):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    real_workers = min(num_workers, len(file_list))
    tasks = [(i, files.tolist(), save_dir, prefix, cutoff, chunk_size) 
             for i, files in enumerate(np.array_split(file_list, real_workers))]

    all_chunks_info = []
    with multiprocessing.Pool(processes=real_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers, desc=prefix):
            all_chunks_info.extend(result)

    all_chunks_info.sort(key=lambda x: x['file_name'])
    metadata = []
    for info in all_chunks_info:
        for i, (n_a, n_e) in enumerate(info['stats']):
            metadata.append({'file_path': info['file_name'], 'index_in_file': i, 'num_atoms': n_a, 'num_edges': n_e})
    torch.save(metadata, os.path.join(save_dir, f"{prefix}_metadata.pt"))

# ==========================================
# 4. 主入口
# ==========================================
if __name__ == '__main__':
    DATA_ROOT = "../mptrj_xyz"
    SAVE_DIR = "../dataset_h5"
    CONFIG = {
        "num_workers": 8,    
        "chunk_size": 5000,   
        "cutoff": 6.0         
    }

    for split in ["test", "val", "train"]: # 建议从小的 test 开始运行验证
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path): continue
        
        files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.xyz')]
        random.shuffle(files)
        
        print(f"\n--- Processing {split}: {len(files)} files ---")
        process_manager(files, SAVE_DIR, split, **CONFIG)