import os
import glob
import random
import torch
import math
import torch.distributed as dist
from torch.utils.data import IterableDataset

class ShardedPyGDataset(IterableDataset):
    def __init__(self, data_dir, prefix, shuffle=False):
        """
        :param data_dir: 数据保存的文件夹路径
        :param prefix: 文件前缀 (e.g., "train" 或 "test")
        :param shuffle: 是否打乱数据 (训练集 True, 测试集 False)
        """
        super().__init__()
        self.data_dir = data_dir
        pattern = os.path.join(data_dir, f"{prefix}_*.pt")
        self.file_paths = sorted(glob.glob(pattern))
        self.shuffle = shuffle
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 找到以 {prefix} 开头的文件！")
        
        # 只在主进程打印
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"📂 [Dataset] Found {len(self.file_paths)} parts for '{prefix}'")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths.copy()

        # ============================================================
        # 1. DDP 切分: 按 GPU Rank 分配文件
        # ============================================================
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # 间隔采样: GPU0->[0,4,8], GPU1->[1,5,9]...
            files = files[rank::world_size]

        # ============================================================
        # 2. Worker 切分: 按 CPU 进程分配文件
        # ============================================================
        if worker_info is not None:
            per_worker = int(math.ceil(len(files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files))
            files = files[iter_start:iter_end]

        if self.shuffle:
            random.shuffle(files)

        for file_path in files:
            try:
                # 读取数据 (可能是 int8/int32/float32 混合)
                chunk_data = torch.load(file_path, weights_only=False)
                
                if self.shuffle:
                    random.shuffle(chunk_data)
                
                for data in chunk_data:
                    # ========================================================
                    # 🔥🔥🔥 核心：强制类型修正 (Float32 / Int64) 🔥🔥🔥
                    # ========================================================
                    
                    # --- A. 索引类必须是 Int64 (Long) ---
                    if hasattr(data, 'edge_index') and data.edge_index is not None:
                        data.edge_index = data.edge_index.to(torch.long)
                    
                    if hasattr(data, 'z') and data.z is not None:
                        data.z = data.z.to(torch.long)
                        
                    if hasattr(data, 'edge_type') and data.edge_type is not None:
                        data.edge_type = data.edge_type.to(torch.long)

                    # --- B. 数值类必须是 Float32 ---
                    # 1. 坐标
                    data.pos = data.pos.to(torch.float32)
                    
                    # 2. 晶胞
                    if hasattr(data, 'cell') and data.cell is not None:
                        data.cell = data.cell.to(torch.float32)
                    
                    # 3. 周期性位移 (处理 int8 压缩)
                    if hasattr(data, 'shifts_int'):
                        data.shifts = data.shifts_int.to(torch.float32)
                        del data.shifts_int # 删除旧属性
                    elif hasattr(data, 'shifts') and data.shifts is not None:
                        data.shifts = data.shifts.to(torch.float32)

                    # 4. 标签 (能量/力/应力)
                    if hasattr(data, 'y') and data.y is not None:
                        data.y = data.y.to(torch.float32)
                        
                    if hasattr(data, 'force') and data.force is not None:
                        data.force = data.force.to(torch.float32)
                        
                    if hasattr(data, 'stress') and data.stress is not None:
                        data.stress = data.stress.to(torch.float32)

                    yield data
                    
            except Exception as e:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"⚠️ Error loading {file_path}: {e}")
                continue
