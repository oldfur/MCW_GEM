import os
import glob
import random
import torch
import math
from torch.utils.data import IterableDataset

class ShardedPyGDataset(IterableDataset):
    def __init__(self, data_dir, prefix, shuffle=False):
        """
        :param data_dir: 数据保存的文件夹路径 (e.g., "processed_dataset")
        :param prefix: 文件前缀 (e.g., "train" 或 "test")
        :param shuffle: 是否打乱数据 (训练集 True, 测试集 False)
        """
        super().__init__()
        self.data_dir = data_dir
        # 1. 找到所有匹配的文件 (e.g., train_w0_part_0.pt, train_w1_part_5.pt ...)
        pattern = os.path.join(data_dir, f"{prefix}_*.pt")
        self.file_paths = sorted(glob.glob(pattern))
        self.shuffle = shuffle
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 找到以 {prefix} 开头的文件！")
        
        print(f"📂 [Dataset] Found {len(self.file_paths)} parts for '{prefix}'")

    def __iter__(self):
        """
        核心流式逻辑：
        每次迭代时，Worker 都会独立执行这个函数。
        在这里进行【数据类型还原】，将压缩存储的数据转回模型需要的格式。
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # 复制一份文件列表，以免影响其他地方
        files = self.file_paths.copy()

        # --- A. 多进程 DataLoader 分配逻辑 ---
        if worker_info is not None:
            # 将文件列表尽可能均匀地分给各个 Worker
            per_worker = int(math.ceil(len(files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files))
            
            # 当前 Worker 只负责处理这一部分文件
            files = files[iter_start:iter_end]

        # --- B. 打乱文件顺序 (实现全局 Shuffle 的效果) ---
        if self.shuffle:
            random.shuffle(files)

        # --- C. 逐个文件读取并 Yield 数据 ---
        for file_path in files:
            try:
                # 1. 加载一个小文件
                chunk_data = torch.load(file_path, weights_only = False)
                
                # 2. (可选) 打乱这批数据的内部顺序
                if self.shuffle:
                    random.shuffle(chunk_data)
                
                # 3. 逐个处理并“吐”出数据
                for data in chunk_data:
                    # ========================================================
                    # 🔥🔥🔥 核心修改：数据类型还原 (De-compression) 🔥🔥🔥
                    # ========================================================
                    
                    # 1. 邻居索引 (Edge Index): int32 -> int64
                    # PyTorch Geometric 的 Message Passing 必须用 int64 (LongTensor)
                    if hasattr(data, 'edge_index') and data.edge_index is not None:
                        if data.edge_index.dtype == torch.int32:
                            data.edge_index = data.edge_index.to(torch.long)

                    # 2. 原子序数 (Z): int8 -> int64
                    # Embedding 层查表必须用 int64
                    if hasattr(data, 'z') and data.z is not None:
                        if data.z.dtype == torch.int8:
                            data.z = data.z.to(torch.long)

                    # 3. 周期性位移 (Shifts): int8 -> float32
                    # 之前为了省空间存成了 int8 (名为 shifts_int)，现在要转回 float32 
                    # 并重命名为 shifts，以便和 cell (float) 进行矩阵乘法
                    if hasattr(data, 'shifts_int'):
                        data.shifts = data.shifts_int.to(torch.float32)
                        # 删除旧属性以节省内存
                        del data.shifts_int
                    elif hasattr(data, 'shifts') and data.shifts.dtype == torch.int8:
                        # 如果名字没改，直接转类型
                        data.shifts = data.shifts.to(torch.float32)

                    # 4. 边类型 (Edge Type): int8 -> int64
                    if hasattr(data, 'edge_type') and data.edge_type is not None:
                        if data.edge_type.dtype == torch.int8:
                            data.edge_type = data.edge_type.to(torch.long)

                    # 5. 确保坐标和力是 float32 (防止意外存成 double)
                    if data.pos.dtype == torch.float64:
                        data.pos = data.pos.to(torch.float32)
                    if hasattr(data, 'force') and data.force is not None:
                        if data.force.dtype == torch.float64:
                            data.force = data.force.to(torch.float32)
                    if hasattr(data, 'cell') and data.cell is not None:
                        if data.cell.dtype == torch.float64:
                            data.cell = data.cell.to(torch.float32)

                    # yield 出去的是完美的、符合模型要求的 Data 对象
                    yield data
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
