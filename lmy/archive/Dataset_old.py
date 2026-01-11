import os
import glob
import random
import torch
from torch.utils.data import IterableDataset
import math
class ShardedPyGDataset(IterableDataset):
    def __init__(self, data_dir, prefix, shuffle=False):
        """
        :param data_dir: 数据保存的文件夹路径 (e.g., "processed_dataset_mp")
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
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # 复制一份文件列表，以免影响其他地方
        files = self.file_paths.copy()

        # --- A. 多进程 DataLoader 分配逻辑 ---
        # 如果 DataLoader 开了 num_workers > 0，必须给每个 Worker 分配不同的文件，
        # 否则所有 Worker 会读同样的数据，导致训练重复！
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
                # 1. 加载一个小文件 (包含 500 个 Data 对象)
                chunk_data = torch.load(file_path)
                
                # 2. (可选) 打乱这 500 个数据的内部顺序
                if self.shuffle:
                    random.shuffle(chunk_data)
                
                # 3. 逐个“吐”出数据给 DataLoader
                for data in chunk_data:
                    yield data
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue