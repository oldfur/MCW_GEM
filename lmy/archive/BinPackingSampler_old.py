import torch
import random
from torch.utils.data import Sampler

class BinPackingSampler(Sampler):
    def __init__(self, metadata, max_cost=3000, edge_weight='auto', shuffle=True, 
                 world_size=1, rank=0, seed=42): # 🔥 新增 seed 参数
        """
        :param seed: 基础随机种子，保证 DDP 各卡初始状态一致
        """
        self.metadata = metadata
        self.max_cost = max_cost
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.seed = seed      # 🔥 保存种子
        self.epoch = 0        # 🔥 新增 epoch 计数器
        
        # ---------------------------------------------------
        # 1. 计算权重 (逻辑保持不变)
        # ---------------------------------------------------
        if edge_weight == 'auto':
            total_atoms = 0
            total_edges = 0
            for item in metadata:
                total_atoms += item['num_atoms']
                total_edges += item['num_edges']
            
            if total_edges > 0:
                self.edge_weight = total_atoms / total_edges
            else:
                self.edge_weight = 0.0
            
            # 仅在主进程打印
            if self.rank == 0:
                print(f"⚖️ [Auto-Balance] Total Atoms: {total_atoms}, Total Edges: {total_edges}")
                print(f"⚖️ [Auto-Balance] Calculated Edge Weight: {self.edge_weight:.6f}")
                print(f"   (这意味着每 {1/self.edge_weight:.1f} 条边 ≈ 1 个原子的显存消耗)")
        else:
            self.edge_weight = float(edge_weight)

        # ---------------------------------------------------
        # 2. 预计算所有 Cost
        # ---------------------------------------------------
        self.indices_with_cost = []
        for i, item in enumerate(metadata):
            # Cost = Atoms + 权重 * Edges
            c = item['num_atoms'] + self.edge_weight * item['num_edges']
            self.indices_with_cost.append((i, c))

    def set_epoch(self, epoch):
        """
        🔥 关键方法：在每个 Epoch 开始前调用，
        确保每一轮的随机扰动不同，但在所有 GPU 上是一致的。
        """
        self.epoch = epoch

    def __iter__(self):
        # ---------------------------------------------------
        # 1. 确定性随机发生器 (Deterministic RNG)
        # ---------------------------------------------------
        # 使用 random.Random 创建局部随机实例，不影响全局 random
        # 种子 = 基础种子 + 当前 Epoch，保证 DDP 同步
        rng = random.Random(self.seed + self.epoch)

        # ---------------------------------------------------
        # 2. 排序 (Sort) - 带随机扰动
        # ---------------------------------------------------
        if self.shuffle:
            # 使用 rng.uniform 而不是 random.uniform
            self.indices_with_cost.sort(
                key=lambda x: x[1] * rng.uniform(0.99, 1.01), 
                reverse=True
            )
        else:
            self.indices_with_cost.sort(key=lambda x: x[1], reverse=True)

        # ---------------------------------------------------
        # 3. 装箱 (Bin Packing)
        # ---------------------------------------------------
        batches = []
        current_batch = []
        current_batch_cost = 0
        
        for idx, cost in self.indices_with_cost:
            # 检查 Cost 是否溢出
            if current_batch_cost + cost > self.max_cost and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_cost = 0
            
            current_batch.append(idx)
            current_batch_cost += cost
        
        if current_batch:
            batches.append(current_batch)

        # ---------------------------------------------------
        # 4. Batch 间打乱 (使用 rng)
        # ---------------------------------------------------
        if self.shuffle:
            rng.shuffle(batches) # 🔥 保证所有 Rank 的 Batch 顺序打乱得一模一样

        # ---------------------------------------------------
        # 5. DDP 分发 (切片)
        # ---------------------------------------------------
        total_batches = len(batches)
        
        # Drop Last (保证整除)
        num_samples_per_rank = total_batches // self.world_size
        batches = batches[:num_samples_per_rank * self.world_size]
        
        # 间隔采样: Rank 0 拿 [0, 8, 16...], Rank 1 拿 [1, 9, 17...]
        my_batches = batches[self.rank::self.world_size]
        
        for batch_indices in my_batches:
            yield batch_indices

    def __len__(self):
        total_cost = sum(x[1] for x in self.indices_with_cost)
        estimated_batches = total_cost / (self.max_cost) 
        return int(estimated_batches // self.world_size)