import torch
import numpy as np
import json
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

# --- 导入自定义模块 ---
from src.data import ChunkedSmartDataset   # 👈 新的数据集
from src.data import BinPackingSampler # 👈 新的采样器
from src.models import HTGPModel
from src.utils import HTGPConfig
from src.engine import PotentialTrainer 

torch.multiprocessing.set_sharing_strategy('file_system')

# ✅ 改为 Float32
torch.set_default_dtype(torch.float32)

# ✅ 🚀 开启 TF32 (你的 H100/A100 显卡神器，精度几乎不掉，速度快很多)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
# ==========================================
# 1. DDP 初始化
# ==========================================
def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        if rank == 0: print(f"🚀 [Init] DDP Enabled: Rank {rank}/{world_size}")
        return local_rank, rank, world_size
    else:
        print("⚠️ Single GPU Mode")
        return 0, 0, 1

LOCAL_RANK, RANK, WORLD_SIZE = init_distributed_mode()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. 参数配置
# ==========================================
DATA_DIR = "/var/tmp/lmy_test/"  # 👈 确保这里有 train_metadata.pt
LOG_DIR = "Checkpoints"
meta_e0_path = "/var/tmp/lmy_test/meta_e0_data.pt"
# 🔥 这里的 BATCH_SIZE 变s成了 "每个 Batch 的最大原子数"
MAX_COST_PER_BATCH = 8100  # H100
NUM_WORKERS = 6            # 根据你的 CPU 核数调整
LR = 1e-3
EPOCHS = 15

if RANK == 0: os.makedirs(LOG_DIR, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = "1"   
# ==========================================
# 3. 准备数据 (Loader)
# ==========================================
if RANK == 0: print("\n[1/5] Initializing Smart DataLoaders...")

# --- A. 训练集 (使用装箱采样) ---
try:
    train_dataset = ChunkedSmartDataset(
        DATA_DIR, 
        metadata_file="train_metadata.pt", 
        cache_size=2, # 缓存 16 个文件块
        rank=RANK,
        world_size=WORLD_SIZE
    )
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 没找到 train_metadata.pt，请先运行 preprocess.py！")

train_sampler = BinPackingSampler(
    train_dataset.metadata,
    max_cost=MAX_COST_PER_BATCH,
    edge_weight="auto", # 边权重
    shuffle=True,
    world_size=WORLD_SIZE,
    rank=RANK
)

# print("test 打印前10个样本的索引和成本", train_sampler.indices_with_cost[:10])  # 打印前10个样本的索引和成本，检查采样器是否正常工作

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler, # 👈 关键：使用 batch_sampler
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=4,
)

# --- B. 测试集 (简单加载即可) ---
# 测试集也可以用装箱来加速推理，但不用 shuffle
try:
    test_dataset = ChunkedSmartDataset(
        DATA_DIR, 
        metadata_file="test_metadata.pt",
        cache_size=2,
        rank=RANK,
        world_size=WORLD_SIZE
    )
    test_sampler = BinPackingSampler(
        test_dataset.metadata,
        max_cost=MAX_COST_PER_BATCH, # 推理时不存梯度，这个值可以设得比训练大一倍
        edge_weight="auto",
        shuffle=False,
        world_size=WORLD_SIZE,
        rank=RANK
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
except FileNotFoundError:
    if RANK == 0: print("⚠️ test_metadata.pt not found, skipping validation setup.")
    test_loader = None

# ==========================================
# 4. 模型构建与编译
# ==========================================
if RANK == 0: print("\n[2/5] Building & Compiling Model...")

config = HTGPConfig(
    num_atom_types=55, 
    hidden_dim=64, 
    num_layers=3, 
    cutoff=6.0, 
    num_rbf=10,
    use_L0=True, 
    use_L1=True,
    use_L2=True, 
    use_gating=True, 
    use_long_range=False
)

model = HTGPModel(config).to(DEVICE)
print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())}")

if RANK == 0: print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())}")

# --- 注入 E0 (从 metadata.pt 加载) ---
if RANK == 0: print("\n[3/5] Loading Atomic References (E0)...")
if os.path.exists(meta_e0_path):
    meta_data = torch.load(meta_e0_path, map_location='cpu', weights_only=False)
    e0_dict = meta_data.get('e0_dict', None)
    model.load_external_e0(e0_dict)
    count = len(e0_dict) if e0_dict else 0
    model.atomic_ref.weight.requires_grad = False
    if RANK == 0:
        print(f"🔒 Injected E0 for {count} elements (Float32).")
else:
    model.atomic_ref.weight = model.atomic_ref.weight.float()
    if RANK == 0: print("⚠️ meta_e0_data.pt not found, skipping E0 injection.")
    
# DDP Wrap
if dist.is_initialized():
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

# # ==========================================
# # 5. Trainer 初始化
# # ==========================================
# # 由于 Sampler 是动态的，我们需要估算 Steps
# # 可以先跑一遍 len(train_sampler) 或者用估算值
exact_total_steps = train_sampler.precompute_total_steps(EPOCHS)

if RANK == 0: print(f"📊 Estimated total steps: {exact_total_steps}")

trainer = PotentialTrainer(
    model, 
    total_steps=exact_total_steps,
    # epochs=EPOCHS,
    max_lr=LR, 
    device=DEVICE, 
    checkpoint_dir=LOG_DIR
)

# ==========================================
# 6. 训练循环
# ==========================================
if RANK == 0: 
    print("\n[4/5] Starting Loop...")
    print("="*80)

for epoch in range(1, EPOCHS + 1):
    # Map-style Dataset 配合 Sampler 不需要手动 set_epoch
    train_sampler.set_epoch(epoch)
    
    # 训练
    train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch)
    
    # 验证
    if test_loader:
        val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
    else:
        val_metrics = {'total_loss': 0.0, 'mae_e': 0.0, 'mae_f': 0.0, 'mae_s_gpa': 0.0}
    
    if RANK == 0:
        print(f"Ep {epoch} | T_Loss: {train_metrics['total_loss']:.4f} V_Loss: {val_metrics['total_loss']:.4f} | "
              f"MAE_F: {train_metrics['mae_f']*1000:.1f}/{val_metrics['mae_f']*1000:.1f} meV/A")
        
        # 保存逻辑
        trainer.save(f'model_epoch_{epoch}.pt')

    if dist.is_initialized(): # 所有rank在这里等 Rank 0 写完文件
        dist.barrier()

if dist.is_initialized(): dist.destroy_process_group()