import torch
import numpy as np
import json
import os
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

# --- 导入自定义模块 ---
from Dataset_dist import ShardedPyGDataset
from Model import HTGPModel
from Utils import HTGPConfig
# 🔥 确保这里引用的是上面新写的 Trainer 文件名
from Potentialtrainer_dist_new import PotentialTrainer 

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
        print(f"🚀 [Init] Rank {rank}/{world_size} on GPU {local_rank} initialized.")
        return local_rank, rank, world_size
    else:
        print("⚠️ Not using DDP, falling back to single GPU.")
        return 0, 0, 1

LOCAL_RANK, RANK, WORLD_SIZE = init_distributed_mode()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

# 🔥 全局强制 Float32
torch.set_default_dtype(torch.float32)

# ==========================================
# 2. 参数配置
# ==========================================
DATA_DIR = "/var/lib/kubelet/MUYU_data"
LOG_DIR = "Checkpoints"
BATCH_SIZE = 12  # 单卡 BatchSize
NUM_WORKERS = 2
LR = 1e-3
EPOCHS = 15     # OneCycleLR 的重要参数

if RANK == 0:
    os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 3. 计算步数 (Steps) - 🔥 OneCycleLR 必需
# ==========================================
if RANK == 0: print("\n[1/5] Calculating dataset size...")

meta_path = os.path.join(DATA_DIR, "meta_data.pt")
CHUNK_SIZE = 5120
CHUNK_SIZE_TEST = 5120

if os.path.exists(meta_path):
    try:
        meta_data = torch.load(meta_path, map_location='cpu', weights_only=False)
        if 'config' in meta_data and 'chunk_size' in meta_data['config']:
            CHUNK_SIZE = meta_data['config']['chunk_size']
            if RANK == 0: print(f"ℹ️  Chunk Size identified: {CHUNK_SIZE}")
    except Exception as e:
        if RANK == 0: print(f"⚠️  Error reading chunk size: {e}")

train_files_count = len(glob.glob(os.path.join(DATA_DIR, "train_*.pt")))
test_files_count = len(glob.glob(os.path.join(DATA_DIR, "test_*.pt")))

if train_files_count == 0:
    raise FileNotFoundError(f"❌ 在 {DATA_DIR} 未找到 train_*.pt 文件！")

# 🔥 DDP 步数修正: 总数 / (Batch * 卡数)
TRAIN_STEPS = (train_files_count * CHUNK_SIZE) // (BATCH_SIZE * WORLD_SIZE)
TEST_STEPS = (test_files_count * CHUNK_SIZE_TEST) // (BATCH_SIZE * WORLD_SIZE)
if TEST_STEPS == 0 and test_files_count > 0: TEST_STEPS = 1

if RANK == 0:
    print(f"📊 Global Files: {train_files_count}")
    print(f"📊 Per-GPU Steps: {TRAIN_STEPS} (World Size: {WORLD_SIZE})")

# ==========================================
# 4. Loader
# ==========================================
if RANK == 0: print("\n[2/5] Initializing DataLoaders...")

train_dataset = ShardedPyGDataset(DATA_DIR, prefix="train", shuffle=True)
test_dataset = ShardedPyGDataset(DATA_DIR, prefix="test", shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # DDP IterableDataset 手动 shuffle
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ==========================================
# 5. 模型构建
# ==========================================
if RANK == 0: print("\n[3/5] Building Model...")

config = HTGPConfig(
    num_atom_types=55, hidden_dim=64, num_layers=3, cutoff=6.0,
    use_L2=True, use_gating=True, use_long_range=False
)

model = HTGPModel(config).float().to(DEVICE)
if RANK == 0: print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())}")

# --- 注入 E0 ---
if RANK == 0: print("\n[4/5] Loading Atomic References (E0)...")
if os.path.exists(meta_path):
    meta_data = torch.load(meta_path, map_location='cpu', weights_only=False)
    e0_dict = meta_data.get('e0_dict', None)
    
    if e0_dict:
        if RANK == 0: print(f"✅ E0 Dict Loaded.")
        with torch.no_grad():
            count = 0
            for z, e in e0_dict.items():
                z_idx = int(z)
                if z_idx < model.atomic_ref.weight.size(0):
                    val = torch.tensor(e, dtype=torch.float32, device=DEVICE)
                    model.atomic_ref.weight[z_idx] = val
                    count += 1
        model.atomic_ref.weight.requires_grad = False
        if RANK == 0: print(f"🔒 Injected E0 for {count} elements (Float32).")
    else:
        raise ValueError("❌ meta_data.pt found but 'e0_dict' is missing!")
else:
    raise FileNotFoundError("❌ meta_data.pt not found!")

# DDP Wrap
if dist.is_initialized():
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

# ==========================================
# 6. Trainer & Loop
# ==========================================
# 传入 steps_per_epoch 和 epochs 给 OneCycleLR
trainer = PotentialTrainer(
    model, 
    steps_per_epoch=TRAIN_STEPS,  # 👈 关键：用于 Scheduler
    epochs=EPOCHS,                # 👈 关键：用于 Scheduler
    lr=LR, 
    device=DEVICE, 
    checkpoint_dir=LOG_DIR
)

# 🚀 记录详细历史 (包含 MAE)
history = {
    'epoch': [],
    'train_loss': [], 'val_loss': [],
    'train_mae_e': [], 'val_mae_e': [],
    'train_mae_f': [], 'val_mae_f': [],
    'train_mae_s': [], 'val_mae_s': []
}

def save_history():
    if RANK == 0:
        with open(f"{LOG_DIR}/history.json", 'w') as f:
            json.dump(history, f, indent=4)

if RANK == 0:
    print("\n" + "="*105)
    # 打印宽表头
    print(f"{'Ep':^4} | {'Loss(T)':^9} {'Loss(V)':^9} | {'MAE_E(T)':^9} {'MAE_E(V)':^9} | {'MAE_F(T)':^9} {'MAE_F(V)':^9} | {'MAE_S(T)':^9} {'MAE_S(V)':^9}")
    print("="*105)
    print("\n[5/5] Starting Training Loop...")

best_val_mae_f = float('inf') # 可以根据 Force MAE 来选模型，也可以用 Loss

for epoch in range(1, EPOCHS + 1):
    # 运行训练
    train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch)
    
    # 运行验证
    val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
    
    # 主进程记录数据
    if RANK == 0:
        history['epoch'].append(epoch)
        
        # Loss
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        
        # Energy MAE
        history['train_mae_e'].append(train_metrics['mae_e'])
        history['val_mae_e'].append(val_metrics['mae_e'])

        # Force MAE
        history['train_mae_f'].append(train_metrics['mae_f'])
        history['val_mae_f'].append(val_metrics['mae_f'])
        
        # Stress MAE
        history['train_mae_s'].append(train_metrics['mae_s_gpa'])
        history['val_mae_s'].append(val_metrics['mae_s_gpa'])

        save_history()

        # 打印 (E转meV, F转meV/A，S用GPa)
        print(f"{epoch:^4} | "
              f"{train_metrics['total_loss']:^9.4f} {val_metrics['total_loss']:^9.4f} | "
              f"{train_metrics['mae_e']*1000:^9.2f} {val_metrics['mae_e']*1000:^9.2f} | "
              f"{train_metrics['mae_f']*1000:^9.2f} {val_metrics['mae_f']*1000:^9.2f} | "
              f"{train_metrics['mae_s_gpa']:^9.4f} {val_metrics['mae_s_gpa']:^9.4f}")

        # 保存策略：这里我以 Force MAE 最小为准，你也可以换成 total_loss
        current_metric = val_metrics['mae_f']
        if current_metric < best_val_mae_f:
            best_val_mae_f = current_metric
            trainer.save('best_model.pt')
        
        # 定期保存
        if epoch % 5 == 0:
            trainer.save(f'checkpoint_ep{epoch}.pt')

    if dist.is_initialized():
        dist.barrier()

if RANK == 0: print("\n🎉 Training Finished!")
if dist.is_initialized(): dist.destroy_process_group()
