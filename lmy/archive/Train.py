import torch
import numpy as np
import json
import os
import glob
from torch_geometric.loader import DataLoader

# --- 导入你的自定义模块 ---
# 请确保这些文件都在同一目录下，或者在 PYTHONPATH 中
from Dataset import ShardedPyGDataset
from Model import HTGPModel
from Utils import HTGPConfig
from Potentialtrainer import PotentialTrainer # 注意文件名大小写匹配

# ==========================================
# 1. 配置参数与设备
# ==========================================
DATA_DIR = "/var/lib/kubelet/MUYU_data"  # 数据集路径
LOG_DIR = "Checkpoints"                  # 日志保存路径
BATCH_SIZE = 16
NUM_WORKERS = 4                          # 读取进程数 (建议 4-8)
LR = 1e-3
EPOCHS = 100

os.makedirs(LOG_DIR, exist_ok=True)

# 检查显卡

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
print(f"🚀 使用设备: {DEVICE}")

# ==========================================
# 2. 自动计算进度条长度 (关键步骤)
# ==========================================
print("\n[1/5] Calculating dataset size...")

meta_path = os.path.join(DATA_DIR, "meta_data.pt")
CHUNK_SIZE = 5120 # 默认值，防止读取失败
if os.path.exists(meta_path):
    try:
        meta_data = torch.load(meta_path, weights_only=False)
        if 'config' in meta_data and 'chunk_size' in meta_data['config']:
            CHUNK_SIZE = meta_data['config']['chunk_size']
            print(f"ℹ️  Chunk Size identified: {CHUNK_SIZE}")
    except Exception as e:
        print(f"⚠️  Error reading chunk size: {e}")

# 统计文件数量
train_files = glob.glob(os.path.join(DATA_DIR, "train_*.pt"))
test_files = glob.glob(os.path.join(DATA_DIR, "test_*.pt"))
train_files_count = len(train_files)
test_files_count = len(test_files)

if train_files_count == 0:
    raise FileNotFoundError(f"❌ 在 {DATA_DIR} 未找到 train_*.pt 文件！")

# 计算总步数 (用于 tqdm 进度条)
# 公式: 总数据量 / BatchSize
# 总数据量 = 文件数 * 每个文件的容量(ChunkSize)
TRAIN_STEPS = (train_files_count * CHUNK_SIZE) // BATCH_SIZE
CHUNK_SIZE_TEST = 5120  # 默认值
TEST_STEPS = (test_files_count * CHUNK_SIZE_TEST) // BATCH_SIZE

# 防止 Test 集太小导致 step 为 0
if TEST_STEPS == 0 and test_files_count > 0: TEST_STEPS = 1

print(f"📊 训练集: {train_files_count} files | Est. Steps: {TRAIN_STEPS}")
print(f"📊 测试集: {test_files_count} files | Est. Steps: {TEST_STEPS}")

# ==========================================
# 3. 实例化 Dataset 和 DataLoader
# ==========================================
print("\n[2/5] Initializing DataLoaders...")

train_dataset = ShardedPyGDataset(DATA_DIR, prefix="train", shuffle=True)
test_dataset = ShardedPyGDataset(DATA_DIR, prefix="test", shuffle=False)

# 注意：流式数据集必须 shuffle=False
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # ❌ 必须为 False
    num_workers=NUM_WORKERS,
    pin_memory=True,        # ✅ 加速
    prefetch_factor=2,      # ✅ 预取
    persistent_workers=True # ✅ 保持 Worker 活跃
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ==========================================
# 4. 模型初始化
# ==========================================
print("\n[3/5] Building Model...")

config = HTGPConfig(
    num_atom_types=60,      # 根据数据集调整
    hidden_dim=128,
    num_layers=2,
    cutoff=6.0,
    use_L2=True,
    use_gating=True,
    use_long_range=False
)

model = HTGPModel(config).to(DEVICE)
print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())}")

# ==========================================
# 5. 注入 E0 (原子平均能量)
# ==========================================
print("\n[4/5] Loading Atomic References (E0)...")

if os.path.exists(meta_path):
    meta_data = torch.load(meta_path, weights_only=False)
    # 获取 e0_dict
    e0_dict = meta_data.get('e0_dict', None)
    
    if e0_dict:
        print(f"✅ E0 Dict Loaded.")
        with torch.no_grad():
            count = 0
            for z, e in e0_dict.items():
                z_idx = int(z)
                if z_idx < model.atomic_ref.weight.size(0):
                    model.atomic_ref.weight[z_idx] = torch.tensor(e, dtype=model.atomic_ref.weight.dtype)
                    count += 1
            print(f"🔒 Injected and froze E0 for {count} elements.")
        
        # 冻结参数
        model.atomic_ref.weight.requires_grad = False
    else:
        raise ValueError("❌ meta_data.pt found but 'e0_dict' is missing!")
else:
    raise FileNotFoundError("❌ meta_data.pt not found! Please run 'calc_e0.py' first.")

# ==========================================
# 6. 训练准备
# ==========================================
trainer = PotentialTrainer(model, lr=LR, device=DEVICE, checkpoint_dir=LOG_DIR)

# 历史记录
history = {
    'epoch': [],
    'train_loss': [], 'val_loss': [],
    'train_mae_e': [], 'val_mae_e': [],
    'train_mae_f': [], 'val_mae_f': [],
    'train_mae_s': [], 'val_mae_s': []
}

def save_history():
    with open(f"{LOG_DIR}/history.json", 'w') as f:
        json.dump(history, f, indent=4)

print("\n" + "="*105)
print(f"{'Epoch':^6} | {'TrainLoss':^10} | {'ValLoss':^10} | "
      f"{'Tr E (meV)':^10} | {'Tr F (meV)':^10} | {'Tr S (GPa)':^10} | "
      f"{'Val E':^10} | {'Val F':^10} | {'Val S':^10}")
print("="*105)

# ==========================================
# 7. 🔥 训练循环
# ==========================================
print("\n[5/5] Starting Training Loop...")

# Optional: Run a baseline validation first (Epoch 0)
# print("Running baseline validation...")
# base_metrics = trainer.validate(test_loader, total_steps=TEST_STEPS)
# print(f"Baseline Val Loss: {base_metrics['total']:.4f}")

best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # 1. 训练 (传入 total_steps 显示进度条)
    train_metrics = trainer.train_epoch(train_loader, total_steps=TRAIN_STEPS)
    
    # 2. 验证
    val_metrics = trainer.validate(test_loader, total_steps=TEST_STEPS)
    
    # 3. 记录日志
    history['epoch'].append(epoch)
    history['train_loss'].append(train_metrics['total'])
    history['val_loss'].append(val_metrics['total'])
    
    # 单位转换: eV -> meV
    history['train_mae_e'].append(train_metrics['mae_e'] * 1000)
    history['val_mae_e'].append(val_metrics['mae_e'] * 1000)
    history['train_mae_f'].append(train_metrics['mae_f'] * 1000)
    history['val_mae_f'].append(val_metrics['mae_f'] * 1000)
    
    history['train_mae_s'].append(train_metrics['mae_s'])
    history['val_mae_s'].append(val_metrics['mae_s'])
    
    save_history()

    # 4. 打印一行日志
    log_str = (
        f"{epoch:^6} | "
        f"{train_metrics['total']:^10.4f} | "
        f"{val_metrics['total']:^10.4f} | "
        f"{train_metrics['mae_e']*1000:^10.2f} | "
        f"{train_metrics['mae_f']*1000:^10.2f} | "
        f"{train_metrics['mae_s']:^10.3f} | "
        f"{val_metrics['mae_e']*1000:^10.2f} | "
        f"{val_metrics['mae_f']*1000:^10.2f} | "
        f"{val_metrics['mae_s']:^10.3f}"
    )
    print(log_str)

    # 5. 保存最佳模型
    if val_metrics['total'] < best_val_loss:
        best_val_loss = val_metrics['total']
        trainer.save('best_model.pt')

print("\n🎉 Training Finished!")
