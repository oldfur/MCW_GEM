import os
import json
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

# --- 导入自定义模块 ---
from src.data import ChunkedSmartDataset_h5, BinPackingSampler
from src.models import HTGPModel
from src.utils import HTGPConfig
from src.engine import PotentialTrainer 

# ==========================================
# 0. 全局环境设置
# ==========================================
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 训练配置
# ==========================================
class Config:
    DATA_DIR = "/dev/shm/dataset_h5_r6_inorg"
    TRAIN_META = "train_metadata.pt"
    TEST_META = "test_metadata.pt"
    E0_PATH = "/dev/shm/dataset_h5_r6_inorg/meta_data.pt"
    LOG_DIR = "Checkpoints"

    MAX_COST_PER_BATCH = 10000 # 4000 cutoff 为7
    LR = 1e-3
    EPOCHS = 100
    
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 2

    # 基础模型参数 (用于新建模型时)
    MODEL_PARAMS = dict(
        num_atom_types=100, 
        hidden_dim=96, 
        num_layers=2, 
        cutoff=6.0, 
        num_rbf=10,
        use_L0=True, 
        use_L1=True,
        use_L2=True, 
        use_gating=True, 
        use_long_range=False,
    )

# ==========================================
# 2. 辅助函数
# ==========================================
def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return local_rank, rank, world_size
    else:
        print("⚠️ Warning: Running in Single GPU Mode")
        return 0, 0, 1

def log_info(msg, rank):
    if rank == 0:
        print(msg)

def get_dataloader(data_dir, meta_file, rank, world_size, is_train=True):
    full_path = os.path.join(data_dir, meta_file)
    if not os.path.exists(full_path):
        if is_train:
            raise FileNotFoundError(f"❌ Error: {meta_file} not found!")
        else:
            log_info(f"⚠️ Warning: {meta_file} not found, skipping...", rank)
            return None, None

    dataset = ChunkedSmartDataset_h5(
        data_dir, metadata_file=meta_file, rank=rank, world_size=world_size
    )

    sampler = BinPackingSampler(
        dataset.metadata,
        max_cost=Config.MAX_COST_PER_BATCH,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    return loader, sampler

def build_model(device, rank, model_config, state_dict=None):
    """
    统一构建逻辑：
    1. 根据 model_config 实例化
    2. 如果有 state_dict 则加载权重
    3. 如果没有 state_dict 则尝试加载 E0
    """
    model = HTGPModel(model_config).to(device)

    if state_dict is not None:
        # --- 加载权重 ---
        if rank == 0:
            log_info("📥 Loading state_dict from checkpoint...", rank)
        
        # 处理 DDP 的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        
        # 加载权重 (Strict=False 以防 E0 buffers 不匹配，视情况而定)
        model.load_state_dict(new_state_dict, strict=False) 
    else:
        # --- 新训练：加载 E0 ---
        if os.path.exists(Config.E0_PATH):
            meta_data = torch.load(Config.E0_PATH, map_location='cpu', weights_only=False)
            e0_dict = meta_data.get('e0_dict', None)
            model.load_external_e0(e0_dict)
            model.atomic_ref.weight.requires_grad = False
            if rank == 0:
                log_info(f"✨ Injected E0 from {Config.E0_PATH}", rank)
        else:
            model.atomic_ref.weight = model.atomic_ref.weight.float()
            log_info("⚠️ E0 file not found, skipping injection.", rank)

    # 打印参数量
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        log_info(f"🧠 Model Parameters: {param_count:,}", rank)

    # DDP 包装
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
    
    return model

# ==========================================
# 3. 主程序 (Main)
# ==========================================
def main():
    # --- A. 初始化环境 ---
    local_rank, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        log_info(f"\n🚀 [Start] World Size: {world_size} | Device: {device}", rank)

    # --- B. 准备数据 ---
    log_info("\n[1/4] Initializing DataLoaders...", rank)
    train_loader, train_sampler = get_dataloader(Config.DATA_DIR, Config.TRAIN_META, rank, world_size, is_train=True)
    test_loader, test_sampler = get_dataloader(Config.DATA_DIR, Config.TEST_META, rank, world_size, is_train=False)

    # --- C. 准备配置 (Restart 逻辑的核心) ---
    # !!! 设置这里 !!!
    RESTART = True
    CHECKPOINT_PATH = "Checkpoints_break_2/model_epoch_47.pt"
    
    start_epoch = 0
    checkpoint_state = None
    
    # 估算总步数 (Trainer 需要用)
    train_total_steps = train_sampler.precompute_total_steps(Config.EPOCHS)
    
    # 默认模型配置
    model_config = HTGPConfig(**Config.MODEL_PARAMS)

    if RESTART:
        log_info(f"\n🔄 Resuming from {CHECKPOINT_PATH}...", rank)
        # 加载 Checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        # 1. 恢复 Epoch
        start_epoch = checkpoint.get('epoch', 47)
        
        # 2. 恢复 Config (非常重要，这就包含了 avg_neighborhood)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            log_info(f"   Loaded config from checkpoint (avg_neigh={model_config.avg_neighborhood:.2f})", rank)
        else:
            # 如果之前的 checkpoint 没存 config (如你上个问题所说)，这里做个兜底
            log_info("⚠️ No config in checkpoint, using default derived from data.", rank)
            model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight
            
        # 3. 获取权重字典
        if 'model_state_dict' in checkpoint:
            checkpoint_state = checkpoint['model_state_dict']
        else:
            # 兼容只保存了 state_dict 的情况
            checkpoint_state = checkpoint 
            
        # 4. 计算 Resume Step (用于 OneCycleLR)
        # 假设 sampler 逻辑不变，估算之前的总步数
        steps_per_epoch_est = train_total_steps // Config.EPOCHS
        resume_step = start_epoch * steps_per_epoch_est - 1
        log_info(f"   Resuming OneCycleLR from step: {resume_step}", rank)
        
    else:
        # 新训练模式
        log_info("\n🆕 Starting New Training...", rank)
        # 计算 avg_neighborhood
        model_config.avg_neighborhood = 1.0 / train_sampler.edge_weight
        resume_step = -1  # 默认值，代表从头开始

    # --- D. 构建模型 ---
    log_info("\n[2/4] Building Model...", rank)
    # 无论是否 restart，统一调用 build_model
    model = build_model(device, rank, model_config, state_dict=checkpoint_state)

    # --- E. 初始化 Trainer ---
    log_info("\n[3/4] Initializing Trainer...", rank)
    
    # 注意：你需要确保你的 PotentialTrainer 能够接收 last_epoch 参数
    # 并传递给 optim.lr_scheduler.OneCycleLR(..., last_epoch=last_epoch)
    trainer = PotentialTrainer(
        model, 
        total_steps=train_total_steps,
        max_lr=Config.LR, 
        device=device, 
        checkpoint_dir=Config.LOG_DIR,
        last_epoch=resume_step  # <--- 将计算好的步数传入
    )

    if RESTART and checkpoint is not None:
        log_info("🔄 Restoring Optimizer, Scheduler and EMA states...", rank)
        trainer.load_checkpoint(checkpoint) # <--- 调用新方法

    # --- F. 训练循环 ---
    log_info(f"\n[4/4] Starting Loop (Epoch {start_epoch + 1} -> {Config.EPOCHS})...", rank)
    log_info("="*60, rank)

    # 循环从 start_epoch + 1 开始
    for epoch in range(start_epoch + 1, Config.EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        
        # 1. Train
        train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch)
        
        # 2. Validate
        if test_loader:
            val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
        else:
            val_metrics = {'total_loss': 0.0, 'mae_f': 0.0}

        # 3. Log & Save
        if rank == 0:
            log_msg = (
                f"Ep {epoch:03d} | "
                f"T_Loss: {train_metrics['total_loss']:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics['mae_f']*1000:.1f}/{val_metrics['mae_f']*1000:.1f} meV/A"
            )
            print(log_msg)
            # 保存 Checkpoint
            save_dict = {
                'epoch': epoch,
                'model_config': model_config,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                
                # --- 原有部分 ---
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                
                # 🔥 新增：务必保存 EMA 状态！
                'ema_state_dict': trainer.ema.state_dict(), 
            }
            torch.save(save_dict, os.path.join(Config.LOG_DIR, f'model_epoch_{epoch}.pt'))

        if dist.is_initialized():
            dist.barrier()

    log_info("\n✅ Training Finished!", rank)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1" 
    main()
