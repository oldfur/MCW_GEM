import os
import json
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

# --- 导入自定义模块 (根据你的项目结构) ---
from src.data import ChunkedSmartDataset_h5, BinPackingSampler
from src.models import HTGPModel
from src.utils import HTGPConfig
from src.engine import PotentialTrainer 

# ==========================================
# 0. 全局环境设置 (Environment Setup)
# ==========================================
# 解决多进程文件打开数限制问题
torch.multiprocessing.set_sharing_strategy('file_system')

# 设置默认精度
torch.set_default_dtype(torch.float32)

# 🚀 开启 TF32 (NVIDIA Ampere/Hopper 架构加速神器)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 训练配置 (Configuration)
# ==========================================
class Config:
    # 路径配置
    DATA_DIR = "../dataset_h5"      # 数据根目录
    TRAIN_META = "train_metadata.pt"         # 训练集元数据
    TEST_META = "test_metadata.pt"           # 测试集元数据
    E0_PATH = "../dataset_h5/meta_data.pt" # 原子能量参考值
    LOG_DIR = "../lmy_Checkpoints"                  # 模型保存路径

    # 训练超参
    # 🔥 注意: 这里的 BATCH_SIZE 指的是 "每个 Batch 的最大原子数 (Cost)"
    MAX_COST_PER_BATCH = 2000  # 针对 H100/A100 优化
    LR = 1e-3
    EPOCHS = 45
    
    # 系统配置
    NUM_WORKERS = 8            # DataLoader 进程数
    PREFETCH_FACTOR = 2        # 预取因子

    # 模型配置 (HTGP)
    MODEL_PARAMS = dict(
        num_atom_types=100, 
        hidden_dim=128, 
        num_layers=2, 
        cutoff=6.0, 
        num_rbf=10,
        use_L0=True, 
        use_L1=True,
        use_L2=True, 
        use_gating=True, 
        use_long_range=False
    )

# ==========================================
# 2. 辅助函数 (Utils)
# ==========================================
def init_distributed_mode():
    """初始化 DDP 分布式环境"""
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
    """仅在主进程打印日志"""
    if rank == 0:
        print(msg)

def get_dataloader(data_dir, meta_file, rank, world_size, is_train=True):
    """构建 Dataset, Sampler 和 DataLoader"""
    full_path = os.path.join(data_dir, meta_file)
    if not os.path.exists(full_path):
        if is_train:
            raise FileNotFoundError(f"❌ 致命错误: 没找到 {meta_file}，请先运行 preprocess.py！")
        else:
            log_info(f"⚠️ Warning: {meta_file} not found, skipping...", rank)
            return None, None

    # 1. Dataset
    dataset = ChunkedSmartDataset_h5(
        data_dir, 
        metadata_file=meta_file, 
        rank=rank,
        world_size=world_size
    )

    # 2. Sampler (训练用 Shuffle, 测试不用)
    sampler = BinPackingSampler(
        dataset.metadata,
        max_cost=Config.MAX_COST_PER_BATCH,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank
    )

    # 3. Loader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler, # 关键：使用 batch_sampler 处理动态 Batch
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    
    return loader, sampler

def build_model(device, rank, avg_neighborhood, **karwgs):
    """构建模型并加载 E0"""
    
    # 初始化配置和模型

    if "restart" not in karwgs:
        model_config = HTGPConfig(**Config.MODEL_PARAMS)
        model_config.avg_neighborhood = avg_neighborhood
        model = HTGPModel(model_config).to(device)
    else:
        model_config = karwgs["model_config"]
        model_config.avg_neighborhood = avg_neighborhood
        model = HTGPModel(model_config).to(device)
        state_dict = karwgs["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        print()
    
    # 打印参数量
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        log_info(f"🧠 Model Parameters: {param_count:,}", rank)


    # 注入 E0 (原子参考能量)
    if "restart" not in karwgs:
        if os.path.exists(Config.E0_PATH):
            # map_location='cpu' 防止占用显存
            meta_data = torch.load(Config.E0_PATH, map_location='cpu', weights_only=False)
            e0_dict = meta_data.get('e0_dict', None)
            
            model.load_external_e0(e0_dict)
            model.atomic_ref.weight.requires_grad = False # 冻结 E0
            if rank == 0:
                log_info(f"Adding E0 from {Config.E0_PATH}...", rank)

        else:
            # 如果没有 E0 文件，确保类型正确
            model.atomic_ref.weight = model.atomic_ref.weight.float()
            log_info("⚠️ meta_e0_data.pt not found, skipping E0 injection.", rank)

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
        log_info("="*60, rank)

    # --- B. 准备数据 ---
    log_info("\n[1/4] Initializing DataLoaders...", rank)
    
    # 训练集
    train_loader, train_sampler = get_dataloader(
        Config.DATA_DIR, Config.TRAIN_META, rank, world_size, is_train=True
    )
    
    # 测试集 
    test_loader, test_sampler = get_dataloader(
        Config.DATA_DIR, Config.TEST_META, rank, world_size, is_train=False
    )
    # 这里的解包逻辑稍微改一下，防止 test_result 为 None 报错
    # test_loader = test_result[0] if test_result else None

    # --- C. 构建模型 ---
    log_info("\n[2/4] Building Model...", rank)\
    
    restart = True  # 是否从检查点恢复训练
    avg_neighborhood = 1 / train_sampler.edge_weight
    if not restart:
        model = build_model(device, rank, avg_neighborhood)
    else:
        checkpoint_path = "../lmy_Checkpoints/model_epoch_5.pt"
        checkpoint_weights = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = checkpoint_weights['model_config']

        model = build_model(device, rank, avg_neighborhood, restart=restart, model_config=saved_config, state_dict=checkpoint_weights)

    # --- D. 初始化 Trainer ---
    log_info("\n[3/4] Initializing Trainer...", rank)
    
    # 估算总步数 (因为是动态 Batch，步数不是固定的 len/bs，需要从 sampler 获取)
    train_total_steps = train_sampler.precompute_total_steps(Config.EPOCHS)
    log_info(f"📊 Estimated total steps for training: {train_total_steps}", rank)

    # 🔥 修改 2: 必须加 if 判断，否则 test_sampler 为 None 时会报错
    if test_sampler is not None:
        test_total_steps = test_sampler.precompute_total_steps(Config.EPOCHS)
        log_info(f"📊 Estimated total steps for testing: {test_total_steps}", rank)

    if not restart:
        trainer = PotentialTrainer(
        model, 
        total_steps=train_total_steps,
        max_lr=Config.LR, 
        device=device, 
        checkpoint_dir=Config.LOG_DIR)
    else:
        trainer = PotentialTrainer(
        model, 
        total_steps=train_total_steps,
        max_lr=Config.LR, 
        device=device, 
        checkpoint_dir=Config.LOG_DIR)

    # --- E. 训练循环 ---
    log_info("\n[4/4] Starting Loop...", rank)
    log_info("="*60, rank)


    for epoch in range(1, Config.EPOCHS + 1):
        # 重要：每个 Epoch 设置随机种子，保证 Shuffle 效果
        train_sampler.set_epoch(epoch)
        
        # 1. Train
        train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch)
        
        # 2. Validate
        if test_loader:
            val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
        else:
            val_metrics = {'total_loss': 0.0, 'mae_f': 0.0}

        # 3. Log & Save (仅 Rank 0)
        if rank == 0:
            log_msg = (
                f"Ep {epoch:03d} | "
                f"T_Loss: {train_metrics['total_loss']:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics['mae_f']*1000:.1f}/{val_metrics['mae_f']*1000:.1f} meV/A"
            )
            print(log_msg)
            trainer.save(f'model_epoch_{epoch}.pt')

        # 4. 同步：确保所有卡都跑完了这个 Epoch
        if dist.is_initialized():
            dist.barrier()

    log_info("\n✅ Training Finished!", rank)
    
    # --- F. 清理 ---
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # 设置 OMP 线程数，防止 CPU 过载
    os.environ["OMP_NUM_THREADS"] = "1" 
    main()