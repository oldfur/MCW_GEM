import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
from tqdm.auto import tqdm
from Utils import scatter_add
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage

def conditional_huber_loss(pred, target, base_delta=0.01):
    """
    自适应 Huber Loss (Adaptive Huber Loss)。
    针对力(Force)数据跨度大的物理特性设计。
    
    机制:
    根据真实力(Target Force)的模长动态调整 Huber Loss 的阈值(delta)。
    - 平衡态(力小): 使用 base_delta, 保持 MSE 的高精度特性。
    - 剧烈动态(力大): 减小 delta, 使 Loss 更早进入 L1 线性区, 防止梯度爆炸。
    
    Args:
        pred: 预测值 (N_atoms, 3)
        target: 真实值 (N_atoms, 3)
        base_delta: 基础阈值, 默认 0.01
    """
    # 计算每个原子的受力模长 (N, 1)
    force_norm = torch.norm(target, dim=1, keepdim=True)
    
    # 初始化缩放因子
    delta_scale = torch.ones_like(force_norm)
    
    # 阶梯式降级策略
    # Force < 100: scale = 1.0
    # 100 <= Force < 200: scale = 0.7
    mask_100_200 = (force_norm >= 100) & (force_norm < 200)
    delta_scale[mask_100_200] = 0.7
    
    # 200 <= Force < 300: scale = 0.4
    mask_200_300 = (force_norm >= 200) & (force_norm < 300)
    delta_scale[mask_200_300] = 0.4
    
    # Force >= 300: scale = 0.1 (极端值使用强鲁棒性 L1)
    mask_300 = (force_norm >= 300)
    delta_scale[mask_300] = 0.1
    
    # 计算最终的 delta (N, 1) -> 广播到 (N, 3)
    adaptive_delta = base_delta * delta_scale
    
    # 手动实现 Huber 计算逻辑
    error = pred - target
    abs_error = torch.abs(error)
    
    # 判定 MSE 区域
    is_mse = abs_error < adaptive_delta
    
    loss_mse = 0.5 * error ** 2
    loss_l1 = adaptive_delta * (abs_error - 0.5 * adaptive_delta)
    
    # 组合并取平均
    loss = torch.where(is_mse, loss_mse, loss_l1)
    return loss.mean()

class PotentialTrainer:
    def __init__(self, model, steps_per_epoch, epochs, lr=1e-3, device='cuda', checkpoint_dir='checkpoints'):
        """
        Args:
            steps_per_epoch: 每个 Epoch 的步数 (用于 Scheduler 规划曲线)
            epochs: 总训练轮次
        """
        self.device = device
        self.model = model.to(self.device)
        
        # ------------------------------------------------------------------
        # 1. 优化器配置
        # 使用 AdamW 并开启 AMSGrad 以提升收敛稳定性
        # Weight Decay 设为 1e-4，提供轻微正则化
        # ------------------------------------------------------------------
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4, 
            amsgrad=True
        )

        # ------------------------------------------------------------------
        # 2. EMA (指数移动平均)
        # 维护一份平滑的权重副本，用于验证和推理，极大提升泛化能力
        # ------------------------------------------------------------------
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        # ------------------------------------------------------------------
        # 3. 学习率调度器 (OneCycleLR)
        # 针对短周期(Few Epochs)大模型训练的策略
        # ------------------------------------------------------------------
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,           # 5% 步数用于预热(Warmup)，防止初期力梯度过大
            div_factor=100.0,        # 初始学习率为 max_lr / 100，起步更稳
            final_div_factor=1000.0, # 最终衰减到极小值
            anneal_strategy='cos'
        )
        
        # Loss 配置
        self.huber_delta = 0.01  # 基础 Delta
        self.w_e = 1.0           # 能量权重
        self.w_f = 10.0          # 力权重 (配合 Huber Loss 使用 10, 若用 MSE 需更大)
        self.w_s = 10.0          # 应力权重
        
        # 获取 rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpoint_dir = checkpoint_dir
        self.train_log_path = os.path.join(self.checkpoint_dir, 'train_log.csv')
        self.val_log_path = os.path.join(self.checkpoint_dir, 'val_log.csv')
        self.EV_A3_TO_GPA = 160.21766 
        
        # 日志初始化
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_loggers()

    def _init_loggers(self):
        headers = ['epoch', 'step', 'lr', 'total_loss', 'loss_e', 'loss_f', 'loss_s', 'mae_e', 'mae_f', 'mae_s_gpa']
        for path in [self.train_log_path, self.val_log_path]:
            # 覆盖模式初始化 (如需续训请改为 'a' 并增加判断)
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

    def log_to_csv(self, mode, data):
        # 只有 rank 0 写入
        if self.rank != 0:
            return
        path = self.train_log_path if mode == 'train' else self.val_log_path
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow([
                data['epoch'], data['step'], f"{data['lr']:.2e}",
                f"{data['total_loss']:.6f}", f"{data['loss_e']:.6f}",
                f"{data['loss_f']:.6f}", f"{data['loss_s']:.6f}",
                f"{data['mae_e']*1000:.6f} meV/atom", f"{data['mae_f']*1000:.6f} meV", f"{data['mae_s_gpa']:.6f} GPa"
            ])

    def step(self, batch, train=True):
        batch = batch.to(self.device)
        
        # --- 1. 开启梯度 (Force & Stress 计算所需) ---
        batch.pos.requires_grad_(True)
        if hasattr(batch, 'cell') and batch.cell is not None:
            batch.cell.requires_grad_(True) 
        
        # --- 2. 构造虚拟应变 (Virtual Strain) ---
        num_graphs = batch.batch.max().item() + 1
        displacement = torch.zeros((num_graphs, 3, 3), dtype=batch.pos.dtype, device=self.device)
        displacement.requires_grad_(True)
        symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        
        # --- 3. 应用变形 ---
        strain_per_atom = symmetric_strain[batch.batch]
        pos_deformed = batch.pos + torch.einsum('ni,nij->nj', batch.pos, strain_per_atom)
        
        original_pos = batch.pos
        original_cell = getattr(batch, 'cell', None)
        
        batch.pos = pos_deformed
        
        if original_cell is not None and original_cell.dim() == 3:
            # batch.cell 通常是 (Batch, 3, 3)
            # 这里的乘法逻辑取决于你的 cell 定义是行向量还是列向量，通常 ASE/PyG 是行向量
            cell_deformed = original_cell + torch.bmm(original_cell, symmetric_strain)
            batch.cell = cell_deformed
        else:
            print("⚠️ Warning: batch.cell is None or not 3D, skipping cell deformation.")
            # 程序停止在这里以防后续报错
            raise ValueError("batch.cell is None or not 3D") 

        # --- 4. 前向传播 ---
        pred_e = self.model(batch).view(-1)
        
        # 恢复原始坐标
        batch.pos = original_pos
        if original_cell is not None: batch.cell = original_cell
        
        # --- 5. 自动求导计算力与应力 ---
        grad_out = torch.ones_like(pred_e)
        grads = torch.autograd.grad(
            outputs=pred_e, 
            inputs=[original_pos, displacement], 
            grad_outputs=grad_out,
            create_graph=train, 
            retain_graph=train,
            allow_unused=True
        )
        
        pred_f = -grads[0] if grads[0] is not None else torch.zeros_like(batch.pos)
        dE_dStrain = grads[1]

        # --- 6. 🔥🔥🔥 修正体积计算 🔥🔥🔥 ---
        if original_cell is not None:
            # 实时计算体积：Vol = det(Cell)
            # torch.linalg.det 计算行列式
            vol = torch.abs(torch.linalg.det(original_cell)) # (Batch,)
            
            # 调整形状以便广播: (Batch,) -> (Batch, 1, 1)
            vol = vol.view(-1, 1, 1)
        else:
            # 非周期性体系（分子），没有体积定义，通常不计算 Stress
            vol = torch.ones_like(dE_dStrain)

        pred_stress = dE_dStrain / vol
        
        # ==================================================================
        # 6. Loss 计算 (使用增强版 Huber Loss)
        # ==================================================================
        target_e = batch.y.view(-1)
        num_atoms = scatter_add(torch.ones_like(batch.batch, dtype=torch.float64), batch.batch, dim=0).view(-1).clamp(min=1)
        
        # Energy: 普通 Huber
        loss_e = F.huber_loss(pred_e / num_atoms, target_e / num_atoms, delta=self.huber_delta)
        
        # Force: 自适应 Conditional Huber
        loss_f = conditional_huber_loss(pred_f, batch.force, base_delta=self.huber_delta)
        
        # Stress: 普通 Huber (带 Mask)
        loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)
        stress_mask_sum = 0
        if hasattr(batch, 'stress') and batch.stress is not None:
            stress_norm = torch.norm(batch.stress.view(num_graphs, -1), dim=1)
            stress_mask = (stress_norm > 1e-6)
            stress_mask_sum = stress_mask.sum().item()
            if stress_mask_sum > 0:
                s_pred = pred_stress.view(num_graphs, -1)[stress_mask]
                s_target = batch.stress.view(num_graphs, -1)[stress_mask]
                loss_s = F.huber_loss(s_pred, s_target, delta=self.huber_delta)

        total_loss = self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        
        # --- 7. Metrics 计算 (MAE, 物理单位) ---
        with torch.no_grad():
            # 使用 L1 Loss 计算 MAE
            mae_e = F.l1_loss(pred_e / num_atoms, target_e / num_atoms).item()
            mae_f = F.l1_loss(pred_f, batch.force).item()
            mae_s_gpa = 0.0
            if stress_mask_sum > 0:
                mae_s_val = F.l1_loss(
                    pred_stress.view(num_graphs, -1)[stress_mask], 
                    batch.stress.view(num_graphs, -1)[stress_mask]
                )
                mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA

        # --- 8. 反向传播与优化 ---
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            
            # 梯度裁剪防止爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # 🔥 关键: 每次参数更新后，立即更新 EMA 影子权重
            self.ema.update()
            
        return {
            'total_loss': total_loss.item(),
            'loss_e': loss_e.item(), 'loss_f': loss_f.item(), 'loss_s': loss_s.item(),
            'mae_e': mae_e, 'mae_f': mae_f, 'mae_s_gpa': mae_s_gpa
        }

    def train_epoch(self, loader, epoch_idx):
        self.model.train()
        pbar = tqdm(loader, desc=f"Train Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        current_step = 0
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0
        
        for batch in pbar:
            # 1. 训练一步
            metrics = self.step(batch, train=True)
            # 打印第一个batch的图，原子和边的索引信息验证它们是否对应的正确性, 看batch.batch是否和原子数对应
            if current_step == 0:
                if self.rank == 0:
                    print("First batch graph info:")
                    print("Number of graphs in batch:", batch.num_graphs)
                    print("Nodes (atoms) in batch:", batch.pos.size(0))
                    print("Edge index:", batch.edge_index)
                    print("Batch indices:", batch.batch)
                    # 看stress是不是不是None和空
                    if hasattr(batch, 'stress') and batch.stress is not None:
                        print("Stress tensor shape:", batch.stress.shape)
                    else:
                        print("No stress tensor in this batch.")
            # 2. 记录 CSV
            log_data = metrics.copy()
            log_data.update({'epoch': epoch_idx, 'step': current_step, 'lr': self.optimizer.param_groups[0]['lr']})
            self.log_to_csv('train', log_data)
            
            # 3. 🔥 Scheduler Step (Batch-level)
            # 必须在每个 batch 后调用，确保 OneCycleLR 曲线生效
            self.scheduler.step()
            
            # 4. 统计
            for k in metrics_sum: metrics_sum[k] += metrics[k]
            count += 1
            current_step += 1
            pbar.set_postfix({'L': f"{metrics['total_loss']:.4f}", 
                              'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                              'MAE_F': f"{metrics['mae_f']*1000:.1f}"})
            
        return {k: v/count for k,v in metrics_sum.items()}

    def validate(self, loader, epoch_idx):
        # 验证时不应使用 train 模式，也不应更新梯度
        # 但如果是 Graph Norm 等层，需注意 eval 模式的行为
        self.model.eval()
        pbar = tqdm(loader, desc=f"Val Ep {epoch_idx}", leave=False, disable=(self.rank != 0))
        metrics_sum = {'mae_e': 0, 'mae_f': 0, 'mae_s_gpa': 0, 'total_loss': 0}
        count = 0
        current_step = 0
        
        # 🔥 关键: 使用 EMA 的平滑权重进行验证，通常能获得更低且更稳的 Error
        with self.ema.average_parameters():
            with torch.set_grad_enabled(True): # 必须开启 grad 才能计算 Force
                for batch in pbar:
                    metrics = self.step(batch, train=False)
                    
                    # 记录 CSV
                    log_data = metrics.copy()
                    log_data.update({'epoch': epoch_idx, 'step': current_step, 'lr': self.optimizer.param_groups[0]['lr']})
                    self.log_to_csv('val', log_data)
                    
                    for k in metrics_sum: metrics_sum[k] += metrics[k]
                    count += 1
                    current_step += 1
                    pbar.set_postfix({'L': f"{metrics['total_loss']:.4f}", 
                              'MAE_e': f"{metrics['mae_e']*1000:.1f}",
                              'MAE_F': f"{metrics['mae_f']*1000:.1f}"})
        
        if count == 0: count = 1
        return {k: v/count for k,v in metrics_sum.items()}

    def save(self, filename='best_model.pt'):
        path = os.path.join(self.checkpoint_dir, filename)
        # 保存时，推荐保存 EMA 处理过的权重作为最佳模型
        with self.ema.average_parameters():
            torch.save(self.model.state_dict(), path)
            