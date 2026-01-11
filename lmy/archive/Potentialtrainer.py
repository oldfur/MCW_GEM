import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm.auto import tqdm # ✅ [新增] 进度条库
from Utils import scatter_add # ✅ [新增] 必须导入，否则 step 函数会报错

class PotentialTrainer:
    def __init__(self, model, lr=1e-3, device='cuda', checkpoint_dir='checkpoints'):
        self.device = device
        self.model = model.to(self.device)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        # 调整了 patience，让它对 Loss 变化更敏感一点
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.75, min_lr=1e-6, patience=10
        )
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        
        # 权重
        self.w_e = 1.0
        self.w_f = 100.0
        self.w_s = 0.1 
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.EV_A3_TO_GPA = 160.21766 

    def step(self, batch, train=True):
        batch = batch.to(self.device)
        
        # 1. 开启梯度 (对原始坐标)
        batch.pos.requires_grad_(True)
        # 必须也对 Cell 开启梯度
        if hasattr(batch, 'cell') and batch.cell is not None:
            batch.cell.requires_grad_(True) 
        
        # ==================================================================
        # A. 构造虚拟应变 (Virtual Strain) 用于计算应力
        # ==================================================================
        num_graphs = batch.batch.max().item() + 1
        
        # 创建位移梯度
        displacement = torch.zeros(
            (num_graphs, 3, 3), 
            dtype=batch.pos.dtype, 
            device=self.device
        )
        displacement.requires_grad_(True)
        
        # 对称化应变
        symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        
        # ==================================================================
        # B. 应用变形 (Deformation)
        # ==================================================================
        # 1. 变形原子坐标
        strain_per_atom = symmetric_strain[batch.batch]
        pos_update = torch.einsum('ni,nij->nj', batch.pos, strain_per_atom)
        pos_deformed = batch.pos + pos_update
        
        # 2. 变形晶胞
        if hasattr(batch, 'cell') and batch.cell is not None:
            if batch.cell.dim() == 3: # [Batch, 3, 3]
                 strain_per_cell = symmetric_strain 
                 cell_update = torch.bmm(batch.cell, strain_per_cell) 
                 cell_deformed = batch.cell + cell_update
            else:
                 cell_deformed = batch.cell
        else:
            cell_deformed = None

        # ==================================================================
        # C. 模型前向
        # ==================================================================
        original_pos = batch.pos
        original_cell = getattr(batch, 'cell', None)
        
        # 临时替换为变形后的坐标
        batch.pos = pos_deformed
        if cell_deformed is not None:
            batch.cell = cell_deformed
        
        # Forward
        pred_e = self.model(batch).view(-1)
        
        # 恢复现场
        batch.pos = original_pos
        if original_cell is not None:
            batch.cell = original_cell
        
        # ==================================================================
        # D. 自动求导计算力与应力
        # ==================================================================
        grad_out = torch.ones_like(pred_e)
        
        # 求导对象: [原始坐标, 虚拟位移]
        grads = torch.autograd.grad(
            outputs=pred_e, 
            inputs=[original_pos, displacement], 
            grad_outputs=grad_out,
            create_graph=train, 
            retain_graph=train,
            allow_unused=True
        )
        
        pred_f = -grads[0] if grads[0] is not None else torch.zeros_like(batch.pos)
        dE_dStrain = grads[1] # Virial
        
        # 计算 Stress (Pressure单位): Sigma = Virial / Volume
        if hasattr(batch, 'volume'):
            vol = batch.volume.view(-1, 1, 1)
        else:
            vol = torch.ones_like(dE_dStrain)
            
        pred_stress = dE_dStrain / vol
        
        # ==================================================================
        # E. Loss 计算
        # ==================================================================
        target_e = batch.y.view(-1)
        # 使用 scatter_add 计算每个图的原子数
        num_atoms = scatter_add(torch.ones_like(batch.batch, dtype=torch.float), batch.batch, dim=0).view(-1).clamp(min=1)
        
        # Energy Loss (Per Atom)
        loss_e = self.criterion_mse(pred_e / num_atoms, target_e / num_atoms)
        
        # Force Loss
        loss_f = self.criterion_mse(pred_f, batch.force)
        
        # Stress Loss (带 Mask)
        if hasattr(batch, 'stress') and batch.stress is not None:
            stress_norm = torch.norm(batch.stress.view(num_graphs, -1), dim=1)
            stress_mask = (stress_norm > 1e-6).float() 
            
            if stress_mask.sum() > 0:
                stress_sq_diff = (pred_stress - batch.stress)**2
                loss_s = (stress_sq_diff.mean(dim=(1, 2)) * stress_mask).sum() / (stress_mask.sum() + 1e-6)
            else:
                loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)
        else:
            loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)

        total_loss = self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        
        # ==================================================================
        # Metrics
        # ==================================================================
        with torch.no_grad():
            mae_e = self.criterion_mae(pred_e / num_atoms, target_e / num_atoms).item()
            mae_f = self.criterion_mae(pred_f, batch.force).item()
            
            mae_s_gpa = 0.0
            if hasattr(batch, 'stress') and batch.stress is not None:
                if stress_mask.sum() > 0:
                    mae_s_val = (torch.abs(pred_stress - batch.stress).mean(dim=(1,2)) * stress_mask).sum() / stress_mask.sum()
                    mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA

        # Optimization
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return total_loss.item(), mae_e, mae_f, mae_s_gpa

    # ==================================================================
    # 🔥 修改后的 train_epoch：支持 total_steps 进度条
    # ==================================================================
    def train_epoch(self, loader, total_steps=None):
        self.model.train()
        
        # 累加器
        metrics = {'total': 0, 'mae_e': 0, 'mae_f': 0, 'mae_s': 0}
        count = 0
        
        # 使用 tqdm 包装 loader
        # total=total_steps 告诉进度条终点在哪里
        pbar = tqdm(loader, total=total_steps, desc="Training", leave=False)
        
        for batch in pbar:
            l, me, mf, ms = self.step(batch, train=True)
            
            metrics['total'] += l
            metrics['mae_e'] += me
            metrics['mae_f'] += mf
            metrics['mae_s'] += ms
            count += 1
            
            # 实时更新进度条后缀
            pbar.set_postfix({
                'Loss': f"{l:.4f}", 
                'MAE_F': f"{mf*1000:.3f}"
            })
            
        # 防止 count 为 0 (例如数据集为空)
        if count == 0: count = 1
            
        return {k: v/count for k, v in metrics.items()}

    # ==================================================================
    # 🔥 修改后的 validate：支持 total_steps 进度条
    # ==================================================================
    def validate(self, loader, total_steps=None):
        self.model.eval()
        
        metrics = {'total': 0, 'mae_e': 0, 'mae_f': 0, 'mae_s': 0}
        count = 0
        
        pbar = tqdm(loader, total=total_steps, desc="Validating", leave=False)
        
        # 验证时开启梯度用于计算 Force，但不需要反向传播优化
        # 使用 torch.set_grad_enabled(True) 确保 step 函数里的 autograd.grad 能工作
        with torch.set_grad_enabled(True): 
            for batch in pbar:
                # 传入 train=False，这样 optimizer 不会 step
                l, me, mf, ms = self.step(batch, train=False)
                
                metrics['total'] += l
                metrics['mae_e'] += me
                metrics['mae_f'] += mf
                metrics['mae_s'] += ms
                count += 1
        
        if count == 0: count = 1
        avg_metrics = {k: v/count for k, v in metrics.items()}
        
        # 根据验证集的 Force MAE 调整学习率
        self.scheduler.step(avg_metrics['mae_f'])
        
        return avg_metrics

    def save(self, filename='best_model.pt'):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
