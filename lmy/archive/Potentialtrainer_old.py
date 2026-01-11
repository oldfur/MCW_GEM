import torch
import torch.nn as nn
import torch.optim as optim
import os

class PotentialTrainer:
    def __init__(self, model, lr=1e-3, device='cuda', checkpoint_dir='checkpoints'):
        self.device = device
        self.model = model.to(self.device)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.75, min_lr=1e-6, patience=5)
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        
        # 权重
        self.w_e = 1.0
        self.w_f = 100.0
        self.w_s = 0.1  # [建议] 应力权重不要给 0，通常给 0.1 或 0.01，否则完全不学
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.EV_A3_TO_GPA = 160.21766 

    def step(self, batch, train=True):
        batch = batch.to(self.device)
        
        # 1. 开启梯度 (对原始坐标)
        batch.pos.requires_grad_(True)
        # [Critical Fix] 必须也对 Cell 开启梯度，因为应力对应变的导数也包含晶胞的变化
        batch.cell.requires_grad_(True) 
        
        # ==================================================================
        # A. 构造虚拟应变 (Virtual Strain)
        # ==================================================================
        num_graphs = batch.batch.max().item() + 1
        
        # 创建位移梯度 (Gradient of displacement field)
        displacement = torch.zeros(
            (num_graphs, 3, 3), 
            dtype=batch.pos.dtype, 
            device=self.device
        )
        displacement.requires_grad_(True)
        
        # 对称化应变: strain = 0.5 * (u + u.T)
        symmetric_strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        
        # ==================================================================
        # B. 应用变形 (Deformation) - [关键修复]
        # ==================================================================
        # 1. 变形原子坐标: R' = R + R @ strain
        strain_per_atom = symmetric_strain[batch.batch]
        pos_update = torch.einsum('ni,nij->nj', batch.pos, strain_per_atom)
        pos_deformed = batch.pos + pos_update
        
        # 2. [FIX] 变形晶胞: Cell' = Cell + Cell @ strain
        # 注意: batch.cell 是 [E, 3, 3] 还是 [B, 3, 3]? 
        # Loader 里我们设的是 [1, 3, 3] 然后 batch 堆叠成 [B, 3, 3]
        # 但在 neighbor_list 处理后 PyG 可能会把它当成普通属性。
        # 最稳妥的方式是根据 batch 索引取
        if batch.cell.dim() == 3: # [Batch, 3, 3]
             strain_per_cell = symmetric_strain # [Batch, 3, 3] 一一对应
             cell_update = torch.bmm(batch.cell, strain_per_cell) # [B, 3, 3]
             cell_deformed = batch.cell + cell_update
        else:
             # 如果 cell 是 [1, 3, 3] 广播的异常情况，通常不会发生
             cell_deformed = batch.cell

        # ==================================================================
        # C. 模型前向 (Substitution Strategy)
        # ==================================================================
        original_pos = batch.pos
        original_cell = batch.cell
        
        batch.pos = pos_deformed
        batch.cell = cell_deformed # [FIX] 传入变形后的 Cell
        
        # 模型内部会用 cell_deformed 和 shifts_int 算出变形后的 shifts
        # 从而保证了梯度流的连通性
        pred_e = self.model(batch).view(-1)
        
        # 恢复现场
        batch.pos = original_pos
        batch.cell = original_cell
        
        # ==================================================================
        # D. 自动求导
        # ==================================================================
        grad_out = torch.ones_like(pred_e)
        
        # 求导对象: [原始坐标, 虚拟位移]
        grads = torch.autograd.grad(
            outputs=pred_e, 
            inputs=[original_pos, displacement], 
            grad_outputs=grad_out,
            create_graph=train, 
            retain_graph=train,
            allow_unused=True # 允许某些 graph 没有应力贡献
        )
        
        pred_f = -grads[0] if grads[0] is not None else torch.zeros_like(batch.pos)
        dE_dStrain = grads[1] # [Batch, 3, 3] (Virial)
        
        # 计算 Stress (Pressure单位): Sigma = Virial / Volume
        # 注意: 这里的 dE_dStrain 实际上已经是 total Virial 了
        if hasattr(batch, 'volume'):
            vol = batch.volume.view(-1, 1, 1)
        else:
            vol = torch.ones_like(dE_dStrain)
            
        pred_stress = dE_dStrain / vol
        
        # ==================================================================
        # E. Loss 计算 (带 Mask)
        # ==================================================================
        target_e = batch.y.view(-1)
        num_atoms = scatter_add(torch.ones_like(batch.z, dtype=torch.float), batch.batch, dim=0).view(-1).clamp(min=1)
        
        loss_e = self.criterion_mse(pred_e / num_atoms, target_e / num_atoms)
        loss_f = self.criterion_mse(pred_f, batch.force)
        
        # [FIX] Stress Masking
        # 检查每个 batch 的 stress 标签是否全为0 (通常认为全0且没有标记说明是无标签)
        # 或者你可以约定 label 中 1e10 代表无标签。这里假设全0张量代表缺失。
        # 更稳健的方法是检查 stress 的模长
        stress_norm = torch.norm(batch.stress.view(num_graphs, -1), dim=1)
        stress_mask = (stress_norm > 1e-6).float() # 只有非零标签才算 Loss
        
        if stress_mask.sum() > 0:
            # 只计算有标签的部分
            stress_sq_diff = (pred_stress - batch.stress)**2
            # 这里的 batch.stress [B, 3, 3], loss 需要 reduce
            loss_s = (stress_sq_diff.mean(dim=(1, 2)) * stress_mask).sum() / (stress_mask.sum() + 1e-6)
        else:
            loss_s = torch.tensor(0.0, device=self.device, requires_grad=train)

        total_loss = self.w_e * loss_e + self.w_f * loss_f + self.w_s * loss_s
        
        # ==================================================================
        # Metrics & Optimization
        # ==================================================================
        with torch.no_grad():
            mae_e = self.criterion_mae(pred_e / num_atoms, target_e / num_atoms).item()
            mae_f = self.criterion_mae(pred_f, batch.force).item()
            
            # Metric 也只算有标签的
            if stress_mask.sum() > 0:
                mae_s_val = (torch.abs(pred_stress - batch.stress).mean(dim=(1,2)) * stress_mask).sum() / stress_mask.sum()
                mae_s_gpa = mae_s_val.item() * self.EV_A3_TO_GPA
            else:
                mae_s_gpa = 0.0

        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return total_loss.item(), mae_e, mae_f, mae_s_gpa

    def train_epoch(self, loader):
        self.model.train()
        metrics = {'total': 0, 'mae_e': 0, 'mae_f': 0, 'mae_s': 0}
        steps = len(loader)
        for batch in loader:
            l, me, mf, ms = self.step(batch, train=True)
            metrics['total'] += l
            metrics['mae_e'] += me
            metrics['mae_f'] += mf
            metrics['mae_s'] += ms
        avg_metrics = {k: v/steps for k, v in metrics.items()}
     #    self.scheduler.step(avg_metrics['mae_f'])    
        return {k: v/steps for k, v in metrics.items()}

    def validate(self, loader):
        self.model.eval()
        metrics = {'total': 0, 'mae_e': 0, 'mae_f': 0, 'mae_s': 0}
        steps = len(loader)
        if steps == 0: return metrics
        # 验证时开启梯度用于计算 Force
        with torch.set_grad_enabled(True): 
            for batch in loader:
                l, me, mf, ms = self.step(batch, train=False)
                metrics['total'] += l
                metrics['mae_e'] += me
                metrics['mae_f'] += mf
                metrics['mae_s'] += ms
        
        avg_metrics = {k: v/steps for k, v in metrics.items()}
        self.scheduler.step(avg_metrics['mae_f'])
        return avg_metrics

    def save(self, filename='best_model.pt'):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
