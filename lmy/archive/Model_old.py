from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import (
    GeometricBasis, LeibnizCoupling, PhysicsGating, CartesianDensityBlock, LatentLongRange)
from Utils import scatter_add, HTGPConfig

# ==========================================
# 7. 主模型 (Main Model)
# ==========================================
class HTGPModel(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        
        # Embedding
        self.embedding = nn.Embedding(config.num_atom_types, config.hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        # Components
        self.geom_basis = GeometricBasis(config)
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                'coupling': LeibnizCoupling(config),
                'gating': PhysicsGating(config),
                'density': CartesianDensityBlock(config),
                'readout': nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, 1)
                )
            }))
            
        if config.use_long_range:
            self.long_range = LatentLongRange(config)
        self.atomic_ref = nn.Embedding(60, 1)
        nn.init.zeros_(self.atomic_ref.weight)
            
    def forward(self, data, capture_weights=False):
        # if batch is None: batch = torch.zeros_like(z)
        
        # 1. 几何计算
        row, col = data.edge_index
        # 处理 shifts_int (PBC)
        if hasattr(data, 'shifts_int') and data.shifts_int is not None:
            batch_cell = data.cell[data.batch[row]]          # (E, 3, 3)
            current_shifts = torch.bmm(
                data.shifts_int.unsqueeze(1), batch_cell
            ).squeeze(1)                                     # (E, 3)
        else:
            current_shifts = torch.zeros(
                (row.size(0), 3),
                device=data.pos.device,
                dtype=data.pos.dtype
            )

        
        vec_ij = data.pos[col] - data.pos[row] + current_shifts
        d_ij = torch.norm(vec_ij, dim=-1).clamp(min=1e-8)

        basis_edges, r_hat = self.geom_basis(vec_ij, d_ij)

        # 2. 状态初始化
        h0 = self.embedding(data.z) # (N, F)
        h1 = None 
        h2 = None
        
        total_energy = 0.0
        
        # 3. 层级传递
        for layer in self.layers:
            # A. 莱布尼茨消息生成
            node_feats = {0: h0, 1: h1, 2: h2}
            raw_msgs = layer['coupling'](node_feats, basis_edges, data.edge_index)
            
            # B. 物理门控
            gated_msgs = layer['gating'](raw_msgs, h0, basis_edges[0], r_hat, h1, data.edge_index, capture_weights=capture_weights)
            
            # C. 密度聚合与更新
            delta_h0, delta_h1, delta_h2 = layer['density'](gated_msgs, row, data.z.size(0))

            # D. 残差更新 (Residual Update)
            h0 = h0 + delta_h0

            if self.cfg.use_L1:
                if h1 is None:
                    h1 = delta_h1 # 第一层直接赋值
                elif delta_h1 is not None:
                    h1 = h1 + delta_h1 # 后续层累加

            if self.cfg.use_L2:
                if h2 is None:
                    h2 = delta_h2
                elif delta_h2 is not None:
                    h2 = h2 + delta_h2

            # E. 能量读出
            atomic_energy = layer['readout'](h0)
            total_energy = total_energy + scatter_add(atomic_energy, data.batch, dim=0, dim_size=data.num_graphs)
            
        # 4. 长程修正
        if self.cfg.use_long_range and self.cfg.use_L1 and h1 is not None:
            e_long = self.long_range(h1, data.pos, data.batch)
            total_energy = total_energy + e_long
        total_energy = total_energy + scatter_add(self.atomic_ref(data.z), data.batch, dim=0, dim_size=data.num_graphs)

        return total_energy
