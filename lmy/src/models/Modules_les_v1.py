import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, List
from src.utils import scatter_add, scatter_mean, HTGPConfig

# ==========================================
# 🔥 核心 JIT 数学引擎 (安全加速区)
# ==========================================

@torch.jit.script
def compute_bessel_math(d: torch.Tensor, r_max: float, freq: torch.Tensor) -> torch.Tensor:
    d_scaled = d / r_max
    prefactor = (2.0 / r_max) ** 0.5
    return prefactor * torch.sin(freq * d_scaled) / (d + 1e-6)
 
@torch.jit.script
def compute_envelope_math(d: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = d / r_cut
    x = torch.clamp(x, min=0.0, max=1.0)
    return 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5

@torch.jit.script
def compute_l2_basis(rbf_feat: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    outer = r_hat.unsqueeze(2) * r_hat.unsqueeze(1) 
    eye = torch.eye(3, dtype=r_hat.dtype, device=r_hat.device).unsqueeze(0)
    trace_less = outer - (1.0/3.0) * eye
    return rbf_feat.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1)

@torch.jit.script
def compute_invariants(den0: Optional[torch.Tensor], 
                       den1: Optional[torch.Tensor], 
                       den2: Optional[torch.Tensor]) -> torch.Tensor:
    # ✅ 修复：使用标准类型标注
    invariants: List[torch.Tensor] = []
    
    if den0 is not None:
        invariants.append(den0)
        
    if den1 is not None:
        sq_sum = torch.sum(den1.pow(2), dim=1) 
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    if den2 is not None:
        sq_sum = torch.sum(den2.pow(2), dim=(1, 2))
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    if len(invariants) > 0:
        return torch.cat(invariants, dim=-1)
    else:
        # 返回空 Tensor (注意处理 device 问题，最好由外部保证 invariants 不为空)
        return torch.zeros(0) 

@torch.jit.script
def compute_gating_projections(h_node1: torch.Tensor, 
                               r_hat: torch.Tensor, 
                               scalar_basis: torch.Tensor,
                               src: torch.Tensor, 
                               dst: torch.Tensor) -> torch.Tensor:
    r_hat_uns = r_hat.unsqueeze(-1)
    p_src = torch.sum(h_node1[src] * r_hat_uns, dim=1)
    p_dst = torch.sum(h_node1[dst] * r_hat_uns, dim=1)
    return torch.cat([scalar_basis, p_src, p_dst], dim=-1)


# ==========================================
# 🧩 模块定义 (普通 nn.Module 区)
# ==========================================

class BesselBasis(nn.Module): 
    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.register_buffer("freq", torch.arange(1, num_basis + 1).float() * np.pi)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return compute_bessel_math(d, self.r_max, self.freq)

class PolynomialEnvelope(nn.Module):
    def __init__(self, r_cut: float, p: int = 5):
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)
    
    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        return compute_envelope_math(d_ij, self.r_cutoff)

class GeometricBasis(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, vec_ij, d_ij):
        raw_rbf = self.rbf_mlp(self.rbf(d_ij.unsqueeze(-1)))
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        # ⚠️ r_hat 计算必须在 Python 层保留，确保梯度传导
        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6)
        
        basis = {}
        basis[0] = rbf_feat
        
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = rbf_feat.unsqueeze(1) * r_hat.unsqueeze(-1)
            
        if self.cfg.use_L2:
            basis[2] = compute_l2_basis(rbf_feat, r_hat)
            
        return basis, r_hat

class LeibnizCoupling(nn.Module): 
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.ModuleDict()
        
        for path_key, active in config.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2: continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1: continue
                
            name = f"{l_in}_{l_edge}_{l_out}_{path_key[3]}"
            self.path_weights[name] = nn.Linear(self.F, self.F, bias=False)

        self.inv_sqrt_f = self.F ** -0.5

    def forward(self, h_nodes: Dict[int, torch.Tensor], basis_edges: Dict[int, torch.Tensor], edge_index):
        src, _ = edge_index
        messages: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: []}
        
        for path_key, active in self.cfg.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, op_type = path_key
            
            if basis_edges.get(l_edge) is None: continue
            
            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights: continue
            
            if h_nodes.get(l_in) is None: continue 
            else: inp = h_nodes[l_in]
            
            h_src = inp[src]
            h_trans = self.path_weights[layer_name](h_src)
            geom = basis_edges[l_edge]
            res = None
            
            # --- Operation Logic ---
            if op_type == 'prod':
                if l_in == 0 and l_edge == 0: res = h_trans * geom
                elif l_in == 0 and l_edge == 1: res = h_trans.unsqueeze(1) * geom
                elif l_in == 0 and l_edge == 2: res = h_trans.unsqueeze(1).unsqueeze(1) * geom
                elif l_in == 1 and l_edge == 0: res = h_trans * geom.unsqueeze(1)
                elif l_in == 2 and l_edge == 0: res = h_trans * geom.unsqueeze(1).unsqueeze(1)
            elif op_type == 'dot':
                res = torch.sum(h_trans * geom, dim=1)
            elif op_type == 'cross':
                g = geom
                if g.dim() == 2: g = g.unsqueeze(-1)
                res = torch.linalg.cross(h_trans, g, dim=1)
            elif op_type == 'outer':
                outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
                trace = torch.einsum('eiif->ef', outer)
                eye = torch.eye(3, device=outer.device).view(1, 3, 3, 1)
                res = outer - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye
            elif op_type == 'mat_vec':
                res = torch.einsum('eijf, ejf -> eif', h_trans, geom)
            elif op_type == 'vec_mat':
                res = torch.einsum('eif, eijf -> ejf', h_trans, geom)
            elif op_type == 'double_dot':
                res = torch.sum(h_trans * geom, dim=(1, 2))
            elif op_type == 'mat_mul_sym':
                raw = torch.einsum('eikf, ekjf -> eijf', h_trans, geom)
                sym = 0.5 * (raw + raw.transpose(1, 2))
                trace = torch.einsum('eiif->ef', sym)
                eye = torch.eye(3, device=sym.device).view(1, 3, 3, 1)
                res = sym - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye

            if res is not None:
                messages[l_out].append(res * self.inv_sqrt_f)
                
        final_msgs: Dict[int, Optional[torch.Tensor]] = {}
        for l in [0, 1, 2]:
            final_msgs[l] = sum(messages[l]) if len(messages[l]) > 0 else None
        return final_msgs

class PhysicsGating(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        self.W_query = nn.Linear(self.F, self.F, bias=False)
        self.W_key = nn.Linear(self.F, self.F, bias=False)
        
        self.phys_bias_mlp = nn.Sequential(
            nn.Linear(3 * self.F, self.F), 
            nn.SiLU(),            
            nn.Linear(self.F, 3 * self.F) 
        )
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=False)
        self.gate_scale = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, msgs, h_node0, scalar_basis, r_hat, h_node1, edge_index, capture_weights=False):
        if not self.cfg.use_gating: return msgs
        
        src, dst = edge_index
        
        if h_node1 is not None:
            phys_input = compute_gating_projections(h_node1, r_hat, scalar_basis, src, dst)
            split_idx = scalar_basis.shape[-1]
            p_ij = phys_input[:, split_idx:]        
        else:
            p_ij = torch.zeros((scalar_basis.shape[0], 2 * self.F), device=scalar_basis.device)
            phys_input = torch.cat([scalar_basis, p_ij], dim=-1)

        q = self.W_query(h_node0[dst]) 
        k = self.W_key(h_node0[src])   
        chem_score = q * k             
        chem_logits = self.channel_mixer(chem_score)
        phys_logits = self.phys_bias_mlp(phys_input)
        
        raw_gates = chem_logits + phys_logits
        gates = torch.sigmoid(raw_gates) * self.gate_scale
        
        if capture_weights: self.scalar_basis_captured = scalar_basis.detach()
        if capture_weights: self.p_ij_captured = p_ij.detach()
        if capture_weights: self.chem_logits_captured = chem_logits.detach()
        if capture_weights: self.phys_logits_captured = phys_logits.detach()

        g_list = torch.split(gates, self.F, dim=-1)
        g0, g1, g2 = [g.contiguous() for g in g_list]

        if capture_weights: self.g0_captured = g0.detach()
        if capture_weights: self.g1_captured = g1.detach()
        if capture_weights: self.g2_captured = g2.detach()
        
        out_msgs: Dict[int, torch.Tensor] = {}
        if 0 in msgs and msgs[0] is not None: out_msgs[0] = msgs[0] * g0
        if 1 in msgs and msgs[1] is not None: out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if 2 in msgs and msgs[2] is not None: out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
            
        return out_msgs

class CartesianDensityBlock(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config
        
        in_dim = 0
        if config.use_L0: in_dim += self.F
        if config.use_L1: in_dim += self.F
        if config.use_L2: in_dim += self.F 
        
        self.scalar_update_mlp = nn.Sequential(
            nn.Linear(in_dim, self.F),
            nn.SiLU(),
            nn.Linear(self.F, self.F)
        )

        if config.use_L1: self.L1_linear = nn.Linear(self.F, self.F, bias=False)
        if config.use_L2: self.L2_linear = nn.Linear(self.F, self.F, bias=False)

        scale_out_dim = 0
        if config.use_L1: scale_out_dim += self.F
        if config.use_L2: scale_out_dim += self.F
        
        if scale_out_dim > 0:
            self.scale_mlp = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, scale_out_dim)
            )
        else:
            self.scale_mlp = None
  
        self.inv_sqrt_deg = 1.0 / (config.avg_neighborhood ** 0.5)

    def forward(self, msgs: Dict[int, torch.Tensor], index: torch.Tensor, num_nodes: int):
        # 1. 密度聚合
        # ✅ 修正：标准类型标注，明确 None
        densities: Dict[int, Optional[torch.Tensor]] = {}
        densities[0], densities[1], densities[2] = None, None, None

        for l in [0, 1, 2]:
            if l in msgs and msgs[l] is not None:
                agg = scatter_add(msgs[l], index, dim=0, dim_size=num_nodes)
                densities[l] = agg * self.inv_sqrt_deg 
            else:
                densities[l] = None

        # 2. 提取不变量
        concat = compute_invariants(densities[0], densities[1], densities[2])

        # 3. 标量更新
        # ✅ 修正：使用 index.device 避免歧义报错
        if concat.numel() > 0:
            delta_h0 = self.scalar_update_mlp(concat)
        else:
            delta_h0 = torch.zeros((num_nodes, self.F), device=index.device)

        # 4. 矢量更新
        delta_h1 = None
        delta_h2 = None

        if self.scale_mlp is not None:
            scales = self.scale_mlp(delta_h0)
            curr_dim = 0
            
            if self.cfg.use_L1 and densities[1] is not None:
                alpha1 = scales[:, curr_dim : curr_dim + self.F] 
                h1_mixed = self.L1_linear(densities[1])
                delta_h1 = h1_mixed * alpha1.unsqueeze(1)
                curr_dim += self.F
                
            if self.cfg.use_L2 and densities[2] is not None:
                alpha2 = scales[:, curr_dim : curr_dim + self.F]
                h2_mixed = self.L2_linear(densities[2])
                delta_h2 = h2_mixed * alpha2.unsqueeze(1).unsqueeze(1)

        return delta_h0, delta_h1, delta_h2

# ==========================================
# 6. 长程场 (Latent Long Range) - Ablation Ready
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 物理常数 (eV * A)
# k_e = 1 / (4 * pi * epsilon_0)
KE_CONST = 14.3996 

@torch.jit.script
def compute_direct_electrostatics_jit(
    q: torch.Tensor, 
    dist: torch.Tensor, 
    batch_mask: torch.Tensor,
    sigma: torch.Tensor,
    ke_const: float  # <--- 🔥 新增参数
) -> torch.Tensor:
    """
    【实空间求和】适用于有限体系或周期性体系的短程修正。
    """
    # 1. 电荷乘积矩阵 q_i * q_j
    qq = q @ q.t()  # (N, N)
    
    # 2. 倒距离 1/r
    inv_dist = 1.0 / (dist + 1e-8)
    
    # 3. 屏蔽因子 (Screening Factor)
    sqrt2 = 1.41421356
    scaled_r = dist / (sqrt2 * sigma)
    shielding = torch.erf(scaled_r)
    
    # 4. 组合能量
    E_matrix = qq * inv_dist * shielding
    
    # 5. 求和
    E_sum = torch.sum(E_matrix * batch_mask)
    
    # 乘以 0.5 (消除双重计数) 和 传入的库仑常数
    return 0.5 * ke_const * E_sum

@torch.jit.script
def compute_bj_damping_vdw_jit(
    c6: torch.Tensor,
    r_vdw: torch.Tensor,
    dist_sq: torch.Tensor,
    batch_mask: torch.Tensor
) -> torch.Tensor:
    """
    【范德华力】Becke-Johnson (BJ) 阻尼形式。
    """
    # 组合规则: 几何平均
    c6_ij = torch.sqrt(c6 @ c6.t())
    rvdw_ij = torch.sqrt(r_vdw @ r_vdw.t())
    
    # 计算 r^6
    dist6 = dist_sq ** 3
    
    # BJ 阻尼分母: r^6 + R_vdw^6
    damping = dist6 + (rvdw_ij ** 6)
    
    # 能量求和 (负号表示吸引)
    E_matrix = - (c6_ij / (damping + 1e-8)) * batch_mask
    return 0.5 * torch.sum(E_matrix)

@torch.jit.script
def generate_k_template(k_cutoff: float, device: torch.device) -> torch.Tensor:
    """
    生成一个通用的整数网格 (n1, n2, n3)。
    """
    n_max = 8 
    rng = torch.arange(-n_max, n_max + 1, device=device, dtype=torch.float32)
    n1, n2, n3 = torch.meshgrid(rng, rng, rng, indexing='ij')
    
    # (M, 3) 整数向量
    n = torch.stack([n1.flatten(), n2.flatten(), n3.flatten()], dim=1)
    
    # 剔除 (0,0,0)
    n_sq = torch.sum(n**2, dim=1)
    mask = n_sq > 0
    return n[mask]

@torch.jit.script
def compute_ewald_kspace_jit(
    q: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    n_grid: torch.Tensor,
    sigma: torch.Tensor,
    k_cutoff: float,
    num_graphs: int,
    ke_const: float
) -> torch.Tensor:
    """
    【倒空间 Ewald 求和】(Batch 维度对齐修复版)
    采用 Soft Masking 而非 Index Masking，确保所有 batch 的 K 维度一致 (4912)。
    """
    # 1. 构建倒格子向量 B = 2*pi * (A^-1)^T
    # cell: (B, 3, 3) -> recip_cell: (B, 3, 3)
    recip_cell = 2 * math.pi * torch.inverse(cell).transpose(1, 2)
    
    # 2. 生成物理 K 向量 (B, M, 3)
    # n_grid: (M, 3) -> k_vecs: (B, M, 3)
    # 这一步生成的 k_vecs 维度是固定的 (B, 4912, 3)
    k_vecs = torch.matmul(n_grid.unsqueeze(0), recip_cell) 
    
    # 3. 计算 K^2 (B, M)
    k_sq = torch.sum(k_vecs**2, dim=-1)
    
    # 🔥 [关键修复] 软截断 (Soft Mask)
    # 不删除元素，而是生成一个 0/1 掩码
    # 只有 |k| < k_cutoff 的项才参与计算
    mask_cutoff = (k_sq < k_cutoff**2).float() # (B, M)
    
    # 4. 计算结构因子 S(k)
    # 扩展 K 向量以匹配原子数: (B, M, 3) -> (N, M, 3)
    k_vecs_expanded = k_vecs[batch] 
    
    # 相位 k*r: (N, M)
    kr = torch.sum(k_vecs_expanded * pos.unsqueeze(1), dim=-1)
    
    cos_kr = torch.cos(kr)
    sin_kr = torch.sin(kr)
    
    # 初始化聚合容器，大小固定为 n_grid.size(0) 即 4912
    # 此时 cos_kr 的第二维度也是 4912，维度匹配，不会报错
    M = n_grid.size(0)
    Sk_real = torch.zeros((num_graphs, M), device=q.device, dtype=q.dtype)
    Sk_imag = torch.zeros((num_graphs, M), device=q.device, dtype=q.dtype)
    
    # index_add_ (N, M) -> (B, M)
    Sk_real.index_add_(0, batch, q * cos_kr)
    Sk_imag.index_add_(0, batch, q * sin_kr)
    
    # |S(k)|^2 (B, M)
    Sk_sq = Sk_real**2 + Sk_imag**2
    
    # 5. 计算能量权重 (Prefactor)
    # prefactor = exp(-sigma^2 * k^2 / 2) / k^2
    # 加上 1e-12 防止除零
    prefactor = torch.exp(-0.5 * sigma**2 * k_sq) / (k_sq + 1e-12)
    
    # 🔥 应用软截断：超范围的 K 点权重置为 0
    prefactor = prefactor * mask_cutoff
    
    # 6. 倒空间能量求和
    # 对 M 维度求和 -> (B,)
    E_recip_raw = torch.sum(prefactor * Sk_sq, dim=1)
    
    # 系数修正
    vol = torch.abs(torch.det(cell)) # (B,)
    coeff = (2 * math.pi * ke_const) / vol
    E_recip = coeff * E_recip_raw
    
    # 7. 减去自能 (Self Energy)
    q_sq = q**2
    q_sq_sum = torch.zeros((num_graphs, 1), device=q.device, dtype=q.dtype)
    q_sq_sum.index_add_(0, batch, q_sq)
    q_sq_sum = q_sq_sum.squeeze(-1)
    
    self_prefactor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    E_self = ke_const * self_prefactor * q_sq_sum
    
    return E_recip - E_self

# ==========================================
# 6. 长程场 (Latent Long Range) - 最终修复版
# ==========================================
class LatentLongRange(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        # 物理常数通过 self.KE 传递给 JIT，避免 closed over global 错误
        self.KE = KE_CONST 
        
        # --- 1. 物理参数预测层 ---
        if config.use_charge:
            # 输入: 标量特征 h0 -> 输出: 电荷 q
            self.q_proj = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 1, bias=False) # 无偏置
            )
            
            for layer in self.q_proj:
                if isinstance(layer, nn.Linear):
                    # 1. 权重初始化：用较大的 std (0.1 ~ 0.2)
                    nn.init.normal_(layer.weight, mean=0, std=0.2)
                    
                    # 2. 偏置初始化
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
            # 可学习的高斯宽度 sigma
                    
            print(">>> 警告: 已对电荷投影层进行暴力初始化 (std=0.2)，旨在打破零梯度陷阱。")
            self.sigma = nn.Parameter(torch.tensor(1.0))

        if config.use_vdw:
            # 输入: 标量特征 h0 -> 输出: C6系数, 范德华半径 R_vdw
            self.vdw_proj = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 2)
            )

        if config.use_dipole:
            # 输入: 矢量特征 h1 -> 输出: 偶极矩 mu
            self.mu_proj = nn.Linear(self.F, 1, bias=False)

        # 缓存: 整数网格模板
        self.register_buffer('n_grid_cache', None)

    def forward(self, h0, h1, pos, batch, cell: Optional[torch.Tensor] = None, capture_descriptors=False):
        """
        前向传播
        """
        total_energy = 0.0
        # ⚠️ 注意: 确保 batch 在 GPU 上
        num_graphs = int(batch.max().item()) + 1
        
        # ----------------------------------------------------
        # 1. 预测物理参数
        # ----------------------------------------------------
        q = None
        c6, r_vdw = None, None
        
        if self.cfg.use_charge:

            if batch[0] == 0: # 防止刷屏
                 # 1. 检查输入特征是否坍塌？(正常应该是 0.5 ~ 1.0)
                 print(f"[DEBUG] h0 stats: mean={h0.abs().mean().item():.5f}, std={h0.std().item():.5f}")
                 
                 # 2. 检查权重是否被覆盖？(正常应该是 0.16 左右，如果是 0.001 就是被覆盖了)
                 last_layer_weight = self.q_proj[-1].weight
                 print(f"[DEBUG] W stats: mean={last_layer_weight.abs().mean().item():.5f}, std={last_layer_weight.std().item():.5f}")
            q = self.q_proj(h0) # (N, 1)
            
            # [物理约束] 电荷中性化
            # 手动实现 batch mean (避免引入额外的 scatter_mean 函数依赖)
            q_sum = torch.zeros(num_graphs, 1, device=q.device, dtype=q.dtype)
            q_sum.index_add_(0, batch, q)
            
            ones = torch.ones_like(q)
            counts = torch.zeros(num_graphs, 1, device=q.device, dtype=q.dtype)
            counts.index_add_(0, batch, ones)
            
            q_mean = q_sum / counts.clamp(min=1.0)
            q = q - q_mean[batch]
            print(f"Charge q info: mean={q.abs().mean().item():.5f}, max={q.max().item():.5f}")
            if capture_descriptors:
                self.charge = q

        if self.cfg.use_vdw:
            vdw_params = self.vdw_proj(h0)
            c6 = F.softplus(vdw_params[:, 0:1])
            r_vdw = F.softplus(vdw_params[:, 1:2])

        # ----------------------------------------------------
        # 2. 分支 A: 周期性体系 (PBC) -> Ewald K-Space
        # ----------------------------------------------------
        is_periodic = False
        if cell is not None:
            # 计算行列式 det，取绝对值，看是否大于微小量
            # 也就是检查盒子是不是扁的或者全 0
            det = torch.abs(torch.linalg.det(cell))
            # 只要有一个 batch 的体积是正常的，我们就认为是周期性的
            # (注意：通常 batch 内所有图的性质应该一致)
            if (det > 1e-6).all():
                is_periodic = True

            # [静电力]
        if is_periodic:
            if self.n_grid_cache is None:
                self.n_grid_cache = generate_k_template(k_cutoff=6.0, device=pos.device)
            
            # 🔥 修复点：传入 self.KE
            e_elec_batch = compute_ewald_kspace_jit(
                q, pos, batch, cell, self.n_grid_cache, 
                self.sigma, k_cutoff=6.0, num_graphs=num_graphs,
                ke_const=self.KE  # <--- Pass as argument
            )
            total_energy += torch.sum(e_elec_batch)

        # ----------------------------------------------------
        # 3. 分支 B: 有限体系 (Cluster) -> Direct Sum
        # ----------------------------------------------------
        else:
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist_sq = torch.sum(diff**2, dim=-1)
            dist = torch.sqrt(dist_sq + 1e-8)
            
            batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0))
            diag_mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
            valid_mask = batch_mask & (~diag_mask)
            mask_float = valid_mask.float() 

            # [静电力]
            if self.cfg.use_charge and q is not None:
                # 🔥 修复点：传入 self.KE
                e_elec = compute_direct_electrostatics_jit(
                    q, dist, mask_float, self.sigma,
                    ke_const=self.KE # <--- Pass as argument
                )
                total_energy += e_elec
            
            # [范德华]
            if self.cfg.use_vdw and c6 is not None:
                e_vdw = compute_bj_damping_vdw_jit(
                    c6, r_vdw, dist_sq, mask_float
                )
                total_energy += e_vdw
                
            # [偶极矩] (可选，此处略去以保持简洁)

        return total_energy * self.cfg.long_range_scale