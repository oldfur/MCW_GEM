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

# ==============================================================================
# 🔥 核心数学内核 (JIT Script)
# 这些函数会被编译为 C++ 运行时，极大提升 For 循环和矩阵操作的速度
# ==============================================================================

@torch.jit.script
def compute_direct_electrostatics_jit(
    q: torch.Tensor, 
    dist: torch.Tensor, 
    batch_mask: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    【实空间求和】适用于有限体系(Cluster)或周期性体系的短程修正。
    
    对应公式: E = 1/2 * k_e * sum_{i,j} (q_i*q_j)/r * erf(r / (sqrt(2)*sigma))
    物理意义: 计算两个宽度为 sigma 的高斯电荷球之间的静电相互作用。
    """
    # 1. 电荷乘积矩阵 q_i * q_j
    qq = q @ q.t()  # (N, N)
    
    # 2. 倒距离 1/r (加 epsilon 防止除 0)
    inv_dist = 1.0 / (dist + 1e-8)
    
    # 3. 屏蔽因子 (Screening Factor): erf(r / (sqrt(2) * sigma))
    # 作用: 
    #   r -> inf: erf -> 1, 恢复标准库仑定律 1/r
    #   r -> 0:   erf/r -> const, 消除 r=0 处的无穷大奇点
    #   短程缺失的 (1-erf)/r 部分由 GNN 负责拟合 (erfc部分)
    sqrt2 = 1.41421356
    scaled_r = dist / (sqrt2 * sigma)
    shielding = torch.erf(scaled_r)
    
    # 4. 组合能量
    # E_matrix = (q_i * q_j / r) * erf(...)
    E_matrix = qq * inv_dist * shielding
    
    # 5. 求和
    # batch_mask: 确保不计算不同分子间的原子
    # diag_mask(外部处理): 确保不计算 i=j
    E_sum = torch.sum(E_matrix * batch_mask)
    
    # 乘以 0.5 (消除双重计数 i-j 和 j-i) 和 库仑常数
    return 0.5 * KE_CONST * E_sum

@torch.jit.script
def compute_bj_damping_vdw_jit(
    c6: torch.Tensor,
    r_vdw: torch.Tensor,
    dist_sq: torch.Tensor,
    batch_mask: torch.Tensor
) -> torch.Tensor:
    """
    【范德华力】Becke-Johnson (BJ) 阻尼形式。
    
    对应公式: E = - sum C6_ij / (r^6 + f(R_vdw)^6)
    物理意义: 模拟伦敦色散力，同时防止 r->0 时能量发散。
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
    用于构建 K 向量。
    """
    # 估计需要的整数范围。对于大多数晶胞，[-10, 10] 足够覆盖 k_cutoff < 6.0
    # 实际应用中可以动态计算，这里为了 JIT 效率设为固定范围
    n_max = 8 
    rng = torch.arange(-n_max, n_max + 1, device=device, dtype=torch.float32)
    n1, n2, n3 = torch.meshgrid(rng, rng, rng, indexing='ij')
    
    # (M, 3) 整数向量
    n = torch.stack([n1.flatten(), n2.flatten(), n3.flatten()], dim=1)
    
    # 剔除 (0,0,0)，因为 Ewald 求和不包含 k=0 项 (背景电荷中和)
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
    sigma: float,
    k_cutoff: float,
    num_graphs: int
) -> torch.Tensor:
    """
    【倒空间 Ewald 求和】适用于周期性体系 (PBC)。
    
    对应 JCTC 论文 Eq. (1) 和 (2):
    E_recip = 1/(2*eps*V) * sum_k [ exp(-sigma^2 k^2 / 2) / k^2 * |S(k)|^2 ]
    """
    # 1. 构建倒格子向量 B = 2*pi * (A^-1)^T
    # cell: (B, 3, 3) -> recip_cell: (B, 3, 3)
    recip_cell = 2 * math.pi * torch.inverse(cell).transpose(1, 2)
    
    # 2. 生成物理 K 向量: K = n @ B
    # n_grid: (M, 3)
    # recip_cell: (B, 3, 3)
    # 结果 k_vecs: (B, M, 3) - 每个 batch 有自己的一套 K 向量
    k_vecs = torch.matmul(n_grid.unsqueeze(0), recip_cell) 
    
    # 3. 过滤 K 向量 (|k| < k_cutoff)
    # 为了保持 batch 维度一致，这里采用 soft mask (乘以0) 或者只保留都在范围内的
    # 简单起见，我们计算所有 n_grid 对应的 k，然后通过权重衰减自然过滤大的 k
    k_sq = torch.sum(k_vecs**2, dim=-1) # (B, M)
    
    # 4. 计算结构因子 S(k) = sum_j q_j * exp(i * k * r_j)
    # 将 k_vecs 映射到每个原子: (B, M, 3) -> (N, M, 3)
    k_vecs_expanded = k_vecs[batch] 
    
    # 计算相角 k * r: (N, M, 3) * (N, 1, 3) -> sum -> (N, M)
    kr = torch.sum(k_vecs_expanded * pos.unsqueeze(1), dim=-1)
    
    # 欧拉公式
    cos_kr = torch.cos(kr)
    sin_kr = torch.sin(kr)
    
    # 按 Batch 聚合求和 S(k)
    # S_real[b, k] = sum_{i in b} q_i * cos(k*r_i)
    # 这是一个高效的 scatter 操作
    Sk_real = torch.zeros(num_graphs, n_grid.size(0), device=q.device, dtype=q.dtype)
    Sk_imag = torch.zeros(num_graphs, n_grid.size(0), device=q.device, dtype=q.dtype)
    
    Sk_real.index_add_(0, batch, q * cos_kr)
    Sk_imag.index_add_(0, batch, q * sin_kr)
    
    # 模方 |S(k)|^2: (B, M)
    Sk_sq = Sk_real**2 + Sk_imag**2
    
    # 5. 计算能量项
    # prefactor = exp(-sigma^2 * k^2 / 2) / k^2
    # 对于 k=0 或极小值，exp/k^2 会爆炸，但我们在 generate_k_template 已经剔除了 n=0
    prefactor = torch.exp(-0.5 * sigma**2 * k_sq) / (k_sq + 1e-12)
    
    # 硬截断: 如果 k^2 很大，prefactor 极小，数值上安全
    # 如果要严格截断:
    # mask = k_sq < k_cutoff**2
    # prefactor = prefactor * mask.float()
    
    # 倒空间能量: Sum_k (prefactor * Sk_sq) -> (B,)
    E_recip_raw = torch.sum(prefactor * Sk_sq, dim=1)
    
    # 6. 系数修正
    # 系数 = 1 / (2 * epsilon_0 * V)
    # 我们有 KE_CONST = 1 / (4 * pi * epsilon_0)
    # 所以 1 / (2 * epsilon_0) = 2 * pi * KE_CONST
    vol = torch.abs(torch.det(cell)) # (B,)
    coeff = (2 * math.pi * KE_CONST) / vol
    
    E_recip = coeff * E_recip_raw
    
    # 7. 减去自能 (Self Energy Correction)
    # 倒空间求和包含了 i=i 的高斯自作用，必须减去
    # E_self = k_e * (1 / (sqrt(2*pi)*sigma)) * sum(q^2)
    q_sq = q**2
    q_sq_sum = torch.zeros(num_graphs, 1, device=q.device, dtype=q.dtype)
    q_sq_sum.index_add_(0, batch, q_sq)
    q_sq_sum = q_sq_sum.squeeze(-1)
    
    self_prefactor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    E_self = KE_CONST * self_prefactor * q_sq_sum
    
    # 总长程能量 (GNN 负责实空间 erfc 部分)
    return E_recip - E_self


class LatentLongRange(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        # --- 1. 物理参数预测层 ---
        if config.use_charge:
            # 输入: 标量特征 h0 -> 输出: 电荷 q
            self.q_proj = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, 1, bias=False) # 无偏置，特征为0则电荷为0
            )
            # 可学习的高斯宽度 sigma (初始值 1.0 A)
            # 决定了实空间和倒空间的分界，以及 GNN 需要拟合的短程范围
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

        # 缓存: 整数网格模板 (避免每次生成)
        self.register_buffer('n_grid_cache', None)

    def forward(self, h0, h1, pos, batch, cell: Optional[torch.Tensor] = None):
        """
        前向传播
        Args:
            h0: (N, F) 标量特征
            h1: (N, 3, F) 矢量特征
            pos: (N, 3) 原子坐标
            batch: (N,) 批次索引
            cell: (B, 3, 3) 晶胞矩阵. 如果为 None, 则认为是有限体系(Cluster)。
        """
        total_energy = 0.0
        num_graphs = int(batch.max()) + 1
        
        # ----------------------------------------------------
        # 1. 预测物理参数 (Physics Parameters)
        # ----------------------------------------------------
        q = None
        c6, r_vdw = None, None
        
        if self.cfg.use_charge:
            q = self.q_proj(h0) # (N, 1)
            
            # [物理约束] 电荷中性化 (Charge Neutrality)
            # 算出每个 graph 的平均电荷，然后减去，确保 sum(q) = 0
            q_sum = torch.zeros(num_graphs, 1, device=q.device, dtype=q.dtype)
            q_sum.index_add_(0, batch, q)
            
            counts = torch.zeros(num_graphs, 1, device=q.device, dtype=q.dtype)
            ones = torch.ones_like(q)
            counts.index_add_(0, batch, ones)
            
            q_mean = q_sum / counts.clamp(min=1.0)
            q = q - q_mean[batch] # 广播减去均值

        if self.cfg.use_vdw:
            vdw_params = self.vdw_proj(h0)
            # 物理量必须为正，使用 Softplus
            c6 = F.softplus(vdw_params[:, 0:1])
            r_vdw = F.softplus(vdw_params[:, 1:2])

        # ----------------------------------------------------
        # 2. 分支 A: 周期性体系 (PBC) -> Ewald K-Space
        # ----------------------------------------------------
        if cell is not None:
            # 确保 cell 形状正确 (B, 3, 3)
            if cell.dim() == 2: cell = cell.unsqueeze(0)
            if cell.shape[0] != num_graphs: 
                cell = cell.expand(num_graphs, -1, -1)

            # [静电力]
            if self.cfg.use_charge and q is not None:
                # 懒加载生成整数网格模板
                if self.n_grid_cache is None:
                    self.n_grid_cache = generate_k_template(k_cutoff=6.0, device=pos.device)
                
                # 计算倒空间能量 + 减去自能
                # 注意: 实空间部分 (erfc) 由 GNN 拟合
                e_elec_batch = compute_ewald_kspace_jit(
                    q, pos, batch, cell, self.n_grid_cache, 
                    self.sigma, k_cutoff=6.0, num_graphs=num_graphs
                )
                total_energy += torch.sum(e_elec_batch)

            # [范德华]
            # PBC 下 VdW 长程部分(> cutoff) 贡献很小，通常由 GNN 隐式学习
            # 或者使用简单的解析积分修正 (Tail Correction)。
            # 为了效率，这里暂不显式计算 PBC VdW 长程。
            pass

        # ----------------------------------------------------
        # 3. 分支 B: 有限体系 (Cluster) -> Direct Sum
        # ----------------------------------------------------
        else:
            # 计算全连接距离矩阵 (O(N^2))
            # 优化: 仅计算坐标差和距离，避免不必要的中间变量
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist_sq = torch.sum(diff**2, dim=-1)
            dist = torch.sqrt(dist_sq + 1e-8)
            
            # Mask: 排除不同 batch 和 自相互作用
            batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0))
            diag_mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
            valid_mask = batch_mask & (~diag_mask)
            mask_float = valid_mask.float() # JIT 需要 float

            # [静电力] Direct Sum with erf Screening
            if self.cfg.use_charge and q is not None:
                e_elec = compute_direct_electrostatics_jit(
                    q, dist, mask_float, self.sigma
                )
                total_energy += e_elec
            
            # [范德华] Direct Sum with BJ Damping
            if self.cfg.use_vdw and c6 is not None:
                e_vdw = compute_bj_damping_vdw_jit(
                    c6, r_vdw, dist_sq, mask_float
                )
                total_energy += e_vdw
                
            # [偶极矩] (可选)
            if self.cfg.use_dipole and h1 is not None:
                # 这里的逻辑比较复杂，为了代码清晰度未放入 JIT，
                # 如果需要可以参考之前的回复将其 JIT 化
                mu = self.mu_proj(h1).squeeze(-1)
                # ... (同之前的实现)

        return total_energy * self.cfg.long_range_scale