import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ase.io import read
from Utils import scatter_add
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from Utils import HTGPConfig
# ==========================================

# class BesselBasis(nn.Module): # 贝塞尔基函数构建模块
#     def __init__(self, r_max, num_basis=8):
#         super().__init__()
#         self.r_max = float(r_max)
#         self.num_basis = num_basis
#         # 预计算频率 (buffer 会自动随 model.to("cuda") 移动)
#         self.register_buffer("freq", torch.arange(1, num_basis + 1).float() * np.pi)

#     def forward(self, d):
#         # d: [Edges, 1] (在 GPU 上)
#         d_scaled = d / self.r_max
#         # print(d.device)
#         prefactor = torch.sqrt(torch.tensor(2.0 / self.r_max, device=d.device))
#         # print(prefactor.device)
#         # print(self.freq.device)
#         # float * GPU_Tensor -> 自动正常运行
#         return prefactor * torch.sin(self.freq * d_scaled) / (d + 1e-6)

# 1. Bessel Math (JIT Engine)
@torch.jit.script
def compute_bessel_math(d: torch.Tensor, r_max: float, freq: torch.Tensor) -> torch.Tensor:
    """Bessel Basis 的纯数学计算部分"""
    d_scaled = d / r_max
    prefactor = (2.0 / r_max) ** 0.5
    return prefactor * torch.sin(freq * d_scaled) / (d + 1e-6)

# ==========================================
# 🚀 还原为普通的 nn.Module (调用上面的 JIT 函数)
# ==========================================

class BesselBasis(nn.Module): 
    # ❌ 删掉了 @torch.jit.script，变回普通类
    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.register_buffer("freq", torch.arange(1, num_basis + 1).float() * np.pi)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # ✅ 调用 JIT 函数，享受加速
        return compute_bessel_math(d, self.r_max, self.freq)


# # [NEW] 包络函数：保证 cutoff 处能量和力平滑衰减为 0
# class PolynomialEnvelope(nn.Module):
#     def __init__(self, r_cut, p=5):
#         super().__init__()
#         self.r_cutoff = r_cut
#         self.p = p # 这里p其实没用到，公式是固定的，为了兼容性保留即可
    
    
#     def forward(self, d_ij):
#         # 1. 归一化距离 x = d / r_cut
#         # 范围从 [0, r_cut] 映射到 [0, 1]
#         x = d_ij / self.r_cutoff
        
#         # 2. 截断保护
#         # 虽然邻居列表通常只包含 < r_cut 的原子，但为了数值安全，
#         # 必须确保 x 不超过 1，否则多项式会发散。
#         # 实际上，对于 x > 1 的部分，包络值应该严格为 0。
#         # 这里的 clamp 保证 x 停留在 1，代入公式结果为 0。
#         x = torch.clamp(x, min=0, max=1)
        
#         # 3. 计算多项式
#         # 1 - 10x^3 + 15x^4 - 6x^5
#         return 1 - 10 * x**3 + 15 * x**4 - 6 * x**

# 2. Envelope Math (JIT Engine)
@torch.jit.script
def compute_envelope_math(d: torch.Tensor, r_cut: float) -> torch.Tensor:
    """Envelope 的纯数学计算部分"""
    x = d / r_cut
    x = torch.clamp(x, min=0.0, max=1.0)
    return 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
# ==========================================
class PolynomialEnvelope(nn.Module):
    # ❌ 删掉了 @torch.jit.script
    def __init__(self, r_cut: float, p: int = 5):
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)
    
    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        # ✅ 调用 JIT 函数
        return compute_envelope_math(d_ij, self.r_cutoff)


@torch.jit.script
def compute_l2_basis(rbf_feat: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    # rbf_feat: (E, F)
    # r_hat: (E, 3)
    
    # 外积: (E, 3, 1) * (E, 1, 3) -> (E, 3, 3)
    outer = r_hat.unsqueeze(2) * r_hat.unsqueeze(1) 
    
    # 构造单位阵，注意使用 type_as 保持设备和类型一致
    eye = torch.eye(3, dtype=r_hat.dtype, device=r_hat.device).unsqueeze(0)
    
    # 去迹
    trace_less = outer - (1.0/3.0) * eye
    
    # 融合: (E, 1, 1, F) * (E, 3, 3, 1) -> (E, 3, 3, F)
    return rbf_feat.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1)

class GeometricBasis(nn.Module): # 几何基底构建模块
    """
    构建正交笛卡尔-厄米特基底 T^(L) 并与径向特征 R(d) 融合
    """
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.SiLU(), # 激活函数 x / (1 + exp(-x))
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, vec_ij, d_ij):
        # vec_ij: (E, 3), d_ij: (E,)
        raw_rbf = self.rbf_mlp(self.rbf(d_ij.unsqueeze(-1))) # (E, F)
        env = self.envelope(d_ij) # 包络
        rbf_feat = raw_rbf * env.unsqueeze(-1)  # (E, F)

        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6) # (E, 3)
        
        basis = {}
        
        # L=0: Scalar [R(d)]
        basis[0] = rbf_feat
        
        # L=1: Vector [R(d) * r_hat]
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = rbf_feat.unsqueeze(1) * r_hat.unsqueeze(-1) # (E, 3, F)
            
        # L=2: Tensor [R(d) * (r_hat x r_hat - I/3)]
        if self.cfg.use_L2:
            # outer = torch.bmm(r_hat.unsqueeze(2), r_hat.unsqueeze(1)) # (E, 3, 3)
            # eye = torch.eye(3, device=vec_ij.device).unsqueeze(0)
            # trace_less = outer - (1.0/3.0) * eye
            # basis[2] = rbf_feat.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1) # (E, 3, 3, F)
            basis[2] = compute_l2_basis(rbf_feat, r_hat) # (E, 3, 3, F)            
        return basis, r_hat

# ==========================================
# 3. 动力学引擎: 莱布尼茨耦合 (Leibniz Coupling)
# ==========================================
class LeibnizCoupling(nn.Module): # 消息传递模块
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.ModuleDict()
        
        # 动态注册需要的线性层
        for path_key, active in config.active_paths.items():
            if not active: continue
            # 检查 L2 开关
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2:
                continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1:
                continue
                
            name = f"{l_in}_{l_edge}_{l_out}_{path_key[3]}"
            self.path_weights[name] = nn.Linear(self.F, self.F, bias=False)

        # [修改点 1] 归一化常数，防止深层数值爆炸
        self.inv_sqrt_f = self.F ** -0.5

    def forward(self, h_nodes, basis_edges, edge_index):
        src, _ = edge_index
        messages = {0: [], 1: [], 2: []}
        
        for path_key, active in self.cfg.active_paths.items():
            if not active: continue
            
            l_in, l_edge, l_out, op_type = path_key
            
            # 安全性检查: 如果该层输入特征不存在 (如第一层没有 h1 h2)
            if basis_edges.get(l_edge) is None: continue
            
            # 1. 获取权重层 & 线性变换
            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights: continue # 被 Config 开关全局禁用
            
            if h_nodes.get(l_in) is None:
                # 构造全零张量，维度根据 l_in 决定
                # l=0: (N, F), l=1: (N, 3, F), l=2: (N, 3, 3, F)
                num_nodes = h_nodes[0].size(0) # 以 h0 为基准获取节点数
                shape = (num_nodes,) + ((3,) * l_in) + (self.F,)
                inp = torch.zeros(shape, device=h_nodes[0].device, dtype=h_nodes[0].dtype)
            else:
                inp = h_nodes[l_in]
            
            h_src = inp[src]
            h_trans = self.path_weights[layer_name](h_src) # Linear Transform
            geom = basis_edges[l_edge]
            
            res = None
            
            # === 运算核心逻辑 (Operation Kernels) ===
            
            # 1. 简单积 (Prod): Scalar scaling
            if op_type == 'prod':              
                # --- Case A: 标量 * 标量 -> 标量 ---
                if l_in == 0 and l_edge == 0: 
                    # (E, F) * (E, F) -> (E, F)
                    res = h_trans * geom
                
                # --- Case B: 标量 驱动 几何 ---
                # (0, 1, 1): s * v -> v (生成初始偶极)
                elif l_in == 0 and l_edge == 1: 
                    # (E, F) * (E, 3, F) -> (E, 3, F)
                    # 需要 unsqueeze h_trans
                    res = h_trans.unsqueeze(1) * geom
                
                # (0, 2, 2): s * t -> t (生成初始四极) [你新增的]
                elif l_in == 0 and l_edge == 2:
                    # (E, F) * (E, 3, 3, F) -> (E, 3, 3, F)
                    res = h_trans.unsqueeze(1).unsqueeze(1) * geom

                # --- Case C: 几何 驱动 标量 (径向缩放) ---
                # (1, 0, 1): v * s -> v (径向缩放) [你新增的]
                elif l_in == 1 and l_edge == 0:
                    # (E, 3, F) * (E, F) -> (E, 3, F)
                    # 需要 unsqueeze geom
                    res = h_trans * geom.unsqueeze(1)
                
                # (2, 0, 2): t * s -> t (径向缩放)
                elif l_in == 2 and l_edge == 0:
                    # (E, 3, 3, F) * (E, F) -> (E, 3, 3, F)
                    res = h_trans * geom.unsqueeze(1).unsqueeze(1)

            # 2. 点积 (Dot): Contraction -> L_out < L_in
            elif op_type == 'dot': # v . v -> s
                res = torch.sum(h_trans * geom, dim=1)

            # 3. 叉积 (Cross): Vector Cross Product -> L_out = L_in = 1
            elif op_type == 'cross': # v x v -> v (Chiral Interaction)
                g = geom
                if g.dim() == 2: # 如果 geom 是 (E, 3)
                     g = g.unsqueeze(-1) # (E, 3, 1) - 广播
                
                res = torch.linalg.cross(h_trans, g, dim=1)   

            # 4. 外积 (Outer): Product -> L_out > L_in
            elif op_type == 'outer': # v x v -> t (Traceless)
                outer = h_trans.unsqueeze(2) * geom.unsqueeze(1) # (E,3,1,F)*(E,1,3,F) -> (E,3,3,F)
                # 去迹 (Remove Trace)
                trace = torch.einsum('eiif->ef', outer)
                eye = torch.eye(3, device=outer.device).view(1, 3, 3, 1)
                res = outer - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye

            # 5. 矩阵-矢量 (Mat-Vec / Vec-Mat): L=2 & L=1 -> L=1
            elif op_type == 'mat_vec': # T . v -> v
                res = torch.einsum('eijf, ejf -> eif', h_trans, geom)
            elif op_type == 'vec_mat': # v . T -> v
                res = torch.einsum('eif, eijf -> ejf', h_trans, geom)
                
            # 6. 双点积 (Double Dot): L=2 : L=2 -> L=0
            elif op_type == 'double_dot': 
                res = torch.sum(h_trans * geom, dim=(1, 2))

            # 7. 对称矩阵乘 (Mat Mul Sym): L=2 x L=2 -> L=2
            elif op_type == 'mat_mul_sym':
                # Matrix Mul
                raw = torch.einsum('eikf, ekjf -> eijf', h_trans, geom)
                # Symmetrize
                sym = 0.5 * (raw + raw.transpose(1, 2))
                # Trace removal
                trace = torch.einsum('eiif->ef', sym)
                eye = torch.eye(3, device=sym.device).view(1, 3, 3, 1)
                res = sym - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye

            if res is not None:
                messages[l_out].append(res * self.inv_sqrt_f) # 应用归一化常数
                
        # 结果聚合 (Simple Summation)
        final_msgs = {}
        for l in [0, 1, 2]:
            final_msgs[l] = sum(messages[l]) if messages[l] else None
        return final_msgs


# 🔥 [JIT 函数] 计算物理投影 (fuse sum + mul + cat)
@torch.jit.script
def compute_gating_projections(h_node1: torch.Tensor, 
                               r_hat: torch.Tensor, 
                               scalar_basis: torch.Tensor,
                               src: torch.Tensor, 
                               dst: torch.Tensor) -> torch.Tensor:
    r_hat_uns = r_hat.unsqueeze(-1)
    # Project: (E, 3, F) * (E, 3, 1) -> sum -> (E, F)
    p_src = torch.sum(h_node1[src] * r_hat_uns, dim=1)
    p_dst = torch.sum(h_node1[dst] * r_hat_uns, dim=1)
    # Concat
    return torch.cat([scalar_basis, p_src, p_dst], dim=-1)

class PhysicsGating(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        # 1. Chemical Matching (保持不变)
        self.W_query = nn.Linear(self.F, self.F, bias=False)
        self.W_key = nn.Linear(self.F, self.F, bias=False)
        
        # 2. Physical Bias Encoder [升级]
        # 输入不再是 2F+1，而是 3F
        # Scalar_Basis(F) + Proj_Src(F) + Proj_Dst(F) = 3*F
        self.phys_bias_mlp = nn.Sequential(
            nn.Linear(3 * self.F, self.F), # 充分融合 距离 和 方向
            nn.SiLU(),            
            nn.Linear(self.F, 3 * self.F) 
        )
        
        # 3. Channel Mixer (保持不变)
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=False)
        
        # 4. Gating Scale (保持不变)
        self.gate_scale = nn.Parameter(torch.ones(1) * 2.0)

    # [注意] 参数列表变了：把 d_ij 换成了 scalar_basis (即 basis_edges[0])
    def forward(self, msgs, h_node0, scalar_basis, r_hat, h_node1, edge_index, capture_weights=False):
        if not self.cfg.use_gating: return msgs
        
        src, dst = edge_index
        
        # # --- A. Physical Geometry Features ---
        # if h_node1 is not None:
        #     # 投影计算 (保持不变)
        #     p_src = torch.sum(h_node1[src] * r_hat.unsqueeze(-1), dim=1, keepdim=False)
        #     p_dst = torch.sum(h_node1[dst] * r_hat.unsqueeze(-1), dim=1, keepdim=False)
        #     p_ij = torch.cat([p_src, p_dst], dim=-1) # (E, 2F)
        # else:
        #     p_ij = torch.zeros((scalar_basis.shape[0], 2 * self.F), device=scalar_basis.device)
            
        # # [关键升级]
        # # 使用 RBF 丰富过的 scalar_basis (E, F) 代替 d_ij (E, 1)
        # # 现在的 phys_input 包含了丰富的距离非线性信息
        # # Input: (E, F + 2F) = (E, 3F)
        # phys_input = torch.cat([scalar_basis, p_ij], dim=-1)

        # --- A. Physical Geometry Features ---
        if h_node1 is not None:
            # 🔥 调用 JIT 函数
            phys_input = compute_gating_projections(h_node1, r_hat, scalar_basis, src, dst)
        else:
            p_ij = torch.zeros((scalar_basis.shape[0], 2 * self.F), device=scalar_basis.device)
            phys_input = torch.cat([scalar_basis, p_ij], dim=-1)

        # --- B. Compute Gating Scores ---
        
        # 1. Chemical (保持不变)
        q = self.W_query(h_node0[dst]) 
        k = self.W_key(h_node0[src])   
        chem_score = q * k             
        chem_logits = self.channel_mixer(chem_score)
        
        # 2. Physical (现在更强了)
        phys_logits = self.phys_bias_mlp(phys_input)
        
        # 3. Fuse & Apply (保持不变)
        raw_gates = chem_logits + phys_logits
        gates = torch.sigmoid(raw_gates) * self.gate_scale
        
        if capture_weights: self.stored_attention = gates.detach()

        g_list = torch.split(gates, self.F, dim=-1)
        g0, g1, g2 = [g.contiguous() for g in g_list]
        
        out_msgs: Dict[int, torch.Tensor] = {}
        if msgs[0] is not None: out_msgs[0] = msgs[0] * g0
        if msgs[1] is not None: out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if msgs[2] is not None: out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
            
        return out_msgs



# ==========================================
# 2. 交互模块 (Interaction Block) - 部分 JIT 化
# ==========================================

# 🔥 [JIT 函数] 计算旋转不变量 (Safe Norm)
# 🔥 [修正版] JIT 函数
@torch.jit.script
def compute_invariants(den0: Optional[torch.Tensor], 
                       den1: Optional[torch.Tensor], 
                       den2: Optional[torch.Tensor]) -> torch.Tensor:
    
    # ✅ 修正后的写法 (Python 3 标准写法):
    invariants: List[torch.Tensor] = []
    
    # L=0
    if den0 is not None:
        invariants.append(den0)
        
    # L=1
    if den1 is not None:
        sq_sum = torch.sum(den1.pow(2), dim=1) 
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    # L=2
    if den2 is not None:
        sq_sum = torch.sum(den2.pow(2), dim=(1, 2))
        norm = torch.sqrt(sq_sum + 1e-8)
        invariants.append(norm)
        
    # Concat
    if len(invariants) > 0:
        return torch.cat(invariants, dim=-1)
    else:
        return torch.empty(0)

class CartesianDensityBlock(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config
        
        # ---------------------------------------------------------
        # 1. 维度计算
        # ---------------------------------------------------------
        in_dim = 0
        if config.use_L0: in_dim += self.F
        if config.use_L1: in_dim += self.F
        if config.use_L2: in_dim += self.F 
        
        # ---------------------------------------------------------
        # 2. 标量更新网络 (The "Brain") - 负责处理化学性质
        # ---------------------------------------------------------
        self.scalar_update_mlp = nn.Sequential(
            nn.Linear(in_dim, self.F),
            nn.SiLU(),
            nn.Linear(self.F, self.F)
        )

        # ---------------------------------------------------------
        # 3. [改进] 矢量/张量通道混合层 (Channel Mixing)
        # ---------------------------------------------------------
        # 注意：必须设置 bias=False 以保证旋转等变性！
        # 这允许特征 i 的电性质去修正特征 j 的磁性质。
        if config.use_L1:
            self.L1_linear = nn.Linear(self.F, self.F, bias=False)
        
        if config.use_L2:
            self.L2_linear = nn.Linear(self.F, self.F, bias=False)

        # ---------------------------------------------------------
        # 4. 矢量/张量缩放网络 (The "Valve")
        # ---------------------------------------------------------
        scale_out_dim = 0
        if config.use_L1: scale_out_dim += self.F
        if config.use_L2: scale_out_dim += self.F
        
        if scale_out_dim > 0:
            self.scale_mlp = nn.Sequential(
                nn.Linear(self.F, self.F),
                nn.SiLU(),
                nn.Linear(self.F, scale_out_dim) # 输出缩放系数 alpha
            )
        else:
            self.scale_mlp = None

        # 5. 数值稳定性常数
        self.inv_sqrt_deg = 1.0 / (50.0 ** 0.5)

    def forward(self, msgs, index, num_nodes):
        # ===========================
        # 1. 密度聚合 (Aggregation)
        # ===========================
        densities: Dict[int, Optional[torch.Tensor]] = {}
        for l in [0, 1, 2]:
            if msgs[l] is not None:
                agg = scatter_add(msgs[l], index, dim=0, dim_size=num_nodes)
                densities[l] = agg * self.inv_sqrt_deg 
            else:
                densities[l] = None

        # ===========================
        # 2. [改进] 提取旋转不变量 (Invariants)
        # ===========================
        # 使用 Safe Norm (sqrt(sum^2 + eps)) 代替原来的 pow(2)
        # 优势：线性梯度，防止小信号梯度消失，防止大信号梯度爆炸
        invariants = []
        
        # # --- L=0 Invariant ---
        # if densities[0] is not None:
        #     invariants.append(densities[0]) 
            
        # # --- L=1 Invariant ---
        # if densities[1] is not None:
        #     # densities[1]: (N, 3, F)
        #     sq_sum = torch.sum(densities[1].pow(2), dim=1) # (N, F)
        #     norm = torch.sqrt(sq_sum + 1e-8)               # Safe Sqrt
        #     invariants.append(norm)
            
        # # --- L=2 Invariant ---
        # if densities[2] is not None:
        #     # densities[2]: (N, 3, 3, F)
        #     sq_sum = torch.sum(densities[2].pow(2), dim=(1, 2)) # (N, F)
        #     norm = torch.sqrt(sq_sum + 1e-8)                    # Safe Sqrt
        #     invariants.append(norm)
        concat = compute_invariants(densities[0], densities[1], densities[2])
        # ===========================
        # 3. 计算标量更新 (Scalar Update)
        # ===========================
        if invariants:
            # concat = torch.cat(invariants, dim=-1)
            delta_h0 = self.scalar_update_mlp(concat) # (N, F)
        else:
            delta_h0 = torch.zeros((num_nodes, self.F), device=index.device)
        # ===========================
        # 4. [改进] 计算矢量/张量更新 (Gated Vector Update)
        # ===========================
        delta_h1 = None
        delta_h2 = None

        if self.scale_mlp is not None:
            # 用“大脑”思考出来的 delta_h0 来决定“肢体”动作的幅度
            scales = self.scale_mlp(delta_h0) # (N, F_L1 + F_L2)
            
            curr_dim = 0
            
            # --- L=1 Update ---
            if self.cfg.use_L1 and densities[1] is not None:
                # 1. 获取门控系数
                alpha1 = scales[:, curr_dim : curr_dim + self.F] 
                
                # 2. [关键改进] 线性特征混合
                # densities[1]: (N, 3, F) -> Linear -> (N, 3, F)
                # Linear 只作用于 F 维度，不破坏空间结构
                h1_mixed = self.L1_linear(densities[1])
                
                # 3. 应用门控 (Gating)
                # (N, 3, F) * (N, 1, F)
                delta_h1 = h1_mixed * alpha1.unsqueeze(1)
                
                curr_dim += self.F
                
            # --- L=2 Update ---
            if self.cfg.use_L2 and densities[2] is not None:
                # 1. 获取门控系数
                alpha2 = scales[:, curr_dim : curr_dim + self.F]
                
                # 2. [关键改进] 线性特征混合
                h2_mixed = self.L2_linear(densities[2])
                
                # 3. 应用门控
                # (N, 3, 3, F) * (N, 1, 1, F)
                delta_h2 = h2_mixed * alpha2.unsqueeze(1).unsqueeze(1)

        return delta_h0, delta_h1, delta_h2

# ==========================================
# 6. 长程场 (Latent Long Range) - Ablation Ready
# ==========================================
class LatentLongRange(nn.Module):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        
        # --- 模块 1: 隐式电荷 (Charge) ---
        # 从标量特征 h0 预测电荷 q (N, 1)
        if config.use_charge:
            self.q_proj = nn.Sequential(
                nn.Linear(self.F, self.F // 2),
                nn.SiLU(),
                nn.Linear(self.F // 2, 1)
            )
            
        # --- 模块 2: 隐式范德华 (Van der Waals) ---
        # 从标量特征 h0 预测色散系数 C6 (N, 1)
        if config.use_vdw:
            self.c6_proj = nn.Sequential(
                nn.Linear(self.F, self.F // 2),
                nn.SiLU(),
                nn.Linear(self.F // 2, 1)
            )
            
        # --- 模块 3: 隐式偶极 (Dipole) ---
        # 从矢量特征 h1 预测偶极矩 mu (N, 3)
        if config.use_dipole:
            # 输入是 (N, 3, F)，我们需要对 F 维度做线性变换
            self.mu_proj = nn.Linear(self.F, 1, bias=False)

    def forward(self, h0, h1, pos, batch):
        """
        h0: (N, F) 标量特征
        h1: (N, 3, F) 矢量特征
        pos: (N, 3) 坐标
        batch: (N,) 批次索引
        """
        energy_total = 0.0
        
        # 1. 构建全连接距离矩阵 (Full Pairwise Distance)
        # 对于 < 3000 原子的体系，这比构建邻居表更快且更准
        # mask 处理不同 batch 之间的无效连接
        batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0)) # (N, N)
        
        diff = pos.unsqueeze(1) - pos.unsqueeze(0) # (N, N, 3)
        # 加一个 epsilon 防止除零
        dist_sq = torch.sum(diff**2, dim=-1) # (N, N)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        # 排除自相互作用 (对角线)
        diag_mask = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        valid_mask = batch_mask & (~diag_mask)
        
        # ==========================================
        # Ablation 1: 电荷守恒库仑作用 (Coulomb)
        # ==========================================
        if self.cfg.use_charge:
            q = self.q_proj(h0) # (N, 1)
            
            # [关键] 强制电荷中性 (Charge Neutrality)
            # 每个 batch 内的电荷和必须为 0
            # 使用 scatter_add 计算每个 batch 的总电荷
            from torch_scatter import scatter_add, scatter_mean
            batch_q_mean = scatter_mean(q, batch, dim=0) # (B, 1)
            q = q - batch_q_mean[batch] # 中心化
            
            # 计算能量: E = q_i * q_j / r
            # switch: 使用 damping function 避免短程奇异性，并平滑过渡 GNN
            # 这里使用简单的 soft-core: 1 / sqrt(r^2 + 1) 或者 damping
            # 简单起见，假设 GNN 处理了短程，我们只加长程
            
            qq = q @ q.t() # (N, N)
            # Taper function: 让长程力在短程 (比如 < 4A) 慢慢消失
            # f_taper = 1 - exp(-a * r)
            f_taper = 1.0 - torch.exp(-0.5 * dist) 
            
            E_coul = torch.sum(qq / dist * f_taper * valid_mask)
            energy_total += 0.5 * E_coul * 14.399 # 14.399 是 eV*A 的转换系数
            
        # ==========================================
        # Ablation 2: 范德华色散 (Dispersion)
        # ==========================================
        if self.cfg.use_vdw:
            # C6 必须为正数，使用 Softplus
            c6 = F.softplus(self.c6_proj(h0)) # (N, 1) softplus 为 ln(1 + exp(x)) 
            
            # 组合规则: C6_ij = sqrt(C6_i * C6_j)
            c6_ij = torch.sqrt(c6 @ c6.t())
            
            # E_vdw = - C6_ij / (r^6 + r_vdw^6)
            # 防止 r->0 时爆炸
            r6 = dist_sq ** 3
            damp_r6 = 20.0 # 经验值，或者设为可学习参数
            
            E_vdw = -torch.sum(c6_ij / (r6 + damp_r6) * valid_mask)
            energy_total += 0.5 * E_vdw

        # ==========================================
        # Ablation 3: 偶极-偶极 (Dipole-Dipole)
        # ==========================================
        if self.cfg.use_dipole and h1 is not None:
            # h1: (N, 3, F) -> projection -> (N, 3, 1) -> (N, 3)
            mu = self.mu_proj(h1).squeeze(-1)
            
            # E_dip = (mu_i . mu_j) / r^3 - 3 (mu_i . r)(mu_j . r) / r^5
            
            # 1. mu_i . mu_j
            mu_dot_mu = mu @ mu.t() # (N, N)
            
            # 2. 方向向量 n_ij = r_ij / r
            # 这部分计算比较费显存，如果你显存不够，可以先只跑 Charge 和 VdW
            n_ij = diff / (dist.unsqueeze(-1) + 1e-8) # (N, N, 3)
            
            # mu . n
            # (N, 1, 3) * (N, N, 3) -> (N, N)
            mu_dot_n_i = torch.sum(mu.unsqueeze(1) * n_ij, dim=-1)
            mu_dot_n_j = torch.sum(mu.unsqueeze(0) * n_ij, dim=-1) # 注意 n_ji = -n_ij
            
            term1 = mu_dot_mu # (N, N)
            # 注意符号：n_ji = -n_ij，所以第二项实际上是 +3? 
            # 标准公式: (mu1.mu2)/r3 - 3(mu1.n)(mu2.n)/r3
            term2 = -3 * mu_dot_n_i * mu_dot_n_j # 这里 n_ij 是反对称的，要注意
            # 这里简单起见，我们用绝对的距离向量计算，物理上会更严谨
            
            # 简化的 damping 1/r^3
            inv_r3 = 1.0 / (dist_sq * dist + 10.0) # damping
            
            E_dip = torch.sum((term1 + term2) * inv_r3 * valid_mask)
            energy_total += 0.5 * E_dip

        return energy_total * self.cfg.long_range_scale