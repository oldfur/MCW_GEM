import torch
from typing import List, Dict

def get_pbc_distance(x1, x2, lattice):
    """
    计算分数坐标下的周期性距离 (Minimum Image Convention)
    x1: (B, N, 3)
    x2: (B, N, 3)
    输出: (B, N, N) 距离矩阵
    """
    diff = x1.unsqueeze(2) - x2.unsqueeze(1) # (B, N, N, 3)
    diff = diff - torch.round(diff) # 处理周期性边界 [-0.5, 0.5], (B,N,N,3)
    G = lattice @ lattice.transpose(-1, -2) # (B,3,3)
    dist2 = torch.einsum("bnmd,bdk,bnmk->bnm", diff, G, diff)

    # 数值安全：避免极小负数（浮点误差）
    return torch.clamp(dist2, min=0.0)


def symmetry_guidance_gradient(coords: torch.Tensor,
                               lattice: torch.Tensor,
                               sym_ops: List[Dict[str, torch.Tensor]],
                               scale: float = 10.0,
                               num_ops_sample: int = 1,
                               bidirectional: bool = True) -> torch.Tensor:
    """
    coords: (B, N, 3) fractional coords, requires_grad=True
    lattice: (B, 3, 3) cell matrix (r = s @ lattice)
    sym_ops: list of dict [{'R': (3,3), 't': (3,)}] in fractional convention
    scale: guidance strength
    num_ops_sample: sample K ops per call and average loss (reduce variance)
    bidirectional: use symmetric Chamfer (A->B and B->A)
    """
    assert coords.requires_grad, "coords must have requires_grad=True"
    device = coords.device
    dtype = coords.dtype

    B, N, _ = coords.shape
    K = max(1, int(num_ops_sample))

    # --- sample K symmetry operations (Python list -> stack tensors) ---
    # Use .item() so list indexing works.
    idx = torch.randint(0, len(sym_ops), (K,), device=device)
    R_list = []
    t_list = []
    for k in range(K):
        op = sym_ops[int(idx[k].item())]
        Rk = op["R"]
        tk = op["t"]
        # ensure tensor, device, dtype
        if not torch.is_tensor(Rk):
            Rk = torch.tensor(Rk)
        if not torch.is_tensor(tk):
            tk = torch.tensor(tk)
        R_list.append(Rk.to(device=device, dtype=dtype))
        t_list.append(tk.to(device=device, dtype=dtype))

    R = torch.stack(R_list, dim=0)  # (K,3,3)
    t = torch.stack(t_list, dim=0)  # (K,3)

    # --- apply symmetry op(s): coords' = coords @ R^T + t ---
    # coords: (B,N,3), R^T: (K,3,3) -> broadcast to (B,K,N,3)
    coords_exp = coords.unsqueeze(1)  # (B,1,N,3)
    coords_trans = torch.matmul(coords_exp, R.transpose(-1, -2)) + t.view(1, K, 1, 3)
    coords_trans = coords_trans.remainder(1.0)  # keep in [0,1)

    # --- compute Chamfer loss under true Cartesian metric (non-orthogonal ok) ---
    # We'll compute loss for each k and average across K.
    losses = []
    for k in range(K):
        y = coords_trans[:, k, :, :]  # (B,N,3)

        d2_xy = get_pbc_distance(coords, y, lattice)  # (B,N,N)
        min_xy, _ = torch.min(d2_xy, dim=2)  # (B,N)

        if bidirectional:
            d2_yx = get_pbc_distance(y, coords, lattice)  # (B,N,N)
            min_yx, _ = torch.min(d2_yx, dim=2)  # (B,N)
            loss_k = 0.5 * (min_xy.mean() + min_yx.mean())
        else:
            loss_k = min_xy.mean()

        losses.append(loss_k)

    loss = torch.stack(losses).mean()

    # --- gradient ---
    grad = torch.autograd.grad(loss, coords, create_graph=False, retain_graph=False)[0]
    return grad * scale
