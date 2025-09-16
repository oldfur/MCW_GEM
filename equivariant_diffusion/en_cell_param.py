import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# utils: from cell params -> cell matrix (orthogonalization)
# and minimum image vector for PBC
# -----------------------
def cell_params_to_matrix(lengths, angles_deg):
    """
    Convert cell params (l1,l2,l3,a1,a2,a3) to 3x3 cell matrix (row vectors or column vectors).
    We'll return a matrix C such that fractional -> cartesian: x_cart = frac @ C
    lengths: (..., 3)
    angles_deg: (..., 3) angles between (b,c)=a1, (a,c)=a2, (a,b)=a3  (common conventions vary)
    This implementation follows one conventional mapping:
      a = l1
      b = l2
      c = l3
      alpha = a1 (angle between b and c)
      beta  = a2 (angle between a and c)
      gamma = a3 (angle between a and b)
    Returns: cell matrix (..., 3, 3)
    """
    # convert to tensors
    l1, l2, l3 = lengths.unbind(-1)
    a1, a2, a3 = angles_deg.unbind(-1)
    alpha = torch.deg2rad(a1)
    beta  = torch.deg2rad(a2)
    gamma = torch.deg2rad(a3)

    cos_alpha = torch.cos(alpha)
    cos_beta  = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # components
    v_x = l1
    v_y = l2 * cos_gamma
    v_z = l3 * cos_beta

    # compute the third column elements
    y_x = 0.0
    y_y = l2 * sin_gamma
    z_x = 0.0
    z_y = l3 * ( (cos_alpha - cos_beta * cos_gamma) / (sin_gamma + 1e-12) )
    z_z = l3 * torch.sqrt( (1 - cos_beta**2) - ((cos_alpha - cos_beta * cos_gamma)**2 / (sin_gamma**2 + 1e-12)) + 1e-12 )

    # assemble (B,3,3)
    # we return matrix where rows are lattice vectors: [a_vector, b_vector, c_vector]
    a_vec = torch.stack([v_x, 0.0 * v_x, 0.0 * v_x], dim=-1)
    b_vec = torch.stack([v_y, y_y, 0.0 * v_x], dim=-1)
    c_vec = torch.stack([v_z, z_y, z_z], dim=-1)
    C = torch.stack([a_vec, b_vec, c_vec], dim=-2)  # (..., 3, 3)
    return C


def minimum_image_vectors(pos, cell_matrix, mask=None):
    """
    Compute pairwise minimum-image relative vectors r_ij (B, N, N, 3) using cell matrix.
    pos: (B, N, 3) in cartesian coordinates
    cell_matrix: (B, 3, 3) lattice vectors as rows (same batch)
    mask: (B, N) optional boolean mask for valid atoms
    Returns:
      rij: (B, N, N, 3) minimal image vector from i to j (cartesian)
      dij: (B, N, N) distances
    Note: This uses fractional coordinates & wraps to [-0.5, 0.5] to get minimum image.
    """
    B, N, _ = pos.shape
    # compute fractional coordinates: frac = pos @ inv(C)
    invC = torch.linalg.inv(cell_matrix)  # (B,3,3)
    frac = torch.einsum("bnd,bdm->bnm", pos, invC)  # (B, N, 3)

    # pairwise differences in fractional space
    frac_i = frac.unsqueeze(2)  # (B, N, 1, 3)
    frac_j = frac.unsqueeze(1)  # (B, 1, N, 3)
    d_frac = frac_j - frac_i    # (B, N, N, 3)

    # wrap to -0.5..0.5 (minimum image)
    d_frac_wrapped = d_frac - torch.round(d_frac)

    # back to cartesian
    rij = torch.einsum("bnm,bdm->bnmd", d_frac_wrapped, cell_matrix)  # (B, N, N, 3)
    # fix shape ordering
    rij = rij  # already (B,N,N,3) due to broadcasting

    dij = torch.linalg.norm(rij, dim=-1)  # (B,N,N)

    if mask is not None:
        m = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        dij = dij * m + (1.0 - m) * 1e6
        rij = rij * m.unsqueeze(-1)

    return rij, dij


# -----------------------
# EGNN Layer
# -----------------------
class EGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim=0, hidden_dim=64, act=nn.SiLU()):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1)  # scalar message
        )
        self.msg_to_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, node_dim)
        )
        # phi for position update (scalar -> scalar)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1)
        )

        # optional residual gating for node features
        self.node_update = nn.GRUCell(node_dim, node_dim)

    def forward(self, h, x, rij, dij, edge_attr=None, mask=None):
        """
        h: (B, N, node_dim)
        x: (B, N, 3)
        rij: (B, N, N, 3) vector i->j (cartesian)
        dij: (B, N, N) distance (non-negative)
        edge_attr: (B, N, N, edge_dim) optional
        mask: (B, N) boolean or 0/1
        """
        B, N, D = h.shape
        # prepare pairs
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        dij_in = dij.unsqueeze(-1)  # (B,N,N,1)

        # concat features
        if edge_attr is not None:
            edge_in = torch.cat([hi, hj, dij_in, edge_attr], dim=-1)
        else:
            edge_in = torch.cat([hi, hj, dij_in], dim=-1)

        # scalar messages m_ij
        m_ij = self.edge_mlp(edge_in).squeeze(-1)  # (B,N,N)

        # optionally mask off invalid pairs
        if mask is not None:
            pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).float()  # (B,N,N)
            m_ij = m_ij * pair_mask + (1.0 - pair_mask) * 0.0
        # position update: x_i <- x_i + sum_j ( rij * phi_x(m_ij) )
        phi = self.phi_x(m_ij.unsqueeze(-1)).squeeze(-1)  # (B,N,N)
        # broadcast and weight vectors
        delta_x = (rij * phi.unsqueeze(-1)).sum(dim=2)  # (B, N, 3)
        x = x + delta_x

        # aggregate messages for node update
        m_aggr = m_ij.unsqueeze(-1)  # (B,N,N,1)
        m_sum = (m_aggr * edge_in[..., :1]).sum(dim=2)  # crude usage; we can instead map m_ij->vec
        # Instead convert m_ij via msg_to_node
        m_vec = self.msg_to_node(m_ij.unsqueeze(-1))  # (B,N,N,node_dim)
        m_node = m_vec.sum(dim=2)  # (B,N,node_dim)

        # update node features via GRUCell
        h_new = self.node_update(m_node.view(-1, D), h.view(-1, D))
        h = h_new.view(B, N, D)

        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h, x


# -----------------------
# Full EGNN Model for crystal cell prediction
# -----------------------
class CrystalEGNN(nn.Module):
    def __init__(self,
                 one_hot_dim,
                 node_dim=128,
                 n_layers=4,
                 hidden_dim=128,
                 use_edge_attr=False):
        super().__init__()
        self.node_dim = node_dim
        self.input_embed = nn.Sequential(
            nn.Linear(one_hot_dim + 3, node_dim),  # optionally include coords too
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim=node_dim, edge_dim=(0 if not use_edge_attr else 4), hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        # pooling -> crystal MLP
        self.crystal_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 6)
        )

    def forward(self, pos, atom_type_onehot, mask=None, cell_matrix=None):
        """
        pos: (B,N,3) cartesian coordinates (training: built with true cell)
        atom_type_onehot: (B,N,one_hot_dim)
        mask: (B,N) 0/1
        cell_matrix: (B,3,3) optional; if provided used for minimum-image distance, else plain cdist
        returns: (B,6) predicted l1,l2,l3,a1,a2,a3
        """
        B, N, _ = pos.shape
        device = pos.device
        if mask is None:
            mask = torch.ones(B, N, device=device)

        # embed nodes (include coords optionally or not; here include coords to give position-aware init)
        inp = torch.cat([pos, atom_type_onehot], dim=-1)  # (B,N,3+one_hot)
        h = self.input_embed(inp)  # (B,N,node_dim)

        # compute rij, dij
        if cell_matrix is not None:
            rij, dij = minimum_image_vectors(pos, cell_matrix, mask=mask)
        else:
            # fallback dense cdist (no PBC)
            rij = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B,N,N,3)
            dij = torch.linalg.norm(rij, dim=-1)  # (B,N,N)
            dij = dij * (mask.unsqueeze(1) * mask.unsqueeze(2)) + (1.0 - (mask.unsqueeze(1) * mask.unsqueeze(2))) * 1e6
            rij = rij * (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)

        # EGNN layers
        x = pos.clone()
        for layer in self.layers:
            h, x = layer(h, x, rij, dij, edge_attr=None, mask=mask)

            # recompute rij/dij after moving positions (if we want position updates to affect subsequent layer)
            if cell_matrix is not None:
                rij, dij = minimum_image_vectors(x, cell_matrix, mask=mask)
            else:
                rij = x.unsqueeze(2) - x.unsqueeze(1)
                dij = torch.linalg.norm(rij, dim=-1)
                dij = dij * (mask.unsqueeze(1) * mask.unsqueeze(2)) + (1.0 - (mask.unsqueeze(1) * mask.unsqueeze(2))) * 1e6
                rij = rij * (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)

        # pool to crystal embedding (mean over valid atoms)
        mask_f = mask.unsqueeze(-1)
        h = h * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        crystal_feat = h.sum(dim=1) / denom  # (B, node_dim)
        out = self.crystal_mlp(crystal_feat)  # (B,6)
        lengths = F.softplus(out[:, :3])  # >0
        angles = torch.sigmoid(out[:, 3:]) * 180.0  # (0,180)
        pred = torch.cat([lengths, angles], dim=-1)
        return pred

if __name__ == "__main__":
    # simple test
    B, N = 4, 50
    one_hot_dim = 10
    pos = torch.randn(B, N, 3)
    atom_type = F.one_hot(torch.randint(0, one_hot_dim, (B, N)), num_classes=one_hot_dim).float()
    mask = (torch.rand(B, N) > 0.2).float()

    model = CrystalEGNN(one_hot_dim=one_hot_dim, node_dim=128, n_layers=4)
    # during training, compute true cell_matrix from ground-truth cell params:
    true_lengths = torch.rand(B, 3) * 10 + 2.0
    true_angles  = torch.rand(B, 3) * 60 + 60.0  # 60-120 deg
    cell_mat = cell_params_to_matrix(true_lengths, true_angles)

    pred = model(pos, atom_type.float(), mask=mask, cell_matrix=cell_mat)
    print(pred.shape)  # (B, 6)
    print(pred)
