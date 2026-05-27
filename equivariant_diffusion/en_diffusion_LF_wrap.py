from equivariant_diffusion import utils
import json
import numpy as np
import math
import os
import torch
from egnn import Equiformer_dynamics
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.mlp import DiffusionMLP
from torch_scatter import scatter_mean
from crystalgrw.data_utils import lattice_params_from_matrix, lattice_params_to_matrix_torch
from equivariant_diffusion.mlp import DiffusionMLP
from tqdm import tqdm
import random
from mp20.crystal import smact_validity
from collections import Counter
import itertools
from guidance.symmetry_guidance import symmetry_guidance_gradient
from mp20.atom_type_mapping import ATOM_TYPE_SOFTMAX_DIM, class_index_to_symbol


DEFAULT_ATOM_TYPE_REPAIR_TOPK = 4

def composition_from_elem_idx(elem_idx, node_mask):
    """
    elem_idx: [N]  (int)
    node_mask: [N,1]
    return:
        elems: tuple[int]
        comps: tuple[int]   # gcd-normalized stoichiometry
    """
    mask = node_mask.squeeze(-1).bool()
    elems_list = elem_idx[mask].tolist()

    counter = Counter(elems_list)
    elems = tuple(sorted(counter.keys()))
    counts = np.array([counter[e] for e in elems], dtype=np.int64)

    # gcd normalization（与你现有代码完全一致）
    gcd = np.gcd.reduce(counts)
    counts = counts // gcd

    return elems, tuple(counts.tolist())

def select_candidate_atoms(logits, node_mask, max_atoms=6):
    """
    logits: [N, C]
    node_mask: [N,1]
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [N]

    entropy = entropy.masked_fill(~node_mask.squeeze(-1).bool(), -1e9)

    k = min(max_atoms, int(node_mask.sum().item()))
    _, idx = torch.topk(entropy, k)
    return idx.tolist()


def repair_composition_single(
    logits,             # [N,C]
    node_mask,          # [N,1]
    smact_validity_fn,
    topk=4,
    max_replace_atoms=2
):
    """
    返回:
        elem_idx: [N]
        repaired: bool
    """

    N, C = logits.shape
    device = logits.device

    # padding 不可选
    logits = logits.clone()
    logits[:, 0] = -1e9

    # 原始 argmax
    elem_idx = torch.argmax(logits, dim=-1) # [N]

    comp, count = composition_from_elem_idx(elem_idx, node_mask)
    if smact_validity_fn(comp, count):
        return elem_idx, False   # 不需要 repair

    # Top-k 候选
    topk_vals, topk_idx = torch.topk(logits, topk, dim=-1)  # [N, K]

    # 选择可替换原子
    candidate_atoms = select_candidate_atoms(
        logits, node_mask, max_atoms=6
    )

    best_solution = None

    for r in range(1, max_replace_atoms + 1):
        for atom_subset in itertools.combinations(candidate_atoms, r):
            # 每个 atom 有 topk 个候选
            choices = [topk_idx[a].tolist() for a in atom_subset]

            for replacement in itertools.product(*choices):
                trial_elem_idx = elem_idx.clone()
                for a, e in zip(atom_subset, replacement):
                    trial_elem_idx[a] = e

                comp, count = composition_from_elem_idx(
                    trial_elem_idx, node_mask
                )

                if smact_validity_fn(comp, count):
                    return trial_elem_idx, True

    # repair 失败，返回原始结果
    return elem_idx, False


def repair_composition_batch(
    logits,          # [B, N, C]
    node_mask,       # [B, N, 1]
    smact_validity_fn,
    topk=4,
    max_replace_atoms=2
):
    B, N, C = logits.shape
    final_elem_idx = []
    repaired_flags = []

    for b in range(B):
        elem_idx, repaired = repair_composition_single(
            logits[b],
            node_mask[b],
            smact_validity_fn,
            topk=topk,
            max_replace_atoms=max_replace_atoms
        )
        final_elem_idx.append(elem_idx)
        repaired_flags.append(repaired)

    final_elem_idx = torch.stack(final_elem_idx, dim=0)
    return final_elem_idx, repaired_flags


def is_valid_cell_params(rl, ra, eps=1e-6):
    """
    rl: [3] = (a, b, c)
    ra: [3] = (alpha, beta, gamma) in degrees
    """
    a, b, c = rl
    alpha, beta, gamma = ra
    # 边长必须 > 0
    if (a <= eps) or (b <= eps) or (c <= eps):
        return False
    # 转弧度
    alpha_r = torch.deg2rad(alpha)
    beta_r  = torch.deg2rad(beta)
    gamma_r = torch.deg2rad(gamma)
    # Gram determinant / volume^2 condition
    G = (
        1 + 2 * torch.cos(alpha_r) * torch.cos(beta_r) * torch.cos(gamma_r)
        - torch.cos(alpha_r)**2 - torch.cos(beta_r)**2 - torch.cos(gamma_r)**2
    )
    # 必须 > 0
    if G <= eps:
        return False
    return True

def batch_valid_mask(rl, ra):
    B = rl.shape[0]
    mask = torch.zeros(B, dtype=torch.bool, device=rl.device)
    for i in range(B):
        mask[i] = is_valid_cell_params(rl[i], ra[i])
    return mask

def zbl_force_mag(r, Z1Z2, Z_i=None, Z_j=None, e2_4pie0=14.3996, a0=0.529):
    """
    r: [B,N,N] distances in Å
    Z1Z2: [B,N,N] product of atomic numbers (float or int)
    Z_i: optional [B,N] atomic numbers for i
    Z_j: optional [B,N] atomic numbers for j
    returns: F_mag [B,N,N] (force magnitude, positive outward) in eV/Å
    """
    eps = 1e-12
    r_safe = r.clamp(min=1e-8)

    # prepare Zi, Zj to compute screening length a
    if (Z_i is None) or (Z_j is None):
        # fallback: infer approximate Z by sqrt(Z1Z2)
        Zi = torch.sqrt(Z1Z2.clamp(min=1.0))
        Zj = Zi
    else:
        Zi = Z_i[:, :, None].float() + eps   # [B,N,1]
        Zj = Z_j[:, None, :].float() + eps   # [B,1,N]

    a = 0.8854 * a0 / (Zi.pow(0.23) + Zj.pow(0.23) + eps)   # [B,N,N]
    x = r_safe / (a + eps)

    # screening function and derivative
    phi = (
        0.1818 * torch.exp(-3.2 * x)
        + 0.5099 * torch.exp(-0.9423 * x)
        + 0.2802 * torch.exp(-0.4029 * x)
        + 0.02817 * torch.exp(-0.2016 * x)
    )
    dphi = (
        -3.2 * 0.1818 * torch.exp(-3.2 * x)
        -0.9423 * 0.5099 * torch.exp(-0.9423 * x)
        -0.4029 * 0.2802 * torch.exp(-0.4029 * x)
        -0.2016 * 0.02817 * torch.exp(-0.2016 * x)
    )

    common = (Z1Z2.float() * e2_4pie0) / (r_safe * r_safe + eps)   # [B,N,N]
    dVdr = -common * (phi + x * dphi)   # dV/dr in eV/Å
    F = -dVdr                           # force magnitude (positive)
    F = torch.clamp(F, max=1e6)         # avoid blow-ups
    return F


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions."""
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)

    if q_sigma.dim() == 0:
        q_sigma = q_sigma.unsqueeze(0)

    if p_sigma.dim() == 0:
        p_sigma = p_sigma.unsqueeze(0)

    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d


def wrap_at_boundary(x: torch.Tensor, wrapping_boundary: float) -> torch.Tensor:
    """Wrap x at the boundary given by wrapping_boundary.
    Args:
      x: tensor of shape (batch_size, dim)
      wrapping_boundary: float): wrap at [0, wrapping_boundary] in all dimensions.
    Returns:
      wrapped_x: tensor of shape (batch_size, dim)
    """
    return torch.remainder(
        x, wrapping_boundary
    )  # remainder is the same as mod, but works with negative numbers.


def get_pbc_offsets(pbc: torch.Tensor, max_offset_integer: int = 3) -> torch.Tensor:
    """Build the Cartesian product of integer offsets of the periodic boundary. That is, if dim=3 and max_offset_integer=1 we build the (2*1 + 1)^3 = 27
       possible combinations of the Cartesian product of (i,j,k) for i,j,k in -max_offset_integer, ..., max_offset_integer. Then, we construct
       the tensor of integer offsets of the pbc vectors, i.e., L_{ijk} = row_stack([i * l_1, j * l_2, k * l_3]).

    Args:
        pbc (torch.Tensor, [batch_size, dim, dim]): The input pbc matrix.
        max_offset_integer (int): The maximum integer offset per dimension to consider for the Cartesian product. Defaults to 3.

    Returns:
        torch.Tensor, [batch_size, (2 * max_offset_integer + 1)^dim, dim]: The tensor containing the integer offsets of the pbc vectors.
    """
    offset_range = torch.arange(-max_offset_integer, max_offset_integer + 1, device=pbc.device)
    meshgrid = torch.stack(
        torch.meshgrid(offset_range, offset_range, offset_range, indexing="xy"), dim=-1
    )
    offset = (pbc[:, None, None, None] * meshgrid[None, :, :, :, :, None]).sum(-2)
    pbc_offset_per_molecule = offset.reshape(pbc.shape[0], -1, 3)
    return pbc_offset_per_molecule


def lattice_volume(lengths, angles):
    # lengths: [B,3], angles: [B,3] in radians
    angles = torch.deg2rad(angles)  # 转为弧度
    l1, l2, l3 = lengths[:,0], lengths[:,1], lengths[:,2]
    alpha, beta, gamma = angles[:,0], angles[:,1], angles[:,2]

    return l1 * l2 * l3 * torch.sqrt(
        1 + 2*torch.cos(alpha)*torch.cos(beta)*torch.cos(gamma)
        - torch.cos(alpha)**2 - torch.cos(beta)**2 - torch.cos(gamma)**2
    )  # [B]


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def ve_loglinear_gamma_schedule(timesteps: int, sigma_min: float, sigma_max: float):
    """
    VE-native schedule:
        sigma(t) = sigma_min * (sigma_max/sigma_min)^t
        gamma(t) = log sigma(t)^2
    returns:
        gamma: numpy array of shape [T+1]
    """
    assert sigma_min > 0 and sigma_max > 0 and sigma_max > sigma_min
    T = timesteps
    t = np.linspace(0.0, 1.0, T + 1, dtype=np.float64)  # [T+1]
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    gamma = 2.0 * np.log(sigma)  # log(sigma^2)
    return gamma.astype(np.float32)


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision, print_info=True):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)
        
        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2


        if print_info:
            print("Predefined noise schedule:")
            # 设置打印选项，只显示头部和尾部
            np.set_printoptions(threshold=10, edgeitems=10)
            print('alphas2', alphas2)
            print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
    

class PredefinedNoiseSchedule_ve(torch.nn.Module):
    """
    VE-native predefined schedule:
        gamma(t) = log sigma(t)^2   (NOT log(sigmas^2/alphas^2))
    """
    def __init__(
        self,
        noise_schedule: str,
        timesteps: int,
        sigma_min: float = 1e-2,
        sigma_max: float = 20.0,
        print_info: bool = True,
    ):
        super().__init__()
        self.timesteps = int(timesteps)

        if noise_schedule in ["ve_loglinear", "ve", "ve_native"]:
            gamma = ve_loglinear_gamma_schedule(self.timesteps, sigma_min, sigma_max)  # [T+1]
        else:
            raise ValueError(f"Unknown VE-native noise_schedule: {noise_schedule}")

        # store as buffer (not a Parameter)
        self.register_buffer("gamma", torch.from_numpy(gamma).float())  # [T+1]

        if print_info:
            with np.printoptions(threshold=10, edgeitems=6):
                print("VE-native noise schedule:")
                print("gamma(head/tail):", gamma)
            # quick sanity: sigma range
            sigma0 = float(np.exp(0.5 * gamma[0]))
            sigma1 = float(np.exp(0.5 * gamma[-1]))
            print(f"sigma(t=0)={sigma0:.6g}, sigma(t=1)={sigma1:.6g}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Continuous lookup with linear interpolation.
        t: [B,1] (or any shape) in [0,1]
        returns gamma(t): same shape as t
        """
        t = t.clamp(0.0, 1.0)
        T = self.timesteps
        # table length is T+1, valid indices [0..T]
        u = t * T  # map [0,1] -> [0,T]
        i0 = torch.floor(u).long()
        i0 = torch.clamp(i0, 0, T - 1)    # so i1 won't exceed T
        i1 = i0 + 1                       # max T
        w = (u - i0.float()).clamp(0.0, 1.0)

        g0 = self.gamma[i0]
        g1 = self.gamma[i1]
        return g0 + (g1 - g0) * w


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):   
    # 计算标准正态分布的累积分布函数（CDF）
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EquiTransVariationalDiffusion_LF_wrap(torch.nn.Module):
    """
    The EquiTransformer Diffusion Module.
    """
    def __init__(
            self,
            dynamics, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0., 0., 0.), include_charges=True, 
            uni_diffusion=False, timesteps2: int = 1000, pre_training=False,
            property_pred=False, prediction_threshold_t=10, 
            target_property=None, use_prop_pred=1, 
            freeze_gradient=False, unnormal_time_step=False, 
            only_noisy_node=False, half_noisy_node=False, 
            sep_noisy_node=False, relay_sampling=0, 
            second_dynamics=None, sampling_threshold_t=10, 
            atom_type_pred=True, bfn_schedule=False, 
            device='cpu', atom_types=5,
            bond_pred=False, bfn_str=False,
            str_schedule_norm=False,
            str_loss_type = 'denoise_loss', 
            str_sigma_h = 0.05, str_sigma_x = 0.05,
            temp_index = 0, optimal_sampling = 0,
            len_dim=3, angle_dim=3, lambda_l=1, lambda_a=1, lambda_f=10, 
            lambda_type=1, lambda_rep=1, cutoff=0.5, adjust_atom_type=False,
            lambda_type_adjust=1, sde_type="ve",
            debug_atom_types=False, debug_atom_dir=None, atom_decoder=None,
            known_atom_class_ids=None, unknown_atom_type_idx=0,
            disable_all_h_guard=False, all_h_guard_topk=DEFAULT_ATOM_TYPE_REPAIR_TOPK,
            all_h_guard_min_non_h=1,
            **kwargs):
        super().__init__()
        self.property_pred = property_pred
        self.prediction_threshold_t = prediction_threshold_t
        self.target_property = target_property
        self.use_prop_pred = use_prop_pred
        
        # add relay sampling and second_phi
        self.relay_sampling = relay_sampling
        self.second_dynamics = second_dynamics
        self.sampling_threshold_t = sampling_threshold_t
        
        # sde type
        self.sde_type = sde_type
        print("SDE type: ", sde_type)

        # bfn schedule
        self.bfn_schedule = bfn_schedule

        # optimal sampling
        self.optimal_sampling = optimal_sampling
        
        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            if self.sde_type == "ve":
                self.gamma = PredefinedNoiseSchedule_ve(noise_schedule="ve_loglinear", timesteps=timesteps)
                print("Using VE-native predefined noise schedule for gamma.")
            else: # "vp"
                self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                    precision=noise_precision)
                print("Using predefined (VP-SDE) noise schedule for gamma.")

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization
        self.uni_diffusion = uni_diffusion
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        print("norm_values: ", norm_values)
        print("norm biases: ", norm_biases)
        self.register_buffer('buffer', torch.zeros(1))
        
        self.pre_training = pre_training
        if self.pre_training:
            self.mask_indicator = False
        else:
            self.mask_indicator = None

        if noise_schedule != 'learned' and self.sde_type == "ve":
            # self.check_issues_norm_values()
            self.check_sigma_max_too_small(
                num_stdevs=8,
                min_coverage=5.0,
                require_gt1=True,
                max_offset_integer=3,
            )
            
        self.freeze_gradient = freeze_gradient
        
        self.unnormal_time_step = unnormal_time_step
        
        self.only_noisy_node = only_noisy_node
        self.half_noisy_node = half_noisy_node
        
        self.sep_noisy_node = sep_noisy_node
        
        self.atom_type_pred = atom_type_pred
        
        self.bond_pred = bond_pred
        # self.gamma_lst = np.load('/mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/gamma.npy')
        # self.gamma_lst = np.load("/home/AI4Science/luy2402/e3_diffusion_for_molecules/data/gamma_luyan/gamma.npy")
        
        bins = 9
        k_c, self.k_l, self.k_r = self.get_k_params(bins)
        self.K_c = torch.tensor(k_c).to(device)
        
        sigma1_coord = 0.001
        sigma1_charges = 0.15
        self.t_min = 0.0001
        self.sigma1_coord = torch.tensor([sigma1_coord], dtype=torch.float32, device=device)
        self.sigma1_charges = torch.tensor([sigma1_charges], dtype=torch.float32, device=device)
        self.atom_types = atom_types
        
        self.bfn_str = bfn_str
        self.str_loss_type = str_loss_type
        # self.str_sigma = str_sigma
        self.str_sigma_x = str_sigma_x
        self.str_sigma_h = str_sigma_h
        self.str_schedule_norm = str_schedule_norm
        self.temp_index = temp_index
        self.saved_grad = None

        # input: (lengths,t) and (angles, t)
        # lengths: [b,3], angles: [b,3], t: [b,1]
        # output: lengths and angles, both are [b,3]
        self.len_dim = len_dim
        self.angle_dim = angle_dim
        self.lambda_l = lambda_l
        self.lambda_a = lambda_a
        # print("use lambda_l: ", lambda_l)
        # print("use lambda_a: ", lambda_a)
        
        self.lambda_f = lambda_f
        self.lambda_type = lambda_type
        self.lambda_rep = lambda_rep
        self.cutoff = cutoff
        self.adjust_atom_type = adjust_atom_type
        self.lambda_type_adjust = lambda_type_adjust
        self.atom_decoder = list(atom_decoder) if atom_decoder is not None else None
        self.known_atom_class_ids = list(known_atom_class_ids) if known_atom_class_ids is not None else None
        self.unknown_atom_type_idx = int(unknown_atom_type_idx)
        self.h_class_idx = 1
        if self.atom_decoder is not None:
            for class_idx, symbol in enumerate(self.atom_decoder):
                if symbol == "H":
                    self.h_class_idx = int(class_idx)
                    break
        self.debug_atom_types = bool(debug_atom_types) or os.environ.get("DEBUG_ATOM_TYPES", "0") == "1"
        self.debug_atom_dir = debug_atom_dir or os.environ.get("DEBUG_ATOM_TYPES_DIR")
        self.disable_all_h_guard_arg = bool(disable_all_h_guard)
        self.all_h_guard_enabled = not self.disable_all_h_guard_arg
        self.atom_type_repair_topk = int(DEFAULT_ATOM_TYPE_REPAIR_TOPK)
        self.all_h_guard_topk = max(1, int(all_h_guard_topk))
        self.all_h_guard_min_non_h = max(1, int(all_h_guard_min_non_h))
        self._atom_debug_decoder_written = False
        self._input_atom_type_debug_written = False
        self._last_prepare_inputs_debug = {}
        self._last_dynamics_atom_debug_info = {}
        self._atom_prediction_step_tensor_dir = None
        self._atom_type_all_h_guard_summary = {}
        self._reset_atom_type_all_h_guard_summary()

        print("use lambda_type: ", lambda_type)
        print("use lambda_rep: ", lambda_rep)
        print("cutoff for repulsion loss: ", cutoff)
        print("use lambda_type_adjust: ", lambda_type_adjust)
        print("adjust atom type during diffusion: ", adjust_atom_type)
        print("disable_all_h_guard arg: ", self.disable_all_h_guard_arg)
        print("all-H guard enabled: ", self.all_h_guard_enabled)
        print("all-H guard top-k: ", self.all_h_guard_topk)
        print("all-H guard min non-H: ", self.all_h_guard_min_non_h)
        os.environ["MCW_ALL_H_GUARD_ENABLED"] = "1" if self.all_h_guard_enabled else "0"
        os.environ["MCW_ALL_H_GUARD_DISABLED_ARG"] = "1" if self.disable_all_h_guard_arg else "0"

        if self.debug_atom_types:
            if not self.debug_atom_dir:
                self.debug_atom_dir = os.path.join(os.getcwd(), "atom_type_debug")
            os.makedirs(self.debug_atom_dir, exist_ok=True)
            self._atom_prediction_step_tensor_dir = os.path.join(
                self.debug_atom_dir,
                "atom_type_step_tensors",
            )
            os.makedirs(self._atom_prediction_step_tensor_dir, exist_ok=True)
            os.environ["DEBUG_ATOM_TYPES"] = "1"
            os.environ["DEBUG_ATOM_TYPES_DIR"] = self.debug_atom_dir
            self._write_atom_debug_line(
                "atom_type_session.jsonl",
                {
                    "event": "session_start",
                    "model": self.__class__.__name__,
                    "num_classes": int(self.num_classes),
                    "atom_type_pred": bool(self.atom_type_pred),
                    "debug_atom_dir": self.debug_atom_dir,
                    "disable_all_h_guard_arg": bool(self.disable_all_h_guard_arg),
                    "all_h_guard_enabled": bool(self.all_h_guard_enabled),
                    "all_h_guard_topk": int(self.all_h_guard_topk),
                    "all_h_guard_min_non_h": int(self.all_h_guard_min_non_h),
                    "all_h_guard_fail_fast_enabled": bool(self._all_h_guard_fail_fast_enabled()),
                    # This implementation predicts atom logits from geometry and does not
                    # maintain an explicit categorical reverse state during sampling.
                    "categorical_sampling_mode": "pred_logits_only",
                },
            )
            self._log_atom_decoder_info()
            print(f"Atom-type debug enabled. Outputs will be written to {self.debug_atom_dir}")

        print(f"{self.__class__.__name__} initialized.")
        
    
    def save_intermediate_grad(self, grad):
        self.saved_grad = grad

    def _write_atom_debug_line(self, filename, payload):
        if not self.debug_atom_types or not self.debug_atom_dir:
            return
        os.makedirs(self.debug_atom_dir, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("pid", os.getpid())
        with open(os.path.join(self.debug_atom_dir, filename), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _tensor_stats(self, tensor):
        tensor = tensor.detach()
        finite_mask = torch.isfinite(tensor)
        finite_values = tensor[finite_mask]
        if finite_values.numel() == 0:
            return {
                "shape": list(tensor.shape),
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
            }
        return {
            "shape": list(tensor.shape),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "inf_count": int(torch.isinf(tensor).sum().item()),
        }

    def _class_index_to_symbol(self, index):
        if self.atom_decoder is None:
            return f"class_{int(index)}"
        return class_index_to_symbol(self.atom_decoder, int(index))

    def _log_atom_decoder_info(self):
        if self._atom_debug_decoder_written:
            return
        self._atom_debug_decoder_written = True
        decoder_index0 = None
        if self.atom_decoder:
            decoder_index0 = self.atom_decoder[0]
        self._write_atom_debug_line(
            "atom_type_session.jsonl",
            {
                "event": "atom_decoder_info",
                "atom_decoder": self.atom_decoder,
                "atom_decoder_index0": decoder_index0,
                "atom_decoder_index0_is_H": decoder_index0 == "H",
                "unknown_atom_type_idx": int(self.unknown_atom_type_idx),
                "class_index0_symbol": self._class_index_to_symbol(0),
                "class_index1_symbol": self._class_index_to_symbol(1),
                "known_atom_class_ids": self.known_atom_class_ids,
            },
        )

    def _reset_atom_type_all_h_guard_summary(self):
        self._atom_type_all_h_guard_summary = {
            "guard_decode_call_count": 0,
            "guard_trigger_count": 0,
            "guard_repair_success_count": 0,
            "total_samples": 0,
            "raw_all_H_count": 0,
            "final_all_H_count": 0,
            "num_all_H_rescued": 0,
            "rescue_logprob_penalties": [],
            "rescued_new_element_distribution": Counter(),
            "single_element_count_before_guard": 0,
            "single_element_count_after_guard": 0,
        }

    def _write_atom_type_all_h_guard_summary(self):
        if not self.debug_atom_types or not self.debug_atom_dir:
            return
        summary = self._atom_type_all_h_guard_summary
        penalties = sorted(float(v) for v in summary["rescue_logprob_penalties"])
        if penalties:
            penalty_count = len(penalties)
            if penalty_count % 2 == 1:
                penalty_median = penalties[penalty_count // 2]
            else:
                penalty_median = 0.5 * (penalties[penalty_count // 2 - 1] + penalties[penalty_count // 2])
            penalty_stats = {
                "count": penalty_count,
                "mean": float(sum(penalties) / penalty_count),
                "median": float(penalty_median),
                "max": float(penalties[-1]),
            }
        else:
            penalty_stats = {
                "count": 0,
                "mean": None,
                "median": None,
                "max": None,
            }

        payload = {
            "disable_all_h_guard_arg": bool(self.disable_all_h_guard_arg),
            "all_h_guard_enabled": bool(self.all_h_guard_enabled),
            "all_h_guard_fail_fast_enabled": bool(self._all_h_guard_fail_fast_enabled()),
            "guard_decode_call_count": int(summary["guard_decode_call_count"]),
            "guard_trigger_count": int(summary["guard_trigger_count"]),
            "guard_repair_success_count": int(summary["guard_repair_success_count"]),
            "total_samples": int(summary["total_samples"]),
            "raw_all_H_count": int(summary["raw_all_H_count"]),
            "final_all_H_count": int(summary["final_all_H_count"]),
            "num_all_H_rescued": int(summary["num_all_H_rescued"]),
            "single_element_count_before_guard": int(summary["single_element_count_before_guard"]),
            "single_element_count_after_guard": int(summary["single_element_count_after_guard"]),
            "rescue_logprob_penalty_stats": penalty_stats,
            "rescued_new_element_distribution": dict(
                sorted(
                    (str(symbol), int(count))
                    for symbol, count in summary["rescued_new_element_distribution"].items()
                )
            ),
        }
        with open(
            os.path.join(self.debug_atom_dir, "atom_type_all_h_guard_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _all_h_guard_fail_fast_enabled(self):
        return bool(self.all_h_guard_enabled or self.debug_atom_types)

    def _raise_all_h_guard_violation(self, stage, payload):
        failure_payload = dict(payload)
        failure_payload.setdefault("event", "all_h_guard_violation")
        failure_payload.setdefault("stage", stage)
        failure_payload.setdefault("all_h_guard_enabled", bool(self.all_h_guard_enabled))
        failure_payload.setdefault("disable_all_h_guard_arg", bool(self.disable_all_h_guard_arg))
        failure_payload.setdefault("debug_atom_types", bool(self.debug_atom_types))
        self._write_atom_debug_line("atom_type_guard_failures.jsonl", failure_payload)
        sample_global_index = failure_payload.get("sample_global_index")
        sample_suffix = (
            f" at global sample index {int(sample_global_index)}"
            if sample_global_index is not None else ""
        )
        decoded_species = failure_payload.get("decoded_species")
        decoded_species_suffix = f", decoded_species={decoded_species}" if decoded_species is not None else ""
        raise RuntimeError(
            f"[AllHGuard] {stage} observed all-H output{sample_suffix}{decoded_species_suffix}"
        )

    def _compute_assignment_log_score(self, probs, class_idx):
        if probs.numel() == 0 or class_idx.numel() == 0:
            return 0.0
        gather = probs.gather(dim=-1, index=class_idx.unsqueeze(-1)).squeeze(-1)
        log_gather = torch.log(gather.clamp(min=1e-12))
        return float(log_gather.sum().item())

    def _get_known_non_h_class_ids(self):
        if self.known_atom_class_ids is None:
            class_ids = list(range(1, self.num_classes))
        else:
            class_ids = [int(idx) for idx in self.known_atom_class_ids]
        return [
            int(class_idx)
            for class_idx in class_ids
            if 0 < int(class_idx) < self.num_classes and int(class_idx) != int(self.h_class_idx)
        ]

    def _build_all_h_guard_candidate_list(self, probs_row, topk):
        k = min(max(1, int(topk)), probs_row.size(-1))
        top_probs, top_idx = torch.topk(probs_row, k=k, dim=-1)
        candidates = []
        for class_idx, prob in zip(top_idx.detach().cpu().tolist(), top_probs.detach().cpu().tolist()):
            class_idx = int(class_idx)
            if class_idx == self.unknown_atom_type_idx or class_idx == self.h_class_idx:
                continue
            candidates.append(
                {
                    "class_idx": class_idx,
                    "symbol": self._class_index_to_symbol(class_idx),
                    "prob": float(prob),
                }
            )
        return candidates

    def _repair_all_h_assignment(self, valid_logits, valid_probs, raw_idx):
        n_real, num_classes = valid_probs.shape
        raw_score = self._compute_assignment_log_score(valid_probs, raw_idx)
        raw_is_all_h = bool(n_real > 0 and raw_idx.eq(self.h_class_idx).all().item())
        result = {
            "raw_all_H": raw_is_all_h,
            "raw_score": raw_score,
            "raw_decoded_class_ids": [int(v) for v in raw_idx.detach().cpu().tolist()],
            "raw_decoded_species": [self._class_index_to_symbol(v) for v in raw_idx.detach().cpu().tolist()],
            "repaired": False,
            "rescue_logprob_penalty": 0.0,
            "raw_unique_species_count": len(set(int(v) for v in raw_idx.detach().cpu().tolist())),
            "final_unique_species_count": len(set(int(v) for v in raw_idx.detach().cpu().tolist())),
            "replacements": [],
        }
        if (not self.all_h_guard_enabled) or (not raw_is_all_h) or n_real == 0:
            result["final_score"] = raw_score
            return raw_idx.clone(), result

        log_probs = torch.log(valid_probs.clamp(min=1e-12))
        site_entropy = -(valid_probs * log_probs).sum(dim=-1)
        valid_non_h_class_ids = self._get_known_non_h_class_ids()
        if not valid_non_h_class_ids:
            result["final_score"] = raw_score
            return raw_idx.clone(), result

        working_idx = raw_idx.clone()
        used_sites = set()
        replacement_budget = min(int(self.all_h_guard_min_non_h), int(n_real))
        topk_plan = []
        topk_plan.append(min(max(1, int(self.all_h_guard_topk)), num_classes))
        topk_plan.append(min(max(topk_plan[0] * 2, topk_plan[0] + 1), num_classes))
        if topk_plan[-1] != num_classes:
            topk_plan.append(num_classes)

        for _ in range(replacement_budget):
            best_choice = None
            for topk in topk_plan:
                found_candidate_in_tier = False
                for site_idx in range(n_real):
                    if site_idx in used_sites:
                        continue
                    candidates = self._build_all_h_guard_candidate_list(valid_probs[site_idx], topk=topk)
                    if not candidates and topk == num_classes:
                        candidates = [
                            {
                                "class_idx": int(class_idx),
                                "symbol": self._class_index_to_symbol(class_idx),
                                "prob": float(valid_probs[site_idx, int(class_idx)].item()),
                            }
                            for class_idx in valid_non_h_class_ids
                        ]
                    if not candidates:
                        continue
                    found_candidate_in_tier = True
                    logprob_h = float(log_probs[site_idx, self.h_class_idx].item())
                    for candidate in candidates:
                        class_idx = int(candidate["class_idx"])
                        logprob_new = float(log_probs[site_idx, class_idx].item())
                        penalty = logprob_h - logprob_new
                        choice = {
                            "site_index": int(site_idx),
                            "old_class": int(self.h_class_idx),
                            "old_symbol": self._class_index_to_symbol(self.h_class_idx),
                            "new_class": class_idx,
                            "new_symbol": candidate["symbol"],
                            "logprob_H": logprob_h,
                            "logprob_new": logprob_new,
                            "rescue_logprob_penalty": float(penalty),
                            "site_entropy": float(site_entropy[site_idx].item()),
                            "topk_candidates_at_repaired_site": candidates,
                            "search_topk": int(topk),
                        }
                        if best_choice is None or choice["rescue_logprob_penalty"] < best_choice["rescue_logprob_penalty"]:
                            best_choice = choice
                if found_candidate_in_tier:
                    break

            if best_choice is None:
                break
            working_idx[best_choice["site_index"]] = int(best_choice["new_class"])
            used_sites.add(int(best_choice["site_index"]))
            result["replacements"].append(best_choice)

        result["repaired"] = len(result["replacements"]) > 0
        result["rescue_logprob_penalty"] = float(
            sum(replacement["rescue_logprob_penalty"] for replacement in result["replacements"])
        )
        result["final_score"] = self._compute_assignment_log_score(valid_probs, working_idx)
        result["final_unique_species_count"] = len(set(int(v) for v in working_idx.detach().cpu().tolist()))
        if result["replacements"]:
            first_replacement = result["replacements"][0]
            result["replaced_site_index"] = int(first_replacement["site_index"])
            result["old_class"] = int(first_replacement["old_class"])
            result["old_symbol"] = first_replacement["old_symbol"]
            result["new_class"] = int(first_replacement["new_class"])
            result["new_symbol"] = first_replacement["new_symbol"]
            result["logprob_H"] = float(first_replacement["logprob_H"])
            result["logprob_new"] = float(first_replacement["logprob_new"])
            result["site_entropy"] = float(first_replacement["site_entropy"])
            result["topk_candidates_at_repaired_site"] = first_replacement["topk_candidates_at_repaired_site"]
        else:
            result["replaced_site_index"] = None
            result["old_class"] = None
            result["old_symbol"] = None
            result["new_class"] = None
            result["new_symbol"] = None
            result["logprob_H"] = None
            result["logprob_new"] = None
            result["site_entropy"] = None
            result["topk_candidates_at_repaired_site"] = []
        result["final_all_H"] = bool(
            working_idx.numel() > 0 and working_idx.eq(self.h_class_idx).all().item()
        )
        return working_idx, result

    def _assert_no_all_h_in_one_hot_batch(self, one_hot, node_mask, source_tag, round_index, stage):
        if not self._all_h_guard_fail_fast_enabled():
            return
        assert one_hot.dim() == 3, f"one_hot must be [B, N, C], got {tuple(one_hot.shape)}"
        elem_idx = torch.argmax(one_hot, dim=-1)
        valid_mask = node_mask.squeeze(-1).bool()
        batch_size = one_hot.size(0)
        for batch_idx in range(batch_size):
            valid_idx = elem_idx[batch_idx, valid_mask[batch_idx]]
            if valid_idx.numel() == 0:
                continue
            if bool(valid_idx.eq(self.h_class_idx).all().item()):
                self._raise_all_h_guard_violation(
                    stage,
                    {
                        "source_tag": source_tag,
                        "round_index": int(round_index),
                        "sample_local_index": int(batch_idx),
                        "sample_global_index": int(round_index * batch_size + batch_idx),
                        "decoded_class_ids": [int(v) for v in valid_idx.detach().cpu().tolist()],
                        "decoded_species": [self._class_index_to_symbol(v) for v in valid_idx.detach().cpu().tolist()],
                    },
                )

    def _prediction_window_start_step_index(self):
        return max(1, int(self.T) - int(self.prediction_threshold_t) + 1)

    def _is_prediction_window_step(self, step_index):
        return int(step_index) >= self._prediction_window_start_step_index()

    def _mask_unknown_atom_class(self, logits):
        masked_logits = logits.detach().clone()
        masked_logits[:, :, self.unknown_atom_type_idx] = -1e9
        return masked_logits

    def _mean_pairwise_cosine_similarity(self, values):
        if values.dim() != 2 or values.size(0) < 2:
            return None
        values = values.float()
        if self.unknown_atom_type_idx == 0 and values.size(-1) > 1:
            values = values[:, 1:]
        if values.size(-1) == 0:
            return None
        normalized = F.normalize(values, p=2, dim=-1, eps=1e-12)
        cosine_matrix = normalized @ normalized.transpose(0, 1)
        pair_mask = ~torch.eye(cosine_matrix.size(0), dtype=torch.bool, device=cosine_matrix.device)
        if not pair_mask.any():
            return None
        return float(cosine_matrix[pair_mask].mean().item())

    def _record_atom_type_state_flow(
        self,
        source_tag,
        round_index,
        step_index,
        atom_type_state_input_available,
        input_state_updated_step_index,
        atom_type_state_output,
    ):
        if not self.debug_atom_types:
            return
        payload = {
            "event": "atom_type_state_flow",
            "source_tag": source_tag,
            "round_index": int(round_index),
            "step_index": int(step_index),
            "prediction_window_start_step_index": int(self._prediction_window_start_step_index()),
            "prediction_window_step": bool(self._is_prediction_window_step(step_index)),
            "atom_type_state_input_available": bool(atom_type_state_input_available),
            "input_state_updated_step_index": int(input_state_updated_step_index)
            if input_state_updated_step_index is not None else None,
            "state_output_available": bool(atom_type_state_output is not None),
            "state_overwritten_with_current_logits": bool(atom_type_state_output is not None),
            "prepare_inputs_used_atom_type_state": bool(self._last_prepare_inputs_debug.get("used_atom_type_state", False)),
            "empty_graph_fallback": bool(self._last_dynamics_atom_debug_info.get("empty_graph", False)),
            "fallback_source": self._last_dynamics_atom_debug_info.get("fallback_source"),
            "used_previous_atom_logits_fallback": bool(
                self._last_dynamics_atom_debug_info.get("fallback_source") == "previous_atom_logits"
            ),
        }
        if atom_type_state_output is not None:
            payload["state_output_shape"] = list(atom_type_state_output.shape)
            payload["state_output_stats"] = self._tensor_stats(atom_type_state_output)
        self._write_atom_debug_line("atom_type_state_flow.jsonl", payload)

    def _record_atom_prediction_window_step(
        self,
        logits,
        node_mask,
        round_index,
        step_index,
        source_tag,
        atom_type_state_input_available,
        input_state_updated_step_index,
    ):
        if not self.debug_atom_types:
            return
        assert logits.dim() == 3, f"prediction-window logits must be [B, N, C], got {tuple(logits.shape)}"
        assert node_mask.dim() == 3 and node_mask.size(-1) == 1, \
            f"prediction-window node_mask must be [B, N, 1], got {tuple(node_mask.shape)}"

        B, N, C = logits.shape
        valid_mask = node_mask.squeeze(-1).bool()
        raw_probs = torch.softmax(logits, dim=ATOM_TYPE_SOFTMAX_DIM)
        masked_logits = self._mask_unknown_atom_class(logits)
        decode_probs = torch.softmax(masked_logits, dim=ATOM_TYPE_SOFTMAX_DIM)
        decoded_idx = torch.argmax(masked_logits, dim=-1)

        batch_payload = {
            "event": "atom_type_prediction_step_batch",
            "source_tag": source_tag,
            "round_index": int(round_index),
            "step_index": int(step_index),
            "prediction_window_start_step_index": int(self._prediction_window_start_step_index()),
            "logits_shape": list(logits.shape),
            "probs_shape": list(decode_probs.shape),
            "atom_type_state_input_available": bool(atom_type_state_input_available),
            "input_state_updated_step_index": int(input_state_updated_step_index)
            if input_state_updated_step_index is not None else None,
            "empty_graph_fallback": bool(self._last_dynamics_atom_debug_info.get("empty_graph", False)),
            "fallback_source": self._last_dynamics_atom_debug_info.get("fallback_source"),
            "used_previous_atom_logits_fallback": bool(
                self._last_dynamics_atom_debug_info.get("fallback_source") == "previous_atom_logits"
            ),
            "logits_stats": self._tensor_stats(logits),
            "decode_probs_stats": self._tensor_stats(decode_probs),
        }
        self._write_atom_debug_line("atom_type_prediction_step_batches.jsonl", batch_payload)

        if self._atom_prediction_step_tensor_dir:
            tensor_payload = {
                "event": "atom_type_prediction_step_tensors",
                "source_tag": source_tag,
                "round_index": int(round_index),
                "step_index": int(step_index),
                "prediction_window_start_step_index": int(self._prediction_window_start_step_index()),
                "sample_global_indices": [int(round_index * B + b) for b in range(B)],
                "sample_local_indices": [int(b) for b in range(B)],
                "atom_type_state_input_available": bool(atom_type_state_input_available),
                "input_state_updated_step_index": int(input_state_updated_step_index)
                if input_state_updated_step_index is not None else None,
                "empty_graph_fallback": bool(self._last_dynamics_atom_debug_info.get("empty_graph", False)),
                "fallback_source": self._last_dynamics_atom_debug_info.get("fallback_source"),
                "used_previous_atom_logits_fallback": bool(
                    self._last_dynamics_atom_debug_info.get("fallback_source") == "previous_atom_logits"
                ),
                "logits": logits.detach().cpu(),
                "raw_probs": raw_probs.detach().cpu(),
                "decode_probs": decode_probs.detach().cpu(),
                "decoded_idx": decoded_idx.detach().cpu(),
                "node_mask": node_mask.detach().cpu(),
            }
            torch.save(
                tensor_payload,
                os.path.join(
                    self._atom_prediction_step_tensor_dir,
                    f"round_{int(round_index):04d}_step_{int(step_index):04d}.pt",
                ),
            )

        for b in range(B):
            mask = valid_mask[b]
            n_real = int(mask.sum().item())
            sample_global_index = int(round_index * B + b)
            valid_logits = logits[b, mask]
            valid_raw_probs = raw_probs[b, mask]
            valid_decode_probs = decode_probs[b, mask]
            valid_decoded_idx = decoded_idx[b, mask]
            topk = min(5, C)

            if n_real > 0:
                top_probs, top_idx = torch.topk(valid_decode_probs, k=topk, dim=-1)
                top1_probs = top_probs[:, 0]
                top2_probs = top_probs[:, 1] if topk > 1 else torch.zeros_like(top1_probs)
                margin = top1_probs - top2_probs
                h_prob = valid_decode_probs[:, 1] if C > 1 else torch.zeros_like(top1_probs)
                node_logits_std = valid_logits.std(dim=-1, unbiased=False)
                logits_excluding_unknown = valid_logits[:, 1:] if C > 1 else valid_logits
                probs_excluding_unknown = valid_decode_probs[:, 1:] if C > 1 else valid_decode_probs
                cross_node_logits_std_mean = float(
                    logits_excluding_unknown.std(dim=0, unbiased=False).mean().item()
                ) if n_real > 1 and logits_excluding_unknown.numel() > 0 else 0.0
                cross_node_probs_std_mean = float(
                    probs_excluding_unknown.std(dim=0, unbiased=False).mean().item()
                ) if n_real > 1 and probs_excluding_unknown.numel() > 0 else 0.0
                mean_cosine_similarity = self._mean_pairwise_cosine_similarity(logits_excluding_unknown)
                raw_class0_prob = valid_raw_probs[:, self.unknown_atom_type_idx]
                top5_per_node = [
                    {
                        "node_index": int(node_idx),
                        "class_ids": [int(v) for v in top_idx[node_idx].detach().cpu().tolist()],
                        "species": [self._class_index_to_symbol(v) for v in top_idx[node_idx].detach().cpu().tolist()],
                        "probs": [float(v) for v in top_probs[node_idx].detach().cpu().tolist()],
                    }
                    for node_idx in range(valid_decode_probs.size(0))
                ]
            else:
                margin = torch.zeros(0, device=logits.device)
                h_prob = torch.zeros(0, device=logits.device)
                node_logits_std = torch.zeros(0, device=logits.device)
                cross_node_logits_std_mean = None
                cross_node_probs_std_mean = None
                mean_cosine_similarity = None
                raw_class0_prob = torch.zeros(0, device=logits.device)
                top5_per_node = []

            sample_payload = {
                "event": "atom_type_prediction_step",
                "source_tag": source_tag,
                "round_index": int(round_index),
                "step_index": int(step_index),
                "prediction_window_start_step_index": int(self._prediction_window_start_step_index()),
                "sample_local_index": int(b),
                "sample_global_index": sample_global_index,
                "real_atom_count": n_real,
                "logits_shape": [n_real, C],
                "probs_shape": [n_real, C],
                "atom_type_state_input_available": bool(atom_type_state_input_available),
                "input_state_updated_step_index": int(input_state_updated_step_index)
                if input_state_updated_step_index is not None else None,
                "empty_graph_fallback": bool(self._last_dynamics_atom_debug_info.get("empty_graph", False)),
                "fallback_source": self._last_dynamics_atom_debug_info.get("fallback_source"),
                "used_previous_atom_logits_fallback": bool(
                    self._last_dynamics_atom_debug_info.get("fallback_source") == "previous_atom_logits"
                ),
                "decoded_argmax_class_ids": [int(v) for v in valid_decoded_idx.detach().cpu().tolist()],
                "decoded_argmax_species": [self._class_index_to_symbol(v) for v in valid_decoded_idx.detach().cpu().tolist()],
                "all_nodes_top1_H": bool(n_real > 0 and valid_decoded_idx.eq(1).all().item()),
                "top5_per_node": top5_per_node,
                "h_probability_per_node": [float(v) for v in h_prob.detach().cpu().tolist()],
                "h_probability_stats": self._tensor_stats(h_prob),
                "raw_class0_probability_stats": self._tensor_stats(raw_class0_prob),
                "top1_top2_margin_per_node": [float(v) for v in margin.detach().cpu().tolist()],
                "top1_top2_margin_stats": self._tensor_stats(margin),
                "node_logits_std_per_node": [float(v) for v in node_logits_std.detach().cpu().tolist()],
                "node_logits_std_stats": self._tensor_stats(node_logits_std),
                "logits_diversity_across_nodes_mean": cross_node_logits_std_mean,
                "probs_diversity_across_nodes_mean": cross_node_probs_std_mean,
                "mean_cosine_similarity_between_node_logits": mean_cosine_similarity,
                "logit_stats": self._tensor_stats(valid_logits) if n_real > 0 else self._tensor_stats(valid_logits),
                "prob_stats": self._tensor_stats(valid_decode_probs) if n_real > 0 else self._tensor_stats(valid_decode_probs),
            }
            self._write_atom_debug_line("atom_type_prediction_step_trace.jsonl", sample_payload)

    def _finalize_atom_type_logits(
        self,
        logits,
        node_mask,
        round_index=0,
        source_tag="sample",
        rl=None,
        ra=None,
        apply_repair=True,
        final_window_ensemble_logits=None,
        final_window_step_indices=None,
    ):
        assert logits.dim() == 3, f"atom logits must be [B, N, C], got {tuple(logits.shape)}"
        assert node_mask.dim() == 3 and node_mask.size(-1) == 1, \
            f"node_mask must be [B, N, 1], got {tuple(node_mask.shape)}"
        B, N, C = logits.shape
        assert node_mask.shape[0] == B and node_mask.shape[1] == N, \
            f"logits/node_mask shape mismatch: logits={tuple(logits.shape)} node_mask={tuple(node_mask.shape)}"
        assert C == self.num_classes, \
            f"atom logits last dim must equal num_classes={self.num_classes}, got {C}"

        decode_logits = logits
        decode_logits_source = "final_step_logits"
        if final_window_ensemble_logits is not None:
            assert final_window_ensemble_logits.shape == logits.shape, \
                f"final_window_ensemble_logits must be [B, N, C], got {tuple(final_window_ensemble_logits.shape)}"
            decode_logits = final_window_ensemble_logits
            decode_logits_source = "final_window_averaged_logits"

        softmax_dim = ATOM_TYPE_SOFTMAX_DIM
        probs = torch.softmax(decode_logits, dim=softmax_dim)
        assert probs.shape == decode_logits.shape, \
            f"probs shape mismatch: decode_logits={tuple(decode_logits.shape)} probs={tuple(probs.shape)}"
        assert torch.isfinite(probs).all(), "Atom probabilities contain NaN/Inf after softmax."

        valid_mask = node_mask.squeeze(-1).bool()
        probs_sum = probs.sum(dim=-1)
        if valid_mask.any():
            valid_probs_sum = probs_sum[valid_mask]
            assert torch.allclose(
                valid_probs_sum,
                torch.ones_like(valid_probs_sum),
                atol=1e-4,
                rtol=1e-4,
            ), "Atom probabilities on valid nodes do not sum to 1."
            valid_class_std = decode_logits[valid_mask].std(dim=-1, unbiased=False)
            all_class_constant = bool(torch.all(valid_class_std <= 1e-8).item())
        else:
            valid_class_std = None
            all_class_constant = False

        argmax_before = torch.argmax(decode_logits, dim=-1)
        assert argmax_before.shape == (B, N), \
            f"argmax(logits) must be [B, N], got {tuple(argmax_before.shape)}"

        masked_logits = decode_logits.clone()
        masked_logits[:, :, self.unknown_atom_type_idx] = -1e9
        masked_probs = torch.softmax(masked_logits, dim=softmax_dim)
        argmax_after_mask = torch.argmax(masked_logits, dim=-1)

        if apply_repair:
            raw_elem_idx, repaired_flags = repair_composition_batch(
                masked_logits,
                node_mask,
                smact_validity_fn=smact_validity,
                topk=self.atom_type_repair_topk,
                max_replace_atoms=2,
            )
        else:
            raw_elem_idx = argmax_after_mask
            repaired_flags = [False for _ in range(B)]

        elem_idx = raw_elem_idx.clone()
        assert elem_idx.shape == (B, N), \
            f"decoded atom type must be [B, N], got {tuple(elem_idx.shape)}"
        class0_on_valid = None
        if valid_mask.any():
            class0_on_valid = elem_idx[valid_mask].eq(self.unknown_atom_type_idx)

        batch_summary = {
            "event": "atom_type_batch_summary",
            "source_tag": source_tag,
            "round_index": int(round_index),
            "softmax_dim": softmax_dim,
            "decode_logits_source": decode_logits_source,
            "final_window_step_indices": [int(v) for v in (final_window_step_indices or [])],
            "logits": self._tensor_stats(logits),
            "decode_logits": self._tensor_stats(decode_logits),
            "probs": self._tensor_stats(probs),
            "argmax_before_shape": list(argmax_before.shape),
            "argmax_after_mask_shape": list(argmax_after_mask.shape),
            "decoded_shape": list(elem_idx.shape),
            "probs_sum_valid_min": float(probs_sum[valid_mask].min().item()) if valid_mask.any() else None,
            "probs_sum_valid_max": float(probs_sum[valid_mask].max().item()) if valid_mask.any() else None,
            "valid_class_std_min": float(valid_class_std.min().item()) if valid_class_std is not None and valid_class_std.numel() > 0 else None,
            "valid_class_std_max": float(valid_class_std.max().item()) if valid_class_std is not None and valid_class_std.numel() > 0 else None,
            "all_valid_nodes_class_constant_logits": all_class_constant,
            "padding_argmax_before_unique": sorted({int(v) for v in argmax_before[~valid_mask].detach().cpu().tolist()}),
            "padding_argmax_after_mask_unique": sorted({int(v) for v in argmax_after_mask[~valid_mask].detach().cpu().tolist()}),
            "samples": [],
            "prepare_inputs_debug": dict(self._last_prepare_inputs_debug),
            "dynamics_debug": dict(self._last_dynamics_atom_debug_info),
        }
        if rl is not None:
            batch_summary["rl_shape"] = list(rl.shape)
        if ra is not None:
            batch_summary["ra_shape"] = list(ra.shape)

        all_h_global_indices = []
        raw_all_h_count = 0
        final_all_h_count = 0
        num_all_h_rescued = 0
        single_element_count_before_guard = 0
        single_element_count_after_guard = 0
        for b in range(B):
            mask = valid_mask[b]
            n_real = int(mask.sum().item())
            valid_logits = decode_logits[b, mask]
            valid_probs = probs[b, mask]
            valid_masked_probs = masked_probs[b, mask]
            raw_idx = argmax_before[b, mask]
            masked_idx = argmax_after_mask[b, mask]
            raw_decoded_idx = raw_elem_idx[b, mask]
            sample_global_index = int(round_index * B + b)
            final_idx, guard_info = self._repair_all_h_assignment(
                valid_logits=valid_logits,
                valid_probs=valid_masked_probs,
                raw_idx=raw_decoded_idx,
            )
            elem_idx[b, mask] = final_idx
            all_from_default_before_mask = bool(n_real > 0 and raw_idx.eq(0).all().item())
            raw_all_h = bool(guard_info["raw_all_H"])
            all_h = bool(n_real > 0 and final_idx.eq(self.h_class_idx).all().item())
            raw_counts = Counter(int(v) for v in raw_decoded_idx.detach().cpu().tolist())
            final_counts = Counter(int(v) for v in final_idx.detach().cpu().tolist())
            raw_species_counts = {
                self._class_index_to_symbol(idx): int(count)
                for idx, count in sorted(raw_counts.items())
            }
            species_counts = {
                self._class_index_to_symbol(idx): int(count)
                for idx, count in sorted(final_counts.items())
            }
            raw_all_h_count += int(raw_all_h)
            final_all_h_count += int(all_h)
            num_all_h_rescued += int(raw_all_h and (not all_h))
            single_element_count_before_guard += int(len(raw_counts) == 1 and n_real > 0)
            single_element_count_after_guard += int(len(final_counts) == 1 and n_real > 0)
            if raw_all_h and not all_h:
                replacement = guard_info["replacements"][0] if guard_info["replacements"] else None
                replacement_desc = (
                    f"site={replacement['site_index']} {replacement['old_symbol']}->{replacement['new_symbol']}"
                    if replacement is not None else "no-replacement-metadata"
                )
                print(
                    f"[AllHGuard] repaired raw all-H sample at global sample index "
                    f"{sample_global_index} ({source_tag}, round={round_index}, batch={b}): {replacement_desc}"
                )
            if self._all_h_guard_fail_fast_enabled() and all_h:
                self._raise_all_h_guard_violation(
                    "finalize_atom_type_logits",
                    {
                        "source_tag": source_tag,
                        "round_index": int(round_index),
                        "sample_local_index": int(b),
                        "sample_global_index": sample_global_index,
                        "raw_all_H": bool(raw_all_h),
                        "raw_decoded_class_ids": [int(v) for v in raw_decoded_idx.detach().cpu().tolist()],
                        "decoded_class_ids": [int(v) for v in final_idx.detach().cpu().tolist()],
                        "decoded_species": [self._class_index_to_symbol(v) for v in final_idx.detach().cpu().tolist()],
                        "repaired": bool(guard_info["repaired"]),
                        "replacements": guard_info["replacements"],
                    },
                )
            identical_logits_across_nodes = False
            zero_logits_across_nodes = False
            if n_real > 0:
                identical_logits_across_nodes = bool(
                    torch.allclose(
                        valid_logits,
                        valid_logits[:1].expand_as(valid_logits),
                        atol=1e-8,
                        rtol=0.0,
                    )
                )
                zero_logits_across_nodes = bool(
                    torch.allclose(
                        valid_logits,
                        torch.zeros_like(valid_logits),
                        atol=1e-8,
                        rtol=0.0,
                    )
                )
            sample_summary = {
                "event": "atom_type_sample_summary",
                "source_tag": source_tag,
                "round_index": int(round_index),
                "sample_local_index": int(b),
                "sample_global_index": sample_global_index,
                "real_atom_count": n_real,
                "unique_species_count": len(final_counts),
                "raw_unique_species_count": len(raw_counts),
                "raw_species_counts": raw_species_counts,
                "species_counts": species_counts,
                "all_H": all_h,
                "raw_all_H": raw_all_h,
                "all_from_default_index_before_mask": all_from_default_before_mask,
                "identical_logits_across_nodes": identical_logits_across_nodes,
                "zero_logits_across_nodes": zero_logits_across_nodes,
                "repaired": bool(repaired_flags[b]),
                "raw_argmax_unique": sorted({int(v) for v in raw_idx.detach().cpu().tolist()}),
                "masked_argmax_unique": sorted({int(v) for v in masked_idx.detach().cpu().tolist()}),
                "raw_decoded_unique": sorted({int(v) for v in raw_decoded_idx.detach().cpu().tolist()}),
                "raw_decoded_class_ids": [int(v) for v in raw_decoded_idx.detach().cpu().tolist()],
                "raw_decoded_species": [self._class_index_to_symbol(v) for v in raw_decoded_idx.detach().cpu().tolist()],
                "decoded_unique": sorted({int(v) for v in final_idx.detach().cpu().tolist()}),
                "decoded_class_ids": [int(v) for v in final_idx.detach().cpu().tolist()],
                "decoded_species": [self._class_index_to_symbol(v) for v in final_idx.detach().cpu().tolist()],
                "raw_score": float(guard_info["raw_score"]),
                "final_score": float(guard_info["final_score"]),
                "all_h_guard_enabled": bool(self.all_h_guard_enabled),
                "all_h_guard_repaired": bool(guard_info["repaired"]),
                "all_h_guard_min_non_h": int(self.all_h_guard_min_non_h),
                "all_h_guard_topk": int(self.all_h_guard_topk),
                "all_h_guard_replacements": guard_info["replacements"],
                "all_h_guard_rescue_logprob_penalty": float(guard_info["rescue_logprob_penalty"]),
                "all_h_guard_replaced_site_index": guard_info.get("replaced_site_index"),
                "all_h_guard_old_class": guard_info.get("old_class"),
                "all_h_guard_old_symbol": guard_info.get("old_symbol"),
                "all_h_guard_new_class": guard_info.get("new_class"),
                "all_h_guard_new_symbol": guard_info.get("new_symbol"),
                "all_h_guard_logprob_H": guard_info.get("logprob_H"),
                "all_h_guard_logprob_new": guard_info.get("logprob_new"),
                "all_h_guard_site_entropy": guard_info.get("site_entropy"),
                "all_h_guard_topk_candidates_at_repaired_site": guard_info.get("topk_candidates_at_repaired_site", []),
                "class0_on_valid_nodes": int(final_idx.eq(self.unknown_atom_type_idx).sum().item()),
                "dummy_atom_type_idx": self._last_prepare_inputs_debug.get("dummy_atom_type_idx"),
                "dummy_atom_type_symbol": self._last_prepare_inputs_debug.get("dummy_atom_type_symbol"),
                "dummy_atom_type_is_H": self._last_prepare_inputs_debug.get("dummy_atom_type_is_H"),
                "empty_graph_fallback": self._last_dynamics_atom_debug_info.get("empty_graph", False),
                "empty_graph_fallback_source": self._last_dynamics_atom_debug_info.get("fallback_source"),
                "all_valid_nodes_class_constant_logits": all_class_constant,
            }
            if n_real > 0:
                sample_summary["valid_logit_stats"] = self._tensor_stats(valid_logits)
                sample_summary["valid_prob_stats"] = self._tensor_stats(valid_masked_probs)
                if self.debug_atom_types or all_h or raw_all_h:
                    topk = min(5, valid_probs.size(-1))
                    top_probs, top_idx = torch.topk(valid_masked_probs, k=topk, dim=-1)
                    sample_summary["top5_per_node"] = [
                        {
                            "node_index": int(node_idx),
                            "class_ids": [int(v) for v in top_idx[node_idx].detach().cpu().tolist()],
                            "species": [self._class_index_to_symbol(v) for v in top_idx[node_idx].detach().cpu().tolist()],
                            "probs": [float(v) for v in top_probs[node_idx].detach().cpu().tolist()],
                        }
                        for node_idx in range(valid_probs.size(0))
                    ]
            if rl is not None:
                sample_summary["rl"] = [float(v) for v in rl[b].detach().cpu().tolist()]
            if ra is not None:
                sample_summary["ra"] = [float(v) for v in ra[b].detach().cpu().tolist()]
            batch_summary["samples"].append(sample_summary)

            if self.debug_atom_types or all_h or raw_all_h or all_from_default_before_mask or zero_logits_across_nodes or all_class_constant:
                self._write_atom_debug_line("atom_type_batches.jsonl", sample_summary)
            if self.debug_atom_types:
                self._write_atom_debug_line(
                    "atom_type_all_h_guard.jsonl",
                    {
                        "event": "atom_type_all_h_guard",
                        "source_tag": source_tag,
                        "round_index": int(round_index),
                        "sample_local_index": int(b),
                        "sample_global_index": sample_global_index,
                        "raw_decoded_class_ids": sample_summary["raw_decoded_class_ids"],
                        "raw_decoded_species": sample_summary["raw_decoded_species"],
                        "raw_all_H": bool(raw_all_h),
                        "raw_score": float(guard_info["raw_score"]),
                        "repaired": bool(guard_info["repaired"]),
                        "replaced_site_index": guard_info.get("replaced_site_index"),
                        "old_class": guard_info.get("old_class"),
                        "old_symbol": guard_info.get("old_symbol"),
                        "new_class": guard_info.get("new_class"),
                        "new_symbol": guard_info.get("new_symbol"),
                        "logprob_H": guard_info.get("logprob_H"),
                        "logprob_new": guard_info.get("logprob_new"),
                        "rescue_logprob_penalty": float(guard_info["rescue_logprob_penalty"]),
                        "site_entropy": guard_info.get("site_entropy"),
                        "topk_candidates_at_repaired_site": guard_info.get("topk_candidates_at_repaired_site", []),
                        "final_species": sample_summary["decoded_species"],
                        "final_score": float(guard_info["final_score"]),
                        "single_element_before_guard": bool(len(raw_counts) == 1 and n_real > 0),
                        "single_element_after_guard": bool(len(final_counts) == 1 and n_real > 0),
                    },
                )

            if sample_summary["class0_on_valid_nodes"] > 0:
                warning_payload = {
                    "event": "class0_on_valid_nodes",
                    "source_tag": source_tag,
                    "round_index": int(round_index),
                    "sample_local_index": int(b),
                    "sample_global_index": sample_global_index,
                    "class0_count": int(sample_summary["class0_on_valid_nodes"]),
                    "decoded_class_ids": [int(v) for v in final_idx.detach().cpu().tolist()],
                    "decoded_species": [self._class_index_to_symbol(v) for v in final_idx.detach().cpu().tolist()],
                }
                self._write_atom_debug_line("atom_type_warnings.jsonl", warning_payload)
                print(
                    f"[AtomTypeDebug] warning: class 0 reached valid nodes at global sample index "
                    f"{sample_global_index} ({source_tag}, round={round_index}, batch={b})"
                )

            if all_h:
                all_h_global_indices.append(sample_global_index)
                print(
                    f"[AtomTypeDebug] all-H sample detected at global sample index "
                    f"{sample_global_index} ({source_tag}, round={round_index}, batch={b})"
                )
                if self.debug_atom_types and self.debug_atom_dir:
                    torch.save(
                        {
                            "source_tag": source_tag,
                            "round_index": int(round_index),
                            "sample_local_index": int(b),
                            "sample_global_index": sample_global_index,
                            "logits_valid": valid_logits.detach().cpu(),
                            "probs_valid": valid_masked_probs.detach().cpu(),
                            "raw_argmax_valid": raw_idx.detach().cpu(),
                            "masked_argmax_valid": masked_idx.detach().cpu(),
                            "raw_decoded_valid": raw_decoded_idx.detach().cpu(),
                            "decoded_valid": final_idx.detach().cpu(),
                            "raw_score": float(guard_info["raw_score"]),
                            "final_score": float(guard_info["final_score"]),
                            "all_h_guard_repaired": bool(guard_info["repaired"]),
                            "all_h_guard_replacements": guard_info["replacements"],
                            "node_mask": node_mask[b].detach().cpu(),
                            "rl": rl[b].detach().cpu() if rl is not None else None,
                            "ra": ra[b].detach().cpu() if ra is not None else None,
                        },
                        os.path.join(
                            self.debug_atom_dir,
                            f"all_h_sample_round_{int(round_index)}_batch_{int(b)}.pt",
                        ),
                    )

        if self.debug_atom_types:
            self._write_atom_debug_line("atom_type_batches.jsonl", batch_summary)
        if all_class_constant:
            self._write_atom_debug_line(
                "atom_type_warnings.jsonl",
                {
                    "event": "all_valid_nodes_class_constant_logits",
                    "source_tag": source_tag,
                    "round_index": int(round_index),
                    "logits": self._tensor_stats(logits),
                },
            )
        if all_h_global_indices:
            self._write_atom_debug_line(
                "all_h_samples.jsonl",
                {
                    "event": "all_h_batch",
                    "source_tag": source_tag,
                    "round_index": int(round_index),
                    "all_h_sample_indices": all_h_global_indices,
                },
            )
        batch_summary["raw_all_H_count"] = int(raw_all_h_count)
        batch_summary["final_all_H_count"] = int(final_all_h_count)
        batch_summary["num_all_H_rescued"] = int(num_all_h_rescued)
        batch_summary["single_element_count_before_guard"] = int(single_element_count_before_guard)
        batch_summary["single_element_count_after_guard"] = int(single_element_count_after_guard)

        summary = self._atom_type_all_h_guard_summary
        summary["guard_decode_call_count"] += int(B)
        summary["guard_trigger_count"] += int(raw_all_h_count)
        summary["guard_repair_success_count"] += int(num_all_h_rescued)
        summary["total_samples"] += int(B)
        summary["raw_all_H_count"] += int(raw_all_h_count)
        summary["final_all_H_count"] += int(final_all_h_count)
        summary["num_all_H_rescued"] += int(num_all_h_rescued)
        summary["single_element_count_before_guard"] += int(single_element_count_before_guard)
        summary["single_element_count_after_guard"] += int(single_element_count_after_guard)
        for sample_payload in batch_summary["samples"]:
            penalty = float(sample_payload["all_h_guard_rescue_logprob_penalty"])
            if sample_payload["all_h_guard_repaired"]:
                summary["rescue_logprob_penalties"].append(penalty)
                for replacement in sample_payload["all_h_guard_replacements"]:
                    summary["rescued_new_element_distribution"][replacement["new_symbol"]] += 1
        self._write_atom_type_all_h_guard_summary()

        h_cat = F.one_hot(elem_idx, self.num_classes) * node_mask
        assert h_cat.shape == logits.shape, \
            f"decoded one-hot shape mismatch: expected {tuple(logits.shape)}, got {tuple(h_cat.shape)}"
        self._assert_no_all_h_in_one_hot_batch(
            h_cat,
            node_mask,
            source_tag=source_tag,
            round_index=round_index,
            stage="finalize_atom_type_logits_return",
        )
        return h_cat

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')
    
    def check_sigma_max_too_small(
        self,
        num_stdevs: int = 8,
        min_coverage: float = 1.0,
        require_gt1: bool = True,
        max_offset_integer: int = 3,
    ):
        """
        Check whether sigma_max (sigma at t=1) is too small for VE-SDE sampling.

        This function is intended for VE-native schedules where:
            gamma(t) = log sigma(t)^2
            sigma(t) = exp(0.5 * gamma(t))

        Parameters
        ----------
        num_stdevs : int
            How many standard deviations define "effective coverage".
            Typical values: 6~10 (default: 8).

        min_coverage : float
            Required minimum coverage relative to normalized unit length (1.0).
            Condition:
                sigma_max * num_stdevs >= min_coverage
            - 1.0  : barely covers one unit interval
            - 5.0+ : strong exploration / flat prior

        require_gt1 : bool
            If True, also require sigma_max > 1.0 (coarse but very useful VE sanity check).

        max_offset_integer : int
            max_offset_integer used in wrapped_normal_score_batch.
            Ensures sigma_max is large enough to mix across periodic images.
        """
        device = self.gamma.gamma.device
        ones  = torch.ones((1,1), device=device)
        # zeros = torch.zeros((1,1), device=device)

        # sigma(t=1)
        gamma_1 = self.gamma(ones)
        sigma_1 = self.sigma(gamma_1, target_tensor=ones).item()

        # -------------------------
        # (A) Coverage check
        # -------------------------
        if sigma_1 * num_stdevs < min_coverage:
            raise ValueError(
                f'sigma_max too small: sigma(t=1)={sigma_1:.5g}. '
                f'sigma_max * num_stdevs = {sigma_1 * num_stdevs:.5g} '
                f'< min_coverage={min_coverage}. '
                f'Increase sigma_max in VE schedule.'
            )

        # -------------------------
        # (B) Simple VE sanity
        # -------------------------
        if require_gt1 and sigma_1 <= 1.0:
            raise ValueError(
                f'sigma_max too small for VE sampling: sigma(t=1)={sigma_1:.5g} <= 1. '
                f'For VE-native crystal generation, sigma_max is typically > 1 '
                f'(often 10–50).'
            )

        # -------------------------
        # (C) Periodic mixing heuristic
        # -------------------------
        # Weak condition: sigma_max should not be far smaller than the wrapping scale
        # implied by the number of periodic offsets.
        sigma_mix_min = 1.0 / (2.0 * max_offset_integer)
        if sigma_1 < sigma_mix_min:
            raise ValueError(
                f'sigma_max too small for periodic mixing: '
                f'sigma(t=1)={sigma_1:.5g} < {sigma_mix_min:.5g} '
                f'(max_offset_integer={max_offset_integer}). '
                f'Increase sigma_max or max_offset_integer.'
            )


    def phi(self, zxh, t, node_mask, edge_mask, context, t2=None, mask_y=None, rl=None, ra=None,
            adjust_type=False, atom_type_state=None):
        """score funtion predict network"""   
        
        # prepare inputs
        zx = zxh[:, :, :self.n_dims]  # [B, N, 3]
        B, N = zx.size(0), zx.size(1)
        input_atom_logits = atom_type_state
        if adjust_type:
            h_cat_pred = zxh[:, :, self.n_dims:self.n_dims+self.num_classes].clone()  # [B, N, num_classes]
            h_cat_pred[:, :, self.unknown_atom_type_idx] = -1e9
            input_atom_logits = h_cat_pred

        # for example, x: R -> [0,1]
        if rl is not None and ra is not None:
            rl, ra = self.phi_unnormalize_la(rl, ra)
        else:
            print("No valid lengths and angles provided for unnormalization in phi.")
            raise ValueError
        zx, atom_types, natoms, rl, ra, batch, previous_atom_logits_flat = self.prepare_inputs_for_equiformer(
            t, zx, rl, ra, node_mask, atom_type_state=input_atom_logits
        )

        # forward for x, h
        net_outs = self.dynamics(t, zx, atom_types, natoms, \
                lengths=rl, angles=ra, batch=batch, previous_atom_logits=previous_atom_logits_flat)
        self._last_dynamics_atom_debug_info = dict(getattr(self.dynamics, "last_atom_debug_info", {}))
        
        # outputs reshape
        net_eps_x, net_pred_h = self.reshape_outputs(
            net_outs, B, N, node_mask, natoms, batch, adjust_type)
        # normalize, cause dynamics works on physical space, so its output is unnormalized
        net_pred_h = self.phi_normalize_h(net_pred_h, node_mask)
        net_eps_xh = torch.cat([net_eps_x, net_pred_h], dim=2) # [B, N, 3 + num_classes]

        if self.property_pred and self.use_prop_pred:
            property_pred = net_outs['property_pred']
            return (net_eps_xh, property_pred)

        return net_eps_xh
    

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    # def sigma(self, gamma, target_tensor):
    #     """Computes sigma given gamma."""
    #     return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)
    
    def sigma(self, gamma, target_tensor):
        """
        Computes sigma given gamma.

        VP mode (legacy VDM):
            gamma = log(sigma_vp^2 / alpha_vp^2)
            sigma_vp^2 = sigmoid(gamma)
            sigma_vp   = sqrt(sigmoid(gamma))

        VE mode (VE-native):
            gamma = log(sigma_ve^2)
            sigma_ve = exp(0.5 * gamma)
        """
        if self.sde_type == "ve":
            sigma = torch.exp(0.5 * torch.clamp(gamma, -30.0, 30.0))
            return self.inflate_batch_array(sigma, target_tensor)
        else:
            return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)


    # def alpha(self, gamma, target_tensor):
    #     """Computes alpha given gamma."""
    #     return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    
    def alpha(self, gamma, target_tensor):
        """
        Computes alpha given gamma.

        VP mode:
            alpha_vp^2 = sigmoid(-gamma)
            alpha_vp   = sqrt(sigmoid(-gamma))

        VE mode:
            VE has no alpha (no shrink). For compatibility, return 1.
        """
        if self.sde_type == "ve":
            ones = torch.ones_like(gamma)
            return self.inflate_batch_array(ones, target_tensor)
        else:
            return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    # def SNR(self, gamma):
    #     """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    #     return torch.exp(-gamma)
    
    def SNR(self, gamma):
        """
        Computes signal-to-noise ratio.

        VP mode:
            SNR = alpha^2 / sigma^2 = exp(-gamma)

        VE mode:
            There is no alpha. If you still want a monotonically decreasing proxy:
                SNR_proxy = 1 / sigma_ve^2 = exp(-gamma)
            which is consistent if gamma = log(sigma_ve^2).
        """
        if self.sde_type == "ve":
            # proxy SNR: 1 / sigma^2
            return torch.exp(-torch.clamp(gamma, -30.0, 30.0))
        else:
            return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims
    
    def normalize_frac_pos_with_h(self, f, h, node_mask):
        f = f * node_mask
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return f, h, delta_log_px
    

    def phi_normalize_h(self, h, node_mask=None):
        if node_mask is not None:
            h = (h.float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        else :
            h = (h.float() - self.norm_biases[1]) / self.norm_values[1]

        return h

    
    def phi_unnormalize_h_cat(self, h_cat, node_mask):
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        return h_cat


    def phi_unnormalize_la(self, l, a):
        l = l * self.norm_values[3] + self.norm_biases[3]
        a = a * self.norm_values[4] + self.norm_biases[4]
        return l, a


    def phi_normalize_la(self, l, a):
        l = (l - self.norm_biases[3]) / self.norm_values[3]
        a = (a - self.norm_biases[4]) / self.norm_values[4]
        return l, a


    def normalize_lengths_angles(self, lengths, angles):
        lengths_norm = (lengths - self.norm_biases[3]) / self.norm_values[3]
        angles_norm = (angles - self.norm_biases[4]) / self.norm_values[4]
        delta_log_pl = -self.len_dim * np.log(self.norm_values[3])
        delta_log_pa = -self.angle_dim * np.log(self.norm_values[4])
        return lengths_norm, angles_norm, delta_log_pl, delta_log_pa
    
    
    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * node_mask
        x = wrap_at_boundary(x, wrapping_boundary=1.0)  # mod 1s

        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_lengths_angles(self, lengths, angles):
        lengths = lengths * self.norm_values[3] + self.norm_biases[3]
        angles = angles * self.norm_values[4] + self.norm_biases[4]
        return lengths, angles


    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


    def sample_normal(self, mu, sigma, node_mask, fix_noise=False, only_coord=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        if only_coord:
            eps = self.sample_combined_position_noise(bs, mu.size(1), node_mask)
        else:
            eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        if eps.shape[2] != mu.shape[2]:
            print("eps: ", eps.shape)
            print("mu: ", mu.shape)
        return mu + sigma * eps
    
    def _gamma_table(self):
        return self.gamma.gamma  # PredefinedNoiseSchedule's tensor [T_sched]

    def interp_gamma(self, t01: torch.Tensor) -> torch.Tensor:
        """t01: [B,1] in [0,1] -> gamma: [B,1] via linear interpolation."""
        tab = self._gamma_table()
        T_sched = int(tab.shape[0])

        t01 = t01.clamp(0.0, 1.0)
        u = t01 * (T_sched - 1)
        i0 = torch.floor(u).long()
        i1 = torch.clamp(i0 + 1, max=T_sched - 1)
        w = (u - i0.float()).clamp(0.0, 1.0)

        i0f = i0.view(-1)
        i1f = i1.view(-1)
        g0 = tab[i0f].view_as(t01)
        g1 = tab[i1f].view_as(t01)
        return g0 + (g1 - g0) * w

    def sigma_ve(self, t01: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """return sigma_ve(t) inflated to [B,1,1] for broadcasting with x."""
        gamma_t = self.interp_gamma(t01)
        gamma_t = torch.clamp(gamma_t, -30.0, 30.0)
        sigma = torch.exp(0.5 * gamma_t)  # [B,1]
        return self.inflate_batch_array(sigma, target_tensor)

    def g2_ve(self, t01: torch.Tensor, x: torch.Tensor, eps: float = None) -> torch.Tensor:
        """return g(t)^2 = d/dt sigma^2(t), inflated to [B,1,1]."""
        tab = self._gamma_table()
        T_sched = int(tab.shape[0])
        if eps is None:
            eps = 1.0 / (T_sched - 1)

        t_plus  = (t01 + eps).clamp(0.0, 1.0)
        t_minus = (t01 - eps).clamp(0.0, 1.0)

        gp = torch.clamp(self.interp_gamma(t_plus),  -30.0, 30.0)
        gm = torch.clamp(self.interp_gamma(t_minus), -30.0, 30.0)

        sigma2_p = torch.exp(gp)
        sigma2_m = torch.exp(gm)

        dsigma2_dt = (sigma2_p - sigma2_m) / (t_plus - t_minus + 1e-12)
        g2 = torch.clamp(dsigma2_dt, min=1e-20)
        return self.inflate_batch_array(g2, x)  # [B,1,1]

    
    def forward(self, *args, **kwargs):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        # 如果 DataParallel 传进来的是一个 tuple/list
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        # 解包参数
        frac_pos, h, lengths, angles, node_mask, edge_mask, context = args[:7]
        mask_indicator = kwargs.get("mask_indicator", None)
        expand_diff = kwargs.get("expand_diff", False)
        property_label = kwargs.get("property_label", None)
        bond_info = kwargs.get("bond_info", None)

        if self.property_pred:
            assert property_label is not None, "property_label should not be None in training"

        # fractional space 
        frac_pos, h, _ = self.normalize_frac_pos_with_h(frac_pos, h, node_mask)
        lengths, angles, _, _ = self.normalize_lengths_angles(lengths, angles)
        
        if self.sde_type == "ve":
            loss, loss_dict = self.compute_loss_score_ve(frac_pos, h, lengths, angles, node_mask, edge_mask, context, t0_always=False,
                                                      property_label=property_label)
        else: # "vp"
            loss, loss_dict = self.compute_loss_score(frac_pos, h, lengths, angles, node_mask, edge_mask, context, t0_always=False,
                                                      property_label=property_label)

        return loss, loss_dict
    

    def compute_loss_score(self, x, h, lengths, angles, node_mask, edge_mask, 
                           context, t0_always, time_upperbond=-1, property_label=None,):
        batch_size = x.size(0)
        # compute atomic counts per structure
        N = node_mask.squeeze(-1).sum(-1)  # [B]
        # scale factor for score loss
        volume = lattice_volume(lengths, angles) # [B]
        scale = (volume / (N + 1e-8)).pow(2/3)

        # whether to include loss term 0 always.
        if t0_always:
            lowest_t = 1
        else:
            lowest_t = 0

        # sample t, t_int: [B, 1]
        t_int = torch.randint(
                lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()
        if time_upperbond >= 0:
            t_int = torch.ones_like(t_int) * time_upperbond
        if self.half_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, 
                    size=(batch_size - half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(
                lowest_t, self.T + 1, 
                size=(half_batch_size, 1), device=x.device).float()
        if self.sep_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, 
                    size=(batch_size - half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(
                self.prediction_threshold_t + 1, self.T + 1, 
                size=(half_batch_size, 1), device=x.device).float()
        
        s_int = t_int - 1 
        t_is_zero = (t_int == 0).float()  # to compute log p(x | z0).

        # Normalize t and s to [0, 1]
        s = s_int / self.T
        t = t_int / self.T
        dt = self.inflate_batch_array(s-t, x)  # [B, 1, 1], for repulsion loss, negative

        # Compute gamma_s and gamma_t
        # gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        # print("sigma_t min/max: ", sigma_t.min().item(), sigma_t.max().item()) # ~0.1, ~0.9

        # compute score function
        mean_t = alpha_t * x
        variance_t = sigma_t.pow(2).expand(-1, x.shape[1], -1) # -1: 该维度保持原来的大小
        wrapped_score = self.wrapped_normal_score_batch(
            x, mean_t, variance_t, node_mask, 
            wrapping_boundary=1.0, max_offset_integer=3
            )   # [B,N,3]
        target = wrapped_score * sigma_t # predict score scaled by sigma_t, 为保持各时间步的loss数值稳定

        # use net to predict score
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)
        fix_h = torch.ones_like(torch.cat([h['categorical'], h['integer']], dim=2))
        z_t = alpha_t * x + sigma_t * eps
        z_t = wrap_at_boundary(z_t, wrapping_boundary=1.0) # wrap, mod 1
        z_t = torch.cat([z_t, fix_h], dim=2)

        # score-matching model
        net_out = self.phi(z_t, t, node_mask, edge_mask, context, rl=lengths, ra=angles)
        pred = net_out[:, :, :self.n_dims] # [B,N,3]

        # compute the l2 score loss         
        delta = (target - pred) * node_mask # [B,N,3]
        denom = node_mask.squeeze(-1).sum(-1) * 3 # [B]
        score_loss = sum_except_batch(delta.square()) / denom # per atom node in one sample
        score_loss = score_loss * scale

        # total loss
        kl_prior = torch.zeros_like(score_loss)
        loss = kl_prior + score_loss
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        loss_dict = {'t': t_int.squeeze(),
                     'loss': loss.squeeze(), 
                     'x_error':(score_loss / sigma_t).squeeze()}

        # ---------------------------------------------------------------------------------
        # predict physical properties below certain t
        # ---------------------------------------------------------------------------------

        # 1) calculate the loss for repulsion term
        h_pred = net_out[:, :, 3:]          # [B,N,C]
        score_pred = pred / (sigma_t + 1e-8)  # [B,N,3], unscaled score prediction
        len_scale = (volume / (N + 1e-8)).pow(1/3).view(batch_size,1,1)  # [B,1,1]
        zx_s = self.reverse_sde_step_given_pred_training(
            z_t[:, :, :3], t, dt, node_mask, score_pred, len_scale
        )
        L = self.compute_lattice_matrix(
            *self.unnormalize_lengths_angles(lengths, angles))  # [B,3,3]
        repulsion_loss = self.zbl_repulsion_loss(
            zx_s, L, h_pred, node_mask, t_int,
            prediction_threshold_t=self.prediction_threshold_t, min_dist=1.5)
        repulsion_loss = self.lambda_rep * repulsion_loss
        loss_dict["repulsion_loss"] = repulsion_loss
        loss += repulsion_loss

        # 2) calculate the loss for atom type
        h_true = torch.cat([h['categorical'], h['integer']], 
                            dim=2).clone().detach().requires_grad_(True).to(torch.float32).to(x.device)
        h_true_idx = h_true.argmax(dim=2)   # [B,N]
        # cross_entropy loss, purely for atom type prediction
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        atom_type_loss = ce_loss(
            h_pred.reshape(-1, h_pred.size(-1)), # logits, [B*N, C]
            h_true_idx.reshape(-1) # targets, [B*N]
        )
        atom_type_loss = atom_type_loss.reshape(batch_size, -1)  # [B, N]
        atom_type_loss = atom_type_loss * node_mask.squeeze(-1)
        atom_type_loss = atom_type_loss.sum(dim=1) / node_mask.squeeze(-1).sum(dim=1).clamp(min=1)
        # mask the loss term with t > prediction_threshold_t
        pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
        pred_loss_mask = pred_loss_mask.squeeze(1)
        atom_type_loss = self.lambda_type * (atom_type_loss * pred_loss_mask)
        loss_dict["atom_type_loss"] = atom_type_loss
        loss += atom_type_loss

        # 3) additional adjustment for atom type prediction during diffusion
        if self.adjust_atom_type:
            print("adjust atom type failed to be used currently.")

        return loss, loss_dict
    

    def compute_loss_score_ve(self, x, h, lengths, angles, node_mask, edge_mask,
                       context, t0_always, time_upperbond=-1, property_label=None):
        batch_size = x.size(0)
        device = x.device

        # compute atomic counts per structure
        N = node_mask.squeeze(-1).sum(-1)  # [B]
        volume = lattice_volume(lengths, angles)  # [B]
        scale = (volume / (N + 1e-8)).pow(2/3)

        # whether to include loss term 0 always.
        lowest_t = 1 if t0_always else 0

        # sample t_int: [B,1] in {lowest_t,...,T}
        t_int = torch.randint(lowest_t, self.T + 1, size=(batch_size, 1), device=device).float()
        if time_upperbond >= 0:
            t_int = torch.ones_like(t_int) * time_upperbond

        if self.half_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:, :] = torch.randint(
                lowest_t, self.prediction_threshold_t + 1,
                size=(batch_size - half_batch_size, 1), device=device
            ).float()
            t_int[:half_batch_size, :] = torch.randint(
                lowest_t, self.T + 1,
                size=(half_batch_size, 1), device=device
            ).float()

        if self.sep_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:, :] = torch.randint(
                lowest_t, self.prediction_threshold_t + 1,
                size=(batch_size - half_batch_size, 1), device=device
            ).float()
            t_int[:half_batch_size, :] = torch.randint(
                self.prediction_threshold_t + 1, self.T + 1,
                size=(half_batch_size, 1), device=device
            ).float()

        # normalize t to [0,1]
        s_int = t_int - 1 
        t = t_int / self.T  # [B,1]
        s = s_int / self.T
        dt = self.inflate_batch_array(s-t, x)  # [B, 1, 1], for repulsion loss, negative

        # ---- VE sigma(t) ----
        sigma_t = self.sigma_ve(t, x)  # [B,1,1]

        # sample eps in position space
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask
        )  # [B,N,3]

        # ---- VE perturbation: z = x + sigma eps ----
        z_pos = x + sigma_t * eps # note: z_pos = x + sqrt(sigma_t^2-sigma_0^2) * eps is standard VE formula
        z_pos = wrap_at_boundary(z_pos, wrapping_boundary=1.0)

        # ---- VE target score on torus: score wrt z_pos, mean=x, var=sigma^2 ----
        mean_t = x  # [B,N,3]
        variance_t = sigma_t.pow(2).expand(-1, x.shape[1], -1)  # [B,N,1]
        variance_t = torch.clamp(variance_t, min=1e-12)         # stability for small sigma

        wrapped_score = self.wrapped_normal_score_batch(
            z_pos, mean_t, variance_t, node_mask,
            wrapping_boundary=1.0, max_offset_integer=3
        )  # [B,N,3]

        # keep your stabilization: predict sigma * score
        target = wrapped_score * sigma_t  # [B,N,3]

        # build network input exactly like before
        fix_h = torch.ones_like(torch.cat([h['categorical'], h['integer']], dim=2))
        z_t = torch.cat([z_pos, fix_h], dim=2)

        net_out = self.phi(z_t, t, node_mask, edge_mask, context, rl=lengths, ra=angles)
        pred = net_out[:, :, :self.n_dims]  # [B,N,3]  ~ sigma*score

        # l2 loss per atom
        delta = (target - pred) * node_mask
        denom = node_mask.squeeze(-1).sum(-1) * 3  # [B]
        score_loss = sum_except_batch(delta.square()) / denom
        score_loss = score_loss * scale

        loss = score_loss  # no KL prior term here (same as your current)
        assert len(loss.shape) == 1

        # logging: estimate avg |score error| magnitude (rough)
        sigma_scalar = torch.clamp(sigma_t.view(batch_size, -1).mean(dim=1), min=1e-12)
        x_error = (score_loss / sigma_scalar)

        loss_dict = {
            't': t_int.squeeze(),
            'loss': loss.squeeze(),
            'x_error': x_error.squeeze(),
        }

        # ---------------------------------------------------------------------------------
        # predict physical properties below certain t
        # ---------------------------------------------------------------------------------

        # 1) calculate the loss for repulsion term
        h_pred = net_out[:, :, 3:]          # [B,N,C]
        score_pred = pred / (sigma_t + 1e-8)  # [B,N,3], unscaled score prediction
        len_scale = (volume / (N + 1e-8)).pow(1/3).view(batch_size,1,1)  # [B,1,1]
        zx_s = self.reverse_sde_step_given_pred_training(
            z_t[:, :, :3], t, dt, node_mask, score_pred, len_scale
        )
        L = self.compute_lattice_matrix(
            *self.unnormalize_lengths_angles(lengths, angles))  # [B,3,3]
        repulsion_loss = self.zbl_repulsion_loss(
            zx_s, L, h_pred, node_mask, t_int,
            prediction_threshold_t=self.prediction_threshold_t, min_dist=1.5)
        repulsion_loss = self.lambda_rep * repulsion_loss
        loss_dict["repulsion_loss"] = repulsion_loss
        loss += repulsion_loss

        # 2) calculate the loss for atom type
        h_true = torch.cat([h['categorical'], h['integer']], 
                            dim=2).clone().detach().requires_grad_(True).to(torch.float32).to(x.device)
        h_true_idx = h_true.argmax(dim=2)   # [B,N]
        # cross_entropy loss, purely for atom type prediction
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        atom_type_loss = ce_loss(
            h_pred.reshape(-1, h_pred.size(-1)), # logits, [B*N, C]
            h_true_idx.reshape(-1) # targets, [B*N]
        )
        atom_type_loss = atom_type_loss.reshape(batch_size, -1)  # [B, N]
        atom_type_loss = atom_type_loss * node_mask.squeeze(-1)
        atom_type_loss = atom_type_loss.sum(dim=1) / node_mask.squeeze(-1).sum(dim=1).clamp(min=1)
        # mask the loss term with t > prediction_threshold_t
        pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
        pred_loss_mask = pred_loss_mask.squeeze(1)
        atom_type_loss = self.lambda_type * (atom_type_loss * pred_loss_mask)
        loss_dict["atom_type_loss"] = atom_type_loss
        loss += atom_type_loss

        # 3) additional adjustment for atom type prediction during diffusion
        if self.adjust_atom_type:
            print("adjust atom type failed to be used currently.")

        return loss, loss_dict

    
    def compute_lattice_matrix(self, lengths, angles):
        """
        lengths: [B, 3] -> a, b, c
        angles:  [B, 3] -> alpha, beta, gamma in DEGREES
        return:  [B, 3, 3] lattice matrix L where columns are v1, v2, v3
        """
        # Convert degrees → radians
        angles = torch.deg2rad(angles)
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]

        cosA = torch.cos(alpha)
        cosB = torch.cos(beta)
        cosG = torch.cos(gamma)
        sinG = torch.sin(gamma)

        # v1 = (a, 0, 0)
        v1 = torch.stack([
            a,
            torch.zeros_like(a),
            torch.zeros_like(a),
        ], dim=1)  # [B,3]

        # v2 = (b*cosγ, b*sinγ, 0)
        v2 = torch.stack([
            b * cosG,
            b * sinG,
            torch.zeros_like(b),
        ], dim=1)

        # v3 computed using triclinic formula
        cx = c * cosB
        cy = c * (cosA - cosB * cosG) / sinG
        cz = torch.sqrt(c**2 - cx**2 - cy**2)

        v3 = torch.stack([cx, cy, cz], dim=1)

        # Columns form the lattice matrix
        L = torch.stack([v1, v2, v3], dim=2)  # [B,3,3]
        return L


    def compute_repulsion_loss_from_fractional(
        self,
        x_hat,          # [B,N,3], fractional coords
        L,              # [B,3,3], lattice matrix
        node_mask,      # [B,N,1]
        t_int,          # [B]
        cutoff=0.5,
        t_threshold=10,
        k=10.0
    ):
        """
        Compute repulsion loss using Cartesian distances obtained from fractional coords.
        No PBC is applied (use nearest cell).
        Returns [B] loss.
        """

        B, N, _ = x_hat.shape
        nm = node_mask.float()  # [B,N,1]
        x_hat = x_hat.float()  # Mask out invalid nodes

        # 1. Fractional -> Cartesian
        X = torch.matmul(x_hat, L)  # [B,N,3]

        # 2. Pairwise distances
        Xi = X.unsqueeze(2)  # [B,N,1,3]
        Xj = X.unsqueeze(1)  # [B,1,N,3]
        dX = Xi - Xj         # [B,N,N,3]

        dist = torch.norm(dX, dim=-1)  # [B,N,N]

        # 3. Mask out invalid nodes
        pair_mask = (nm * nm.transpose(1, 2)).squeeze(-1)  # [B,N,N]

        # Remove self-distances
        eye = torch.eye(N, device=x_hat.device).unsqueeze(0)
        pair_mask = pair_mask * (1 - eye)

        # 4. Compute repulsion
        rep = torch.expm1(k * torch.relu(cutoff - dist)) # [B,N,N], exp()-1
        rep = rep * pair_mask

        # Normalize by number of valid pairs
        denom = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
        rep_loss = rep.sum(dim=(1, 2)) / denom  # [B]

        # 5. Only apply when t small
        t_mask = (t_int.view(-1) <= t_threshold).float()
        rep_loss = rep_loss * t_mask  # [B]

        return rep_loss


    def zbl_repulsion_loss(
        self,
        x_hat,          # [B,N,3], fractional in [0,1]
        L,              # [B,3,3]
        h_pred,         # [B,N,C], logits for atom types
        node_mask,      # [B,N,1]
        t_int,          # [B,1]
        prediction_threshold_t=500,
        min_dist=0.8,   # Å
    ):

        B, N, _ = x_hat.shape
        device = x_hat.device

        # ---------------------------------------------------------
        # 1. 直接使用 argmax 类型作为真实原子序数 Z
        # ---------------------------------------------------------
        Z = h_pred.argmax(dim=-1).float()     # [B,N]
        x_hat = x_hat.float()                 # [B,N,3]
        Z_i = Z.unsqueeze(2)                  # [B,N,1]
        Z_j = Z.unsqueeze(1)                  # [B,1,N]

        Zij = Z_i * Z_j                       # [B,N,N]

        # ---------------------------------------------------------
        # 2. 分数坐标差 + 最小镜像
        # ---------------------------------------------------------
        dx = x_hat.unsqueeze(2) - x_hat.unsqueeze(1)  # [B,N,N,3]
        dx = dx - torch.round(dx) # wrap in fractional
        dx_flat = dx.reshape(B, N*N, 3)
        rij = torch.matmul(dx_flat, L).reshape(B, N, N, 3) # [B,N,N,3] cartesian
        dist = torch.norm(rij, dim=-1) + 1e-12       # [B,N,N]

        # ---------------------------------------------------------
        # 3.  ZBL potential V(r)
        # ---------------------------------------------------------
        # screening length a
        a0 = 0.529 # Bohr radius (Å)
        a = 0.8854 * a0 / (Zij.sqrt() + 1e-12)   # [B,N,N]

        # screening function φ(r/a)
        c = torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], device=device)
        d = torch.tensor([3.2,    0.9423,  0.4029, 0.2016], device=device)

        x = dist / (a + 1e-12)
        phi = (c[0] * torch.exp(-d[0] * x) +
            c[1] * torch.exp(-d[1] * x) +
            c[2] * torch.exp(-d[2] * x) +
            c[3] * torch.exp(-d[3] * x))

        ke = 14.3996  # eV·Å/e^2
        V = ke * Zij * phi / dist   # [B,N,N]

        # ---------------------------------------------------------
        # 4. 蒙特卡洛式 repulsion：只对近距离施加
        # ---------------------------------------------------------
        close_mask = (dist < min_dist).float()

        # 去掉 self-pairs
        eye = torch.eye(N, device=device).unsqueeze(0)
        close_mask = close_mask * (1 - eye)

        # node mask 使无效原子不参与
        nm = node_mask.squeeze(-1).float() # [B,N]
        pair_mask = nm.unsqueeze(1) * nm.unsqueeze(2)  # [B,N,N]

        mask = close_mask * pair_mask

        rep = (V * mask).sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + 1e-6)  # [B]
                
        # ---------------------------------------------------------
        # 5. 时间步 mask
        # ---------------------------------------------------------
        pred_mask = (t_int <= prediction_threshold_t).float().squeeze(1) # 噪声小时计算 repulsion loss
        rep = rep * pred_mask

        return rep


    @torch.no_grad()
    def sample_score(self, LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, context, 
               fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None):
        """Samples from the model using score function."""

        print('use LatticeGenModel to sample l and a, beginning...')
        rl, ra = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
        rl = rl.to(node_mask.device)
        ra = ra.to(node_mask.device)
        print('sample lengths and angles done.')

        if fix_noise:
            print("using fixed noise...")
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        print('sample T', self.T)
        print('use score function to sample x and h beginning...')

        for s in tqdm(reversed(range(0, self.T)), desc="Sampling diffusion steps"):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
                                    
            zx = z[:, :, :self.n_dims]
            z = self.sample_p_zs_given_zt_score(s_array, t_array, zx, rl, ra, node_mask, edge_mask, 
                                                context, fix_noise=fix_noise, pesudo_context=pesudo_context)                    
        
        # 采用 score-matching 后，不需要 sample_p_x_given_z0() 这一步
        print('sample x and h done.')
        
        # extract x, h
        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0).to(z.device)
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]

        # unnormalize x,h
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        # post-process h
        h_cat = self._finalize_atom_type_logits(
            h_cat,
            node_mask,
            round_index=0,
            source_tag="sample_score_legacy",
            rl=rl,
            ra=ra,
            apply_repair=False,
        )
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h, rl, ra
    

    def sample_p_zs_given_zt_score(self, s, t, zt, rl, ra, 
                             node_mask, edge_mask, context, 
                             fix_noise=False, yt=None, ys=None, force_t_zero=False, 
                             force_t2_zero=False, pesudo_context=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        alpha_s = self.alpha(gamma_s, target_tensor=zt)
        alpha_t = self.alpha(gamma_t, target_tensor=zt)

        # Neural net prediction.
        score = self.phi(zt, t, node_mask, edge_mask, context, rl=rl, ra=ra) / sigma_t # scale back to score with sigma_t

        # Compute mu for p(zs | zt).
        score_pos = score[:, :, :3]
        atom_type_pred = score[:, :, 3:]
        mu = zt / alpha_t_given_s + (sigma2_t_given_s / alpha_t_given_s) * score_pos # difference here

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
       
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # mod 1 to frac space
        zs[:, :, :self.n_dims] = wrap_at_boundary(zs[:, :, :self.n_dims], wrapping_boundary=1.0) # mod 1

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat([zs[:, :, :self.n_dims], atom_type_pred], dim=2)

        return zs


    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_frac_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        if self.atom_type_pred:
            return z_x
        return z
    

    def sample_combined_position_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_frac_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x


    @torch.no_grad()
    def sample(self, LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, context, 
               fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None, n_corrector_steps=1,
               num_rounds=1, seed_base=None, rl=None, ra=None, sample_realistic_LA=False, lambda_sym=0.1):
        """Samples from the model using score function."""    
        results = []
        self._reset_atom_type_all_h_guard_summary()

        for i in range(num_rounds):
            # ---------------------------------
            # Independent RNG state each round
            # ---------------------------------
            if seed_base is not None:
                seed = seed_base + i
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
            # ---------------------------------
            if sample_realistic_LA:
                assert rl is not None and ra is not None, "When sample_realistic_LA is True, rl and ra must be provided."
                x, h = self.sample_score_sde_Lattice(
                    rl, ra, n_samples, n_nodes, node_mask, edge_mask,
                    context, fix_noise, condition_generate_x,
                    annel_l, pesudo_context, n_corrector_steps,
                    round_index=i, lambda_sym=lambda_sym
                )
            else:
                # x, h, rl, ra = self.sample_score(LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, 
                #                     context, fix_noise, condition_generate_x, annel_l, pesudo_context)            
                x, h, rl, ra = self.sample_score_sde(
                    LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask,
                    context, fix_noise, condition_generate_x,
                    annel_l, pesudo_context,
                    n_corrector_steps, round_index=i, lambda_sym=lambda_sym
                )

            results.append((x, h, rl, ra, node_mask.clone()))

        return results
        

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info
    

    def prepare_inputs_for_equiformer(
        self,
        t,
        x,
        lengths,
        angles,
        node_mask,
        edge_mask=None,
        atom_type_state=None,
    ):
        """
        t:         [B, 1]  时间步 (如果需要可以传给 decoder)
        x:         [B, N, 3]  原子坐标 (分数坐标)
        lengths:   [B, 3]  晶格长度
        angles:    [B, 3]  晶格角度
        node_mask: [B, N, 1]  有效原子标记
        edge_mask: [B*N*N, 1] 可选，不一定需要，因为 EquiformerV2 可 on-the-fly 构图
        """

        B, N, _ = x.shape
        device = x.device

        # 1) 有效原子数 [B]
        natoms = node_mask.squeeze(-1).sum(dim=1).long()   # 每个晶体真实的原子个数，强制转为整数类型
        # 2) 展平后的坐标 [N_total, 3]
        pos = x[node_mask.squeeze(-1).bool()]       # 取出有效原子
        # x[mask]  ≡  x.view(-1)[mask.view(-1)]
        # 3) batch 索引 [N_total]
        batch = torch.arange(B, device=device).repeat_interleave(natoms)
        valid_mask = node_mask.squeeze(-1).bool()
        previous_atom_logits_flat = None
        used_atom_type_state = atom_type_state is not None

        if atom_type_state is not None:
            assert atom_type_state.shape == (B, N, self.num_classes), \
                f"atom_type_state must be [B, N, C], got {tuple(atom_type_state.shape)}"
            assert torch.isfinite(atom_type_state).all(), "atom_type_state contains NaN/Inf."
            masked_atom_type_state = atom_type_state.detach().clone()
            masked_atom_type_state[:, :, self.unknown_atom_type_idx] = -1e9
            atom_types_full = torch.argmax(masked_atom_type_state, dim=-1).long()
            atom_types = atom_types_full[valid_mask]
            previous_atom_logits_flat = masked_atom_type_state[valid_mask]
            if atom_types.numel() > 0 and atom_types.eq(self.unknown_atom_type_idx).any():
                self._write_atom_debug_line(
                    "atom_type_warnings.jsonl",
                    {
                        "event": "prepare_inputs_argmax_reached_unknown",
                        "unknown_atom_type_idx": int(self.unknown_atom_type_idx),
                        "atom_type_state_shape": list(atom_type_state.shape),
                    },
                )
        else:
            # Class 0 is the canonical unknown/pad token for sampling-time inputs.
            atom_types = torch.full_like(batch, fill_value=self.unknown_atom_type_idx)
        assert atom_types.shape == batch.shape, \
            f"flattened atom_types must match flattened batch, got {tuple(atom_types.shape)} vs {tuple(batch.shape)}"

        dummy_atom_type_idx = int(self.unknown_atom_type_idx)
        dummy_atom_type_symbol = self._class_index_to_symbol(dummy_atom_type_idx)
        dummy_atom_type_is_h = dummy_atom_type_symbol == "H"
        self._last_prepare_inputs_debug = {
            "used_atom_type_state": bool(used_atom_type_state),
            "atom_type_state_shape": list(atom_type_state.shape) if atom_type_state is not None else None,
            "dummy_atom_type_idx": dummy_atom_type_idx,
            "dummy_atom_type_symbol": dummy_atom_type_symbol,
            "dummy_atom_type_is_H": dummy_atom_type_is_h,
            "real_atom_count_total": int(valid_mask.sum().item()),
        }
        if atom_type_state is not None:
            self._last_prepare_inputs_debug["input_atom_type_unique"] = sorted(
                {int(v) for v in atom_types.detach().cpu().tolist()}
            )
        if self.debug_atom_types and (not self._input_atom_type_debug_written or dummy_atom_type_is_h):
            self._input_atom_type_debug_written = True
            payload = {
                "event": "prepare_inputs_for_equiformer",
                "used_atom_type_state": bool(used_atom_type_state),
                "dummy_atom_type_idx": dummy_atom_type_idx,
                "dummy_atom_type_symbol": dummy_atom_type_symbol,
                "dummy_atom_type_is_H": dummy_atom_type_is_h,
                "previous_atom_logits_available": previous_atom_logits_flat is not None,
            }
            if dummy_atom_type_is_h:
                payload["warning"] = "dummy atom type maps to H unexpectedly"
                print("[AtomTypeDebug] warning: prepare_inputs_for_equiformer dummy atom type still maps to H.")
            self._write_atom_debug_line("atom_type_session.jsonl", payload)

        return pos, atom_types, natoms, lengths, angles, batch, previous_atom_logits_flat


    
    def reshape_outputs(self, outs, B, N, node_mask, natoms, batch,  adjust_type=False):
        """
        outs: 模型输出字典，outs["coords"] 是 [Σ natoms, 3]
        B, N: batch_size, 最大原子数
        node_mask: [B, N, 1]
        natoms: [B] 每个样本的有效原子数
        batch: [Σ natoms] 每个原子所属的样本索引
        """
        device = node_mask.device
        pred_x = torch.zeros(B, N, 3, device=device, dtype=outs["coords"].dtype)

        for i in range(B):
            n_i = natoms[i]
            pred_x[i, :n_i] = outs["coords"][batch == i]

        atom_types_flat = outs["atom_types"]
        if adjust_type:
            atom_types_flat = outs["atom_types_adjust"]
        pred_atom_types = self.reshape_atom_types(atom_types_flat, node_mask, self.num_classes, batch, natoms)

        return pred_x, pred_atom_types


    def reshape_atom_types(self, atom_types_flat, node_mask, num_classes, batch=None, natoms=None):
        """
        atom_types_flat: [Σ natoms, H]
        node_mask: [B, N, 1]
        num_classes: H
        batch: [Σ natoms] 每个原子对应的样本索引
        natoms: [B] 每个样本有效原子数
        """
        B, N, _ = node_mask.shape
        H = num_classes
        device = atom_types_flat.device

        atom_types_full = torch.zeros(B, N, H, device=device, dtype=atom_types_flat.dtype)

        for i in range(B):
            n_i = natoms[i]
            atom_types_full[i, :n_i] = atom_types_flat[batch == i]

        return atom_types_full
    

    def wrapped_normal_score_batch(
        self,
        x,                    # [B, N, 3]
        mean,                 # [B, N, 3]
        variance_diag,        # [B, N] or [B, N, 1], sigma_t ** 2
        node_mask,            # [B, N, 1]
        wrapping_boundary: float = 1.0,
        max_offset_integer=3,
        ):
        B, N, D = x.shape
        device = x.device

        wrapping_boundary = wrapping_boundary * \
            torch.eye(x.shape[-1], device=device)[None].expand(B, -1, -1) # [B, 3, 3]

        # ---- 1. 构造 offsets: [B, K, 3] ----
        offsets = get_pbc_offsets(
            wrapping_boundary,  # [B,3,3]
            max_offset_integer
        )  # [B, K, 3]

        K = offsets.shape[1]

        # ---- 2. 构造 (x - mean): [B,N,1,3] ----
        diffs = (x - mean)[:, :, None, :]   # [B, N, 1, 3]

        # ---- 3. 加 offsets: [B,N,K,3] ----
        diffs_k = diffs + offsets[:, None, :, :]   # broadcasting to [B, N, K, 3]

        # ---- 4. 计算每个镜像的距离平方: [B,N,K] ----
        dists_sqr_k = diffs_k.pow(2).sum(dim=-1)   # [B,N,K]

        # ---- 5. 归一化权重 softmax ----
        # variance_diag: [B,N,1]
        if variance_diag.dim() == 2:
            variance_diag = variance_diag.unsqueeze(-1)

        weights = torch.softmax(
            -dists_sqr_k / (2 * variance_diag),   # broadcasting
            dim=-1
        )  # [B,N,K]

        # ---- 6. 计算 score ----
        score = -(weights[..., None] * diffs_k).sum(dim=2) / variance_diag  # [B,N,3]

        # ---- 7. mask 非原子 ----
        score = score * node_mask        # [B,N,3]

        return score

    
    def beta_from_alpha(self, t, x, eps=None):
        """
        Finite-difference approximation:
            beta(t) = -2 d/dt log(alpha(t))
        t: [B,1] normalized in [0,1]
        x: [B,N,3] to determine inflation shape
        """

        if eps is None:
            eps = 1.0 / self.T  # consistent with discrete schedule

        # ensure valid range
        t_plus  = (t + eps).clamp(0.0, 1.0)
        t_minus = (t - eps).clamp(0.0, 1.0)

        # lookup gamma(t)  -> [B,1]
        gamma_plus  = self.gamma(t_plus)
        gamma_minus = self.gamma(t_minus)

        # alpha = sqrt(sigmoid(-gamma))
        def alpha_from_gamma(g):
            return torch.sqrt(torch.sigmoid(-g))

        alpha_plus = alpha_from_gamma(gamma_plus) # [B,1]
        alpha_minus = alpha_from_gamma(gamma_minus) # [B,1]

        # central difference derivative of log(alpha)
        dlog_alpha = torch.log(alpha_plus) - torch.log(alpha_minus)
        dlog_alpha = dlog_alpha / (t_plus - t_minus + 1e-12)  # numeric safety

        beta = -2.0 * dlog_alpha

        beta = self.inflate_batch_array(beta, x)

        return beta

    
    def f_and_g(self, x, t):
        """
        Given current state x and time t, return drift f(x,t) and diffusion g(t)
        VP-SDE: f(x,t) = -0.5 * beta(t) * x
                g(t)   = sqrt(beta(t))
        """
        beta_t = self.beta_from_alpha(t, x)    # [B,1]

        # reshape to broadcast
        beta_t = beta_t.view(-1,1,1)        # [B,1,1]

        f = -0.5 * beta_t * x               # [B,N,3]
        g = torch.sqrt(beta_t)              # [B,1,1]
        
        return f, g
    
    @torch.no_grad()
    def reverse_sde_step(
        self,
        x,         # [B,N,3]
        t,         # scalar float (current time)
        dt,        # negative float (t_next - t)
        rl, ra,
        node_mask,
        edge_mask,
        context,
        len_scale=None,
        atom_type_state=None,
    ):  
        B, N, D = x.shape
        device = x.device

        # 1) prepare input
        t_tensor = torch.full((B,1), fill_value=t, device=device)

        # 2) model forward (must output score)
        net_out = self.phi(
            x, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra,
            atom_type_state=atom_type_state,
        )

        # 若模型预测的是乘以sigma_t的score，故转换为score需要除以sigma_t
        sigma_t = self.sigma(self.gamma(t_tensor), target_tensor=x)
        sigma_t = sigma_t.view(B,1,1)
        score = net_out[:, :, :3] / sigma_t
        score = score / len_scale # scale score according to length scale

        # 3) get drift+diffusion
        f, g = self.f_and_g(x, t_tensor)   # f=[B,N,3], g=[B,1,1]

        # 4) noise
        noise = torch.randn_like(x) * node_mask

        # 5) update
        x_next = (
            x
            + (f - (g*g)*score) * dt
            + g * (abs(dt)**0.5) * noise
        )

        # 6) wrap fractional coordinates
        x_next = torch.remainder(x_next, 1.0)
        z = torch.cat([x_next[:, :, :3], net_out[:, :, 3:]], dim=2)

        return z
    
    @torch.no_grad()
    def reverse_sde_step_all(
        self,
        zt,         # [B,N,3+C]
        t,         # scalar float (current time)
        dt,        # negative float (t_next - t)
        rl, ra,
        node_mask,
        edge_mask,
        context,
        model_out_is_eps=False,
        len_scale=None,
    ):  
        """adapted to full zt input including atom type feature"""
        B, N, _ = zt.shape
        device = zt.device
        x = zt[:, :, :3]  # [B,N,3]

        # 1) prepare input
        t_tensor = torch.full((B,1), fill_value=t, device=device)

        # 2) model forward (must output score)
        net_out = self.phi(
            zt, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra, adjust_type=True
        )

        # 若模型预测的是乘以sigma_t的score，故转换为score需要除以sigma_t
        sigma_t = self.sigma(self.gamma(t_tensor), target_tensor=x)
        sigma_t = sigma_t.view(B,1,1)
        score = net_out[:, :, :3] / sigma_t
        score = score / len_scale # scale score according to length scale

        # 3) get drift+diffusion
        f, g = self.f_and_g(x, t_tensor)   # f=[B,N,3], g=[B,1,1]

        # 4) noise
        noise = torch.randn_like(x) * node_mask

        # 5) update
        x_next = (
            x
            + (f - (g*g)*score) * dt
            + g * (abs(dt)**0.5) * noise
        )

        # 6) wrap fractional coordinates
        x_next = torch.remainder(x_next, 1.0)
        z = torch.cat([x_next[:, :, :3], net_out[:, :, 3:]], dim=2)

        return z

    def reverse_sde_step_given_pred_training(
        self,
        x,         # [B,N,3]
        t,         # scalar float (current time), [B,1]
        dt,        # negative float (t_next - t)
        node_mask,
        score_pred,   # [B,N,3], predicted score from training
        len_scale=1.0,
    ):  
        score = score_pred / len_scale # scale score according to length scale
        f, g = self.f_and_g(x, t)   # f=[B,N,3], g=[B,1,1]
        noise = torch.randn_like(x) * node_mask
        x_next = (
            x
            + (f - (g*g)*score) * dt
            + g * (abs(dt)**0.5) * noise
        )
        zx = wrap_at_boundary(x_next, wrapping_boundary=1.0) # mod 1

        return zx
    
    # VE-SDE
    def f_and_g_ve(self, x, t_tensor):
        """
        VE-SDE:
            dx = g(t) dW
        return f=[B,N,3]=0, g=[B,1,1]
        """
        g2 = self.g2_ve(t_tensor, x)  # [B,1,1]
        f = torch.zeros_like(x)
        g = torch.sqrt(torch.clamp(g2, min=1e-20))
        return f, g
    
    @torch.no_grad()
    def reverse_sde_step_ve(
        self,
        x,         # [B,N,3]
        t,         # scalar float
        dt,        # negative float
        rl, ra,
        node_mask,
        edge_mask,
        context,
        len_scale=None,
        atom_type_state=None,
    ):
        B, N, D = x.shape
        device = x.device
        t_tensor = torch.full((B, 1), fill_value=float(t), device=device)

        # sigma(t) for unscaling network output
        sigma_t = self.sigma_ve(t_tensor, x)  # [B,1,1]

        net_out = self.phi(
            x, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra,
            atom_type_state=atom_type_state,
        )
        pred_scaled_score = net_out[:, :, :3]                # ≈ sigma * score
        score = pred_scaled_score / torch.clamp(sigma_t, 1e-12)

        if len_scale is not None:
            score = score / len_scale

        f, g = self.f_and_g_ve(x, t_tensor)

        noise = torch.randn_like(x) * node_mask
        dt_abs = -dt if dt < 0 else dt  # should be positive

        # reverse VE update
        drift = (f - (g * g) * score) * node_mask
        x_next = x + drift * dt + g * (dt_abs ** 0.5) * noise

        x_next = torch.remainder(x_next, 1.0)
        z = torch.cat([x_next, net_out[:, :, 3:]], dim=2)
        return z

    
    @torch.no_grad()
    def sample_score_sde(
        self, LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, context, 
        fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None, 
        n_corrector_steps=1, snr=0.01, round_index=0, lambda_sym=0.1
    ):  
        # =======================================================
        # Sampling cell length/angles
        # =======================================================

        # rl, ra = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
        # rl = torch.abs(rl).to(node_mask.device)
        # ra = ra.to(node_mask.device)

        max_retry = 10
        for _ in range(max_retry):
            rl, ra = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
            rl = torch.abs(rl).to(node_mask.device)  # [B,3]
            ra = ra.to(node_mask.device)  # [B,3]
            valid = batch_valid_mask(rl, ra)
            # 如果全部合法，直接使用
            if valid.all():
                break
            # 否则局部重采样非法部分
            invalid_idx = (~valid).nonzero().flatten().tolist()
            print(f"Found {len(invalid_idx)} invalid cells, resampling ...")
            # 再采样一批
            rl_new, ra_new = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
            rl_new = rl_new.to(node_mask.device)
            ra_new = ra_new.to(node_mask.device)
            for i in invalid_idx:
                rl[i] = rl_new[i]
                ra[i] = ra_new[i]

        # 最后检查
        valid = batch_valid_mask(rl, ra)
        if not valid.all():
            print("Warning: some cells still invalid after retries, applying clamping fix.")
            # 强制修复 (可选)
            rl = rl.clamp(min=1.0)
            ra = ra.clamp(min=30.0, max=150.0)
        
        # =======================================================
        # Sample frac coordinates and atom types via score-based SDE
        # =======================================================

        volume = lattice_volume(rl, ra)     # [B]
        cell = self.compute_lattice_matrix(rl, ra)  # [B,3,3]
        N = node_mask.squeeze(-1).sum(-1)   # [B]        
        B = n_samples
        device = node_mask.device
        len_scale = (volume / (N + 1e-8)).pow(1/3).view(B,1,1)  # [B,1,1]

        if torch.isnan(len_scale).any():
            num_nan = torch.isnan(len_scale).sum().item()
            print(f"len_scale has {num_nan} NaN values! Replacing with 1.0 to avoid issues.")
            len_scale = torch.where(torch.isnan(len_scale), torch.ones_like(len_scale), len_scale)

        # 1) init Gaussian prior at t=1
        if fix_noise:
            z = self.sample_combined_position_feature_noise(
                1, n_nodes, node_mask
            ).expand(n_samples, -1, -1).clone()
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        print(f"> Reverse SDE steps = {self.T}")
        print('SDE type:', self.sde_type)
        print('Corrector steps:', n_corrector_steps)
        # 2) time grid, t in [1 → 0]
        t_grid = torch.linspace(1.0, 0.0, self.T+1).to(device)
        atom_type_state = None
        last_atom_state_updated_step_index = None
        final_window_logits = []
        final_window_step_indices = []

        for i in tqdm(range(self.T), desc="Sampling SDE steps"):
        # --- begin of for T steps
            if torch.isnan(z).any():
                print("NaN detected in z during sampling at step", i)

            t      = float(t_grid[i].item())
            t_next = float(t_grid[i+1].item())
            dt     = t_next - t               # negative
            step_index = i + 1
            atom_type_state_input_available = atom_type_state is not None
            input_state_updated_step_index = last_atom_state_updated_step_index

            # 只取前 3 维坐标
            zx = z[:, :, :3]
            if z.size(-1) >= self.n_dims + self.num_classes:
                atom_type_state = z[:, :, self.n_dims:self.n_dims+self.num_classes]
            t_tensor = torch.full((B,1), fill_value=t, device=device)
            t_next_tensor = torch.full((B,1), fill_value=t_next, device=device)

            # =======================================================
            # Corrector (Langevin)
            # =======================================================

            if self.sde_type == 've': # VE-SDE
                for _ in range(n_corrector_steps):
                    net_out = self.phi(
                        zx, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra,
                        atom_type_state=atom_type_state,
                    )
                    atom_type_state = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
                    sigma_t = self.sigma_ve(t_tensor, zx).view(B, 1, 1)  # [B,1,1]
                    score = net_out[:, :, :3] / torch.clamp(sigma_t, min=1e-12)
                    score = score / len_scale
                    noise = torch.randn_like(zx) * node_mask
                    grad_norm  = score.reshape(B, -1).norm(dim=-1)         # [B]
                    noise_norm = noise.reshape(B, -1).norm(dim=-1)         # [B]
                    # Song PC-style: eps = 2 * (snr * ||noise|| / ||grad||)^2
                    eps = 2.0 * (snr * noise_norm / (grad_norm + 1e-12))**2   # [B]
                    eps = torch.minimum(eps, (0.1 * sigma_t.squeeze())**2)   # 经验：每次 corrector 不要走超过 sigma 的某个比例
                    eps = eps.view(B, 1, 1)
                    zx = zx + eps * score + torch.sqrt(2.0 * eps) * noise
                    zx = torch.remainder(zx, 1.0)
            else: # VP-SDE
                for _ in range(n_corrector_steps):
                    # 网络前向 -> 得分
                    net_out = self.phi(
                        zx, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra,
                        atom_type_state=atom_type_state,
                    )
                    atom_type_state = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
                    gamma_t = self.gamma(t_tensor)
                    sigma_t = self.sigma(gamma_t, zx).view(B,1,1)
                    score = net_out[:, :, :3] / sigma_t
                    score = score / len_scale # scale score according to length scale
                    # 计算 alpha_t_pc, see SDE paper P23 Algorithm 3, 5
                    alpha_t_pc = self.alpha(gamma_t, zx) ** 2
                    noise = torch.randn_like(zx) * node_mask # 噪声
                    # Langevin 步
                    # SNR 根据 PC 论文设置
                    grad_norm = score.reshape(B, -1).norm(dim=-1) # [B]
                    noise_norm = noise.reshape(B, -1).norm(dim=-1) # [B]
                    eps = 2.0 * ((snr * noise_norm / (grad_norm + 1e-10))**2) * alpha_t_pc.squeeze() # [B]
                    eps = eps.view(B,1,1)
                    zx = zx + eps * score + torch.sqrt(2.0 * eps) * noise
                    zx = torch.remainder(zx, 1.0) # mod 1

            # =======================================================
            # Predictor (reverse SDE Euler-Maruyama)
            # =======================================================
            
            # 一次 SDE 反向步
            if self.sde_type == 've':
                z = self.reverse_sde_step_ve(
                    x=zx,
                    t=t,
                    dt=dt,
                    rl=rl, ra=ra,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    context=context,
                    len_scale=len_scale,
                    atom_type_state=atom_type_state,
                )
            else: # vp
                z = self.reverse_sde_step(
                    x=zx,
                    t=t,
                    dt=dt,
                    rl=rl, ra=ra,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    context=context,
                    len_scale=len_scale,
                    atom_type_state=atom_type_state,
                )
            if z.size(-1) >= self.n_dims + self.num_classes:
                atom_type_state = z[:, :, self.n_dims:self.n_dims+self.num_classes]
                last_atom_state_updated_step_index = step_index

            # =======================================================
            # Symmetry guidance
            # =======================================================
            
            # 示例：简单的反演对称性
            # lambda_sym 为引导强度系数
            if lambda_sym > 1e-6:
                current_space_group_ops = [
                    {
                        "R": -torch.eye(3),
                        "t": torch.zeros(3),
                    },
                    {
                        "R": torch.eye(3),
                        "t": torch.zeros(3),
                    }
                ]
                x_t = zx
                x_prev_standard = z[:, :, :3] 
                # 计算对称性引导梯度, 需要开启梯度追踪
                with torch.enable_grad():
                    x_in = x_t.detach().requires_grad_(True)
                    sym_grad = symmetry_guidance_gradient(x_in, cell, current_space_group_ops, 
                                                        scale=5.0, num_ops_sample=len(current_space_group_ops), 
                                                        bidirectional=True)
                    sym_grad_norm = torch.norm(sym_grad)
                # 修正去噪方向 (梯度下降，让 Loss 变小)
                standard_step_norm = torch.norm(x_prev_standard - x_t) # 计算原始预测更新的模长
                actual_lambda = (standard_step_norm / (sym_grad_norm + 1e-8)) * lambda_sym # 动态调整 lambda：确保对称引导比例适中
                x_prev_guided = x_prev_standard - actual_lambda * sym_grad
                # x_prev_guided = x_prev_standard - lambda_sym * sym_grad
                z[:, :, :3] = x_prev_guided

                if i == 0:
                    print("Applied symmetry guidance success!")

            # =======================================================
            # Repulsion correction
            # =======================================================
            
            # 在 t 很小时加入 ZBL 排斥力
            if i >= self.T - self.prediction_threshold_t:
                # ZBL-based relaxation step
                # zx = self.zbl_relax_step(
                #     z, rl, ra,
                #     node_mask=node_mask,
                #     dt=dt,
                #     eps=2e-3,
                #     r_cut=0.5,   
                # )
                zx = z[:, :, :3]
                zx = self.local_repulsion_correction(
                        zx,
                        cell,
                        node_mask,
                        d_min=0.5,
                        margin=0.05,
                        alpha=0.5
                )
                z[:, :, :3] = zx
        # --- end of for T steps

            self._record_atom_type_state_flow(
                source_tag="sample_score_sde",
                round_index=round_index,
                step_index=step_index,
                atom_type_state_input_available=atom_type_state_input_available,
                input_state_updated_step_index=input_state_updated_step_index,
                atom_type_state_output=atom_type_state,
            )
            if self._is_prediction_window_step(step_index) and atom_type_state is not None:
                atom_type_logits = self.phi_unnormalize_h_cat(atom_type_state.clone(), node_mask)
                final_window_logits.append(atom_type_logits.detach().clone())
                final_window_step_indices.append(int(step_index))
                self._record_atom_prediction_window_step(
                    logits=atom_type_logits,
                    node_mask=node_mask,
                    round_index=round_index,
                    step_index=step_index,
                    source_tag="sample_score_sde",
                    atom_type_state_input_available=atom_type_state_input_available,
                    input_state_updated_step_index=input_state_updated_step_index,
                )

        print('Sampling finished.')

        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0, device=z.device)
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]

        # unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        # post-process one-hot / integer
        # h_cat[:, :, 0] = 0.0  # ensure padding type prob = 0 before argmax
        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        logits = h_cat.clone()  # [B,N,C]
        final_window_ensemble_logits = None
        if final_window_logits:
            final_window_ensemble_logits = torch.stack(final_window_logits, dim=0).mean(dim=0)
        h_cat = self._finalize_atom_type_logits(
            logits,
            node_mask,
            round_index=round_index,
            source_tag="sample_score_sde",
            rl=rl,
            ra=ra,
            apply_repair=True,
            final_window_ensemble_logits=final_window_ensemble_logits,
            final_window_step_indices=final_window_step_indices,
        )
        self._assert_no_all_h_in_one_hot_batch(
            h_cat,
            node_mask,
            source_tag="sample_score_sde",
            round_index=round_index,
            stage="sample_score_sde_output_one_hot",
        )

        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}

        return x, h, rl, ra


    @torch.no_grad()
    def sample_score_sde_Lattice(
        self, rl, ra, n_samples, n_nodes, node_mask, edge_mask, context, 
        fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None, 
        n_corrector_steps=1, snr=0.01, round_index=0, lambda_sym=0.1
    ):
        print('Sampling cell given real length/angles ...')
        rl = rl.to(node_mask.device)
        ra = ra.to(node_mask.device)
        B = n_samples
        device = node_mask.device
        volume = lattice_volume(rl, ra)     # [B]
        N = node_mask.squeeze(-1).sum(-1)   # [B]

        len_scale = (volume / (N + 1e-8)).pow(1/3).view(B,1,1)  # [B,1,1]

        # 1) init Gaussian prior at t=1
        if fix_noise:
            z = self.sample_combined_position_feature_noise(
                1, n_nodes, node_mask
            ).expand(n_samples, -1, -1).clone()
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        print(f"> Reverse SDE steps = {self.T}")
        print('Corrector steps:', n_corrector_steps)
        # 2) time grid, t in [1 → 0]
        t_grid = torch.linspace(1.0, 0.0, self.T+1).to(device)
        atom_type_state = None
        last_atom_state_updated_step_index = None
        final_window_logits = []
        final_window_step_indices = []

        for i in tqdm(range(self.T), desc="Sampling SDE steps"):
            t      = float(t_grid[i].item())
            t_next = float(t_grid[i+1].item())
            dt     = t_next - t               # negative
            step_index = i + 1
            atom_type_state_input_available = atom_type_state is not None
            input_state_updated_step_index = last_atom_state_updated_step_index

            # 只取前 3 维坐标
            zx = z[:, :, :3]
            if z.size(-1) >= self.n_dims + self.num_classes:
                atom_type_state = z[:, :, self.n_dims:self.n_dims+self.num_classes]
            t_tensor = torch.full((B,1), fill_value=t, device=device)

            # =======================================================
            # Corrector (Langevin)
            # =======================================================
            
            for _ in range(n_corrector_steps):
                # 网络前向 -> 得分
                net_out = self.phi(
                    zx, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra,
                    atom_type_state=atom_type_state,
                )
                atom_type_state = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
                gamma_t = self.gamma(t_tensor)
                sigma_t = self.sigma(gamma_t, zx).view(B,1,1)
                score = net_out[:, :, :3] / sigma_t
                score = score / len_scale # length scale
                # 计算 alpha_t_pc, 见 SDE 论文 P23 Algorithm 3, 5
                alpha_t_pc = self.alpha(gamma_t, zx) ** 2
                
                # 噪声
                noise = torch.randn_like(zx) * node_mask

                # Langevin 步
                # SNR 根据 PC 论文设置
                grad_norm = score.reshape(B, -1).norm(dim=-1) # [B]
                noise_norm = noise.reshape(B, -1).norm(dim=-1) # [B]
                eps = 2  * ((snr * noise_norm / (grad_norm + 1e-10))**2) * alpha_t_pc.squeeze() # [B]
                eps = eps.view(B,1,1)
                zx = zx + eps * score + torch.sqrt(2.0 * eps) * noise
                zx = torch.remainder(zx, 1.0) # mod 1

            # =======================================================
            # Predictor (reverse SDE Euler-Maruyama)
            # =======================================================

            # 一次 SDE 反向步
            z = self.reverse_sde_step(
                x=zx,
                t=t,
                dt=dt,
                rl=rl, ra=ra,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=context,
                len_scale=len_scale,
                atom_type_state=atom_type_state,
            )
            if z.size(-1) >= self.n_dims + self.num_classes:
                atom_type_state = z[:, :, self.n_dims:self.n_dims+self.num_classes]
                last_atom_state_updated_step_index = step_index

            # =======================================================
            # Repulsion correction
            # =======================================================
            
            # 在 t 很小时加入 ZBL 排斥力
            if t < 0.01:
                zx = self.zbl_relax_step(
                    z, rl, ra,
                    node_mask=node_mask,
                    dt=dt,
                    eps=2e-3,
                    r_cut=0.5,   
                )
                z[:, :, :3] = zx

            self._record_atom_type_state_flow(
                source_tag="sample_score_sde_Lattice",
                round_index=round_index,
                step_index=step_index,
                atom_type_state_input_available=atom_type_state_input_available,
                input_state_updated_step_index=input_state_updated_step_index,
                atom_type_state_output=atom_type_state,
            )
            if self._is_prediction_window_step(step_index) and atom_type_state is not None:
                atom_type_logits = self.phi_unnormalize_h_cat(atom_type_state.clone(), node_mask)
                final_window_logits.append(atom_type_logits.detach().clone())
                final_window_step_indices.append(int(step_index))
                self._record_atom_prediction_window_step(
                    logits=atom_type_logits,
                    node_mask=node_mask,
                    round_index=round_index,
                    step_index=step_index,
                    source_tag="sample_score_sde_Lattice",
                    atom_type_state_input_available=atom_type_state_input_available,
                    input_state_updated_step_index=input_state_updated_step_index,
                )

        print('Sampling finished.')

        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0, device=z.device)
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]

        # unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        # post-process one-hot / integer
        final_window_ensemble_logits = None
        if final_window_logits:
            final_window_ensemble_logits = torch.stack(final_window_logits, dim=0).mean(dim=0)
        h_cat = self._finalize_atom_type_logits(
            h_cat,
            node_mask,
            round_index=round_index,
            source_tag="sample_score_sde_Lattice",
            rl=rl,
            ra=ra,
            apply_repair=False,
            final_window_ensemble_logits=final_window_ensemble_logits,
            final_window_step_indices=final_window_step_indices,
        )
        self._assert_no_all_h_in_one_hot_batch(
            h_cat,
            node_mask,
            source_tag="sample_score_sde_Lattice",
            round_index=round_index,
            stage="sample_score_sde_Lattice_output_one_hot",
        )
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}

        return x, h


    def zbl_relax_step(
        self, z, rl, ra, node_mask,
        dt, eps=1e-2, r_cut=0.5, n_inner=5
    ):  
        B, N, _ = z.shape
        device = z.device
        frac = z[:,:,:self.n_dims]
        h_cat = z[:,:,self.n_dims:self.n_dims+self.num_classes]
        cell = self.compute_lattice_matrix(rl, ra)  # [B,3,3]
        inv_cell = torch.linalg.pinv(cell) # [B,3,3]
        x = torch.einsum('bnm,bmk->bnk', frac, cell)   # [B,N,3]
        
        # pairwise differences (cartesian)
        dx = x[:, :, None, :] - x[:, None, :, :]        # [B,N,N,3]
        dx = self.pbc_minimum_image(dx, cell, inv_cell)     # [B,N,N,3], minimum-image
        dist = torch.norm(dx, dim=-1) 

        # masks
        mask = node_mask.squeeze(-1).bool()  # [B,N]
        pair_mask = mask[:, :, None] & mask[:, None, :]  # [B,N,N]
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
        pair_mask = pair_mask & (~eye)
        close_mask = (dist < r_cut) & pair_mask  # [B,N,N]
        if close_mask.sum() == 0:
            return frac # nothing to do
        else:
            print(f"ZBL relaxation: {close_mask.sum(dim=(1, 2)).tolist()} close pairs found.")
        
            # compute Z from one-hot h_cat 
            class_idx = torch.argmax(h_cat, dim=-1)
            Z = (class_idx * node_mask.squeeze(-1).long()).to(device)   # [B,N], padding zeros
            Zi = Z.float()
            Zj = Z.float()
            Z1Z2 = Zi[:, :, None] * Zj[:, None, :]

            # unit vectors
            r_hat = dx / (dist[..., None] + 1e-12)         # [B,N,N,3]
            # compute pairwise force magnitude
            F_mag = - zbl_force_mag(dist, Z1Z2, Z_i=Zi, Z_j=Zj)   # [B,N,N]
            F_mag = F_mag * close_mask.float()
            # per-pair vector forces (cartesian)
            F_pair = F_mag[..., None] * r_hat # [B,N,N,3]
            # net force on each atom (sum over j)
            F_cart = F_pair.sum(dim=2) # [B,N,3]

            # convert cartesian force -> fractional-coord 'force'
            # small cart displacement dx corresponds to dfrac = dx @ inv_cell
            F_frac = torch.einsum('bnk,bkl->bnl', F_cart, inv_cell)   # [B,N,3]

            # update fractional coords
            step = eps * F_frac * abs(float(dt))   # use abs(dt) to scale step size
            frac_new = frac + step
            
            ## debugging info
            # step_cart = torch.einsum('bnk,bkl->bnl', step, cell)  # [B,N,3] Cartesian
            # step_cart_mag = torch.norm(step_cart, dim=-1)       # [B,N] Å
            # idx_b, idx_i, idx_j = torch.where(close_mask)
            # for b,i,j in zip(idx_b.tolist(), idx_i.tolist(), idx_j.tolist()):
            #     print(f"Batch {b}, atoms {i}-{j}, step_cart_mag_i \
            #           = {step_cart_mag[b,i]:.4f} Å, step_cart_mag_j = {step_cart_mag[b,j]:.4f} Å")
            # self.debug_zbl_force_direction(dist, dx, F_pair, close_mask)
            
            frac_new = wrap_at_boundary(frac_new, wrapping_boundary=1.0)
            
            return frac_new
    
    # def pbc_minimum_image(self, dx, cell, inv_cell):
    #     """
    #     dx: [B,N,N,3] raw cartesian difference vectors (xi - xj)
    #     cell: [B,3,3] with rows = a,b,c
    #     inv_cell: [B,3,3] inverse matrices
    #     Returns dx mapped to minimum image under PBC, same shape as dx.
    #     """
    #     # dx (cart) -> fractional differences: d_frac = dx @ inv_cell^T
    #     # einsum notation: 'bnjk,bkl->bnjl' where last index maps to fractional components
    #     d_frac = torch.einsum('bnjk,bkl->bnjl', dx, inv_cell)   # [B,N,N,3]
    #     d_frac = d_frac - torch.round(d_frac) # wrap to [-0.5,0.5]
    #     dx_pbc = torch.einsum('bnjk,bkl->bnjl', d_frac, cell)   # back to cart
        
    #     return dx_pbc
    
    def pbc_minimum_image(self, dx, cell, inv_cell):
        """
        dx: [B,N,N,3] raw cartesian difference vectors (xi - xj)
        cell: [B,3,3] with rows = a,b,c
        inv_cell: [B,3,3] inverse matrices

        Returns:
            dx_pbc: [B,N,N,3] shortest cartesian vectors under PBC
                    (equivalent to pymatgen.coord_cython.pbc_shortest_vectors)
        """
        device = dx.device
        dtype = dx.dtype
        B, N, _, _ = dx.shape

        # cart -> fractional differences
        d_frac = torch.einsum('bnjk,bkl->bnjl', dx, inv_cell)  # [B,N,N,3]

        # all integer shifts in {-1,0,1}^3
        shifts = torch.tensor(
            [[i, j, k] for i in (-1, 0, 1)
                        for j in (-1, 0, 1)
                        for k in (-1, 0, 1)],
            device=device,
            dtype=dtype
        )  # [27,3]

        # apply shifts
        # [B,N,N,1,3] + [1,1,1,27,3] -> [B,N,N,27,3]
        d_frac_all = d_frac.unsqueeze(-2) + shifts.view(1, 1, 1, 27, 3)

        # back to cartesian
        dx_all = torch.einsum(
            'bnjlk,bkm->bnjlm', d_frac_all, cell
        )  # [B,N,N,27,3]

        # choose shortest
        dist2 = (dx_all ** 2).sum(dim=-1)          # [B,N,N,27]
        min_idx = dist2.argmin(dim=-1)              # [B,N,N]

        dx_pbc = dx_all.gather(
            dim=-2,
            index=min_idx[..., None, None].expand(-1, -1, -1, 1, 3)
        ).squeeze(-2)                               # [B,N,N,3]

        return dx_pbc

    
    def debug_zbl_force_direction(self, dist, dx, F_pair, close_mask):
        """
        Debug ZBL force direction.

        Parameters:
        -----------
        dist : torch.Tensor [B,N,N]
            Pairwise Cartesian distances.
        dx : torch.Tensor [B,N,N,3]
            Pairwise Cartesian difference vectors (i -> j).
        F_pair : torch.Tensor [B,N,N,3]
            Pairwise force vectors on i due to j.
        close_mask : torch.BoolTensor [B,N,N]
            Mask indicating which pairs are close enough to apply ZBL.

        Prints:
        -------
        For each batch, the closest pairs:
            - distance
            - |force|
            - cos(theta) between force and vector i->j
            - Suggestion if force is not pushing apart
        """
        B, N, _ = dx.shape[:3]

        idx_b, idx_i, idx_j = torch.where(close_mask)
        for b, i, j in zip(idx_b.tolist(), idx_i.tolist(), idx_j.tolist()):
            if i >= j:
                continue  # only upper triangle to avoid duplicates

            rij = dx[b, i, j]
            dist_ij = dist[b, i, j]
            F_ij = F_pair[b, i, j]
            F_norm = torch.norm(F_ij).item()

            # cos(theta) between force vector and rij (should be -1 for repulsion)
            cos_theta = torch.dot(F_ij, rij) / (torch.norm(F_ij) * torch.norm(rij) + 1e-12)
            direction_ok = "OK" if cos_theta < 0 else "WRONG"

        print(f"Batch {b}, atoms {i}-{j}, dist={dist_ij:.4f} Å, |F|={F_norm:.4f}, \
              cos(theta)={cos_theta:.4f}, direction={direction_ok}")
        

    def local_repulsion_correction(
        self,
        frac,                # [B, N, 3] fractional coords
        cell,                # [B, 3, 3] cell matrix (rows = a,b,c vectors)
        node_mask,           # [B, N, 1] atom mask
        d_min=0.5,
        margin=0.05,         # push to d_min + margin
        alpha=0.5,           # strength factor (0.3~0.7 recommended)
        eps=1e-12
    ):
        """
        Perform local, pairwise repulsion correction:
        - Detect atom pairs with minimal-image distance < d_min
        - Push them in Cartesian coordinates up to (d_min + margin)
        - Convert displacement back to fractional space
        """
        B, N, _ = frac.shape
        device = frac.device
        inv_cell = torch.linalg.pinv(cell) # [B,3,3]
        if cell.isnan().any() or inv_cell.isnan().any():
            print("NaN detected in cell or inv_cell!")
            inv_cell = torch.nan_to_num(inv_cell, nan=0.0, posinf=0.0, neginf=0.0)
            cell = torch.nan_to_num(cell, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.einsum('bnm,bmk->bnk', frac, cell)   # [B,N,3]
        
        # pairwise differences (cartesian)
        dx = x[:, :, None, :] - x[:, None, :, :]        # [B,N,N,3]
        dx = self.pbc_minimum_image(dx, cell, inv_cell)     # [B,N,N,3], minimum-image
        dist = torch.norm(dx, dim=-1) 

        # masks
        mask = node_mask.squeeze(-1).bool()  # [B,N]
        pair_mask = mask[:, :, None] & mask[:, None, :]  # [B,N,N]
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
        pair_mask = pair_mask & (~eye)
        close_mask = (dist < d_min) & pair_mask  # [B,N,N]

        if not close_mask.any():
            return frac  # no correction needed
        
        print(f"ZBL relaxation: {(close_mask.sum(dim=(1, 2))/2).tolist()} close pairs found.")
        target = (d_min + margin)   # target distance
        direction = dx / (dist[..., None] + eps)                   # [B, N, N, 3]
        if direction.isnan().any():
            direction = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
        delta_mag = (target - dist).clamp(min=0.0)                 # [B, N, N]
        if delta_mag.isnan().any():
            delta_mag = torch.nan_to_num(delta_mag, nan=0.0, posinf=0.0, neginf=0.0)
        delta = delta_mag[..., None] * direction                   # [B, N, N, 3]
        delta = delta * close_mask[..., None].float()

        # disp_i_cart =  0.5 * delta.sum(dim=2)   # [B,N,3], i 方向的位移贡献
        # disp_j_cart = -0.5 * delta.sum(dim=1)   # [B,N,3], j 方向的位移贡献 (negative because j moves opposite)
        # disp_cart = disp_i_cart + disp_j_cart
        
        disp_cart = 0.5 * delta.sum(dim=2)   # [B,N,3]
        disp_frac = torch.einsum('bnm,bmk->bnk', disp_cart, inv_cell)   # [B,N,3]
        frac_new = frac + disp_frac
        frac_new = wrap_at_boundary(frac_new, wrapping_boundary=1.0)

        if torch.isnan(frac_new).any():
            print("NaN detected in local_repulsion_correction!")
            idx = torch.isnan(frac_new).nonzero()
            print("NaN detected at indices:", idx)
            print("frac before update:", frac[idx[:,0], idx[:,1], :])
            print("disp_frac:", disp_frac[idx[:,0], idx[:,1], :])

        return frac_new
