from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import Equiformer_dynamics
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.mlp import DiffusionMLP
from torch_scatter import scatter_mean
from crystalgrw.data_utils import lattice_params_from_matrix
from equivariant_diffusion.mlp import DiffusionMLP
from tqdm import tqdm
import random


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
            lambda_type=1, lambda_rep=1, cutoff=0.5,
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
            self.gamma_lengths = GammaNetwork()
            self.gamma_angles = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)
            self.gamma_lengths = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision, print_info=False)
            self.gamma_angles = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision, print_info=False)
            print("Using predefined noise schedule for gamma, l and a is same as x.")

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

        if noise_schedule != 'learned':
            self.check_issues_norm_values()
            
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

        print("use lambda_type: ", lambda_type)
        print("use lambda_rep: ", lambda_rep)
        print("cutoff for repulsion loss: ", cutoff)

        print(f"{self.__class__.__name__} initialized.")
        
    
    def save_intermediate_grad(self, grad):
        self.saved_grad = grad

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

    def phi(self, zxh, t, node_mask, edge_mask, context, t2=None, mask_y=None, rl=None, ra=None):
        """noise predict network"""   
        # h_cat = zxh[:, :, self.n_dims:self.n_dims+self.num_classes] # [B, N, num_classes]
        # h_cat = self.phi_unnormalize_h_cat(h_cat, node_mask)
        # h_int = zxh[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+self.include_charges] \
        #     if self.include_charges else torch.zeros(0).to(z.device) # [B, N, 1]
        
        """prepare input for eps x"""
        zx = zxh[:, :, :self.n_dims]  # [B, N, 3]
        B, N= zx.size(0), zx.size(1)

        # for example, x: R -> [0,1]
        if rl is not None and ra is not None:
            rl, ra = self.phi_unnormalize_la(rl, ra)
        else:
            print("No valid lengths and angles provided for unnormalization in phi.")
            raise ValueError
        # atom_types 默认全是 1 ，作为dynamics的输入
        zx, atom_types, natoms, rl, ra, batch = \
            self.prepare_inputs_for_equiformer(t, zx, rl, ra, node_mask)
        
        """forward for x, h"""
        net_outs = self.dynamics(t, zx, atom_types, natoms, \
                lengths=rl, angles=ra, batch=batch)
        
        # outputs reshape
        net_eps_x, net_pred_h = self.reshape_outputs(
            net_outs, B, N, node_mask, natoms, batch)
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

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
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

        # Compute gamma_s and gamma_t
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

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

        # final loss
        kl_prior = torch.zeros_like(score_loss)
        loss = kl_prior + score_loss
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        loss_dict = {'t': t_int.squeeze(),
                     'loss': loss.squeeze(), 
                     'x_error':(score_loss / sigma_t).squeeze()}

        # calculate the loss for atom type
        h_true = torch.cat([h['categorical'], h['integer']], 
                            dim=2).clone().detach().requires_grad_(True).to(torch.float32).to(x.device)
        h_true_idx = h_true.argmax(dim=2)   # [B,N]
        h_pred = net_out[:, :, 3:]          # [B,N,C]
        # cross_entropy loss
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        atom_type_loss = ce_loss(
            h_pred.reshape(-1, h_pred.size(-1)), # logits, [B*N, C]
            h_true_idx.reshape(-1) # targets, [B*N]
        )
        atom_type_loss = atom_type_loss.reshape(batch_size, -1)  # [B, N]
        atom_type_loss = atom_type_loss * node_mask.squeeze(-1)
        atom_type_loss = atom_type_loss.mean(dim=1)
        # mask the loss term with t > prediction_threshold_t
        pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
        pred_loss_mask = pred_loss_mask.squeeze(1)
        atom_type_loss = self.lambda_type * (atom_type_loss * pred_loss_mask)
        loss_dict["atom_type_loss"] = atom_type_loss
        loss += atom_type_loss

        # calculate the loss for repulsion term
        score_pred = pred / (sigma_t + 1e-8)  # scale back
        x_hat = (z_t[:, :, :3] - sigma_t * score_pred) / (alpha_t + 1e-8)   # [B,N,3]
        x_hat = wrap_at_boundary(x_hat, wrapping_boundary=1.0) # clean sample
        L = self.compute_lattice_matrix(
            *self.unnormalize_lengths_angles(lengths, angles))  # [B,3,3]
        repulsion_loss = self.compute_repulsion_loss_from_fractional(
            x_hat, L, node_mask, t_int.squeeze(), cutoff=self.cutoff, t_threshold=self.prediction_threshold_t)
        repulsion_loss = self.lambda_rep * repulsion_loss
        loss_dict["repulsion_loss"] = repulsion_loss
        loss += repulsion_loss

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
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
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
               num_rounds=1, seed_base=None, rl=None, ra=None, sample_realistic_LA=False):
        """Samples from the model using score function."""    
        results = []

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
                    annel_l, pesudo_context, n_corrector_steps
                )
            else:
                # x, h, rl, ra = self.sample_score(LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, 
                #                     context, fix_noise, condition_generate_x, annel_l, pesudo_context)            
                x, h, rl, ra = self.sample_score_sde(
                    LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask,
                    context, fix_noise, condition_generate_x,
                    annel_l, pesudo_context,
                    n_corrector_steps
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
    

    def prepare_inputs_for_equiformer(self, t, x, lengths, angles, node_mask, edge_mask=None):
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
        # 3) batch 索引 [N_total]
        batch = torch.arange(B, device=device).repeat_interleave(natoms)
        # 4) 原子种类 [N_total] (这里用dummy，实际应替换成数据里的 atom_types)
        atom_types = torch.ones_like(batch)     # [N_total]  全部设为1，表示 dummy 原子类型

        return pos, atom_types, natoms, lengths, angles, batch


    
    def reshape_outputs(self, outs, B, N, node_mask, natoms, batch):
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
        model_out_is_eps=False,
        len_scale=None,
    ):  
        B, N, D = x.shape
        device = x.device

        # 1) prepare input
        t_tensor = torch.full((B,1), fill_value=t, device=device)

        # 2) model forward (must output score)
        net_out = self.phi(
            x, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra
        )

        # 若模型预测的是乘以sigma_t的score，故转换为score需要除以sigma_t
        sigma_t = self.sigma(self.gamma(t_tensor), target_tensor=x)   # [B,1] or [B,1,1]
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
    def sample_score_sde(
        self, LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, context, 
        fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None, 
        n_corrector_steps=1, snr=0.01
    ):
        print('Sampling cell length/angles ...')
        rl, ra = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
        rl = rl.to(node_mask.device)
        ra = ra.to(node_mask.device)
        print('sample lengths and angles done.')

        volume = lattice_volume(rl, ra)     # [B]
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
        print('Corrector steps:', n_corrector_steps)
        # 2) time grid, t in [1 → 0]
        t_grid = torch.linspace(1.0, 0.0, self.T+1).to(device)

        for i in tqdm(range(self.T), desc="Sampling SDE steps"):
            t      = float(t_grid[i].item())
            t_next = float(t_grid[i+1].item())
            dt     = t_next - t               # negative

            # 只取前 3 维坐标
            zx = z[:, :, :3]
            t_tensor = torch.full((B,1), fill_value=t, device=device)

            # =======================================================
            # Corrector (Langevin)
            # =======================================================
            
            for _ in range(n_corrector_steps):
                # 网络前向 -> 得分
                net_out = self.phi(zx, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra)
                gamma_t = self.gamma(t_tensor)
                sigma_t = self.sigma(gamma_t, zx).view(B,1,1)
                score = net_out[:, :, :3] / sigma_t
                score = score / len_scale # scale score according to length scale
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
                model_out_is_eps=False,
                len_scale=len_scale,
            )

        print('Sampling finished.')

        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0, device=z.device)
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]

        # unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        # post-process one-hot / integer
        # Insert this BEFORE line 2118
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}

        return x, h, rl, ra

    @torch.no_grad()
    def sample_score_sde_Lattice(
        self, rl, ra, n_samples, n_nodes, node_mask, edge_mask, context, 
        fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None, 
        n_corrector_steps=1, snr=0.01
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

        for i in tqdm(range(self.T), desc="Sampling SDE steps"):
            t      = float(t_grid[i].item())
            t_next = float(t_grid[i+1].item())
            dt     = t_next - t               # negative

            # 只取前 3 维坐标
            zx = z[:, :, :3]
            t_tensor = torch.full((B,1), fill_value=t, device=device)

            # =======================================================
            # Corrector (Langevin)
            # =======================================================
            
            for _ in range(n_corrector_steps):
                # 网络前向 -> 得分
                net_out = self.phi(zx, t_tensor, node_mask, edge_mask, context, rl=rl, ra=ra)
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
                model_out_is_eps=False,
                len_scale=len_scale,
            )

        print('Sampling finished.')

        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0, device=z.device)
        h_cat = z[:, :, self.n_dims:self.n_dims+self.num_classes]

        # unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        # post-process one-hot / integer
        # Insert this BEFORE line 2118
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}

        return x, h

