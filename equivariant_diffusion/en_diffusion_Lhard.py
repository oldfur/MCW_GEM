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


class EquiTransVariationalDiffusion_Lhard(torch.nn.Module):
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
            len_dim=3, angle_dim=3, lambda_l=1, lambda_a=1,
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
        print("use lambda_l: ", lambda_l)
        print("use lambda_a: ", lambda_a)

        self.length_mlp = DiffusionMLP(input_dim=3, output_dim=3, 
                         hidden_dims=[128, 128], use_self_attn=False)
        self.angle_mlp = DiffusionMLP(input_dim=3, output_dim=3, 
                         hidden_dims=[128, 128], use_self_attn=False)
        
        print("EquiTransVariationalDiffusion_Lfirst initialized.")
        
    
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
        B = zx.size(0)
        N = zx.size(1)
        zx = self.phi_unnormalize_x(zx) # unnormalize, for that gnn dynamics works on physical space
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
        net_eps_x, net_pred_h = self.phi_normalize_xh(net_eps_x, net_pred_h, node_mask)
        net_eps_xh = torch.cat([net_eps_x, net_pred_h], dim=2) # [B, N, 3 + num_classes]

        if self.property_pred and self.use_prop_pred:
            property_pred = net_outs['property_pred']
            return (net_eps_xh, property_pred)

        return net_eps_xh
    

    def phi_la(self, zl, za, t):
        """noise predict network for lengths and angles only"""   
        
        net_eps_l = self.length_mlp(zl, t)
        net_eps_a = self.angle_mlp(za, t)

        return net_eps_l, net_eps_a
    
    def phi_xh(self, zxh, t, rl, ra, node_mask, edge_mask, context, t2=None, mask_y=None):
        """noise predict network for x and h only"""   

        zx = zxh[:, :, :self.n_dims]  # [B, N, 3]
        B, N  = zx.size(0), zx.size(1)
        zx = self.phi_unnormalize_x(zx)
        zx, atom_types, natoms, rl, ra, batch = \
            self.prepare_inputs_for_equiformer(t, zx, rl, ra, node_mask)
        
        # forward for x, h
        net_outs = self.dynamics(t, zx, atom_types, natoms, \
                lengths=rl, angles=ra, batch=batch)
        
        # output
        net_eps_x, net_pred_h = self.reshape_outputs(
            net_outs, B, N, node_mask, natoms, batch)
        net_eps_x, net_pred_h = self.phi_normalize_xh(net_eps_x, net_pred_h, node_mask)
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

    def normalize(self, x, h, node_mask):
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0]) 

        # Casting to float in case h still has long or int type.
        h_cat = h['categorical'].float() * node_mask
        h_int = h['integer'].float()
        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    
    def normalize_cartesian_pos_with_h(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    

    def phi_normalize_xh(self, x, h, node_mask=None):
        x = x / self.norm_values[0]
        if node_mask is not None:
            h = (h.float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        else :
            h = (h.float() - self.norm_biases[1]) / self.norm_values[1]

        return x, h


    def phi_unnormalize_x(self, x):
        x = x * self.norm_values[0]
        return x
    
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
        x = x * self.norm_values[0]
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

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        # print("x_shape: ", x.shape)
        # print("h_cat_shape: ", h_cat.shape)
        # print("h_int_shape: ", h_int.shape)
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        if self.atom_type_pred:
            h_fix = torch.ones_like(torch.cat([h_cat, h_int], dim=2)) 
            return torch.cat([x, h_fix], dim=2)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output
    
    def unnormalize_z_with_lengths_angles(self, z, z_l, z_a, node_mask):
        """
        对 z、z_l 和 z_a 进行反归一化，假设 norm_values[3]、norm_values[4] 和 norm_biases[3]、norm_biases[4] 已定义。
        输入:
            z: [batch_size, num_nodes, n_dims + num_classes + include_charges]
            z_l: [batch_size, num_nodes, 3]
            z_a: [batch_size, num_nodes, 3]
            node_mask: [batch_size, num_nodes, 1]
        输出:
            output: 反归一化后的张量
        """
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize xh, lengths and angles
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        lengths, angles = self.unnormalize_lengths_angles(z_l, z_a)

        if self.atom_type_pred:
            h_fix = torch.ones_like(torch.cat([h_cat, h_int], dim=2)) 
            return torch.cat([x, h_fix], dim=2), lengths, angles
        
        xh = torch.cat([x, h_cat, h_int], dim=2)

        return xh, lengths, angles

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

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(zT | x) and the prior p(zT) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        if self.atom_type_pred:
            kl_distance_h = 0.0
        return kl_distance_x + kl_distance_h


    def compute_x_pred(self, net_out, zt, gamma_t):
        # print(f"net_out.shape: {net_out.shape}, zt.shape: {zt.shape}, gamma_t.shape: {gamma_t.shape}")
        """Commputes x_pred, i.e. the most likely prediction of x."""
        net_out_new = net_out
        sigma_t = self.sigma(gamma_t, target_tensor=net_out_new)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out_new)
        eps_t = net_out_new
        x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        return x_pred

    def compute_length_pred(self, l_out, zt_l, gamma_t):
        """Commputes length predictions."""
        net_out_new = l_out
        sigma_t = self.sigma(gamma_t, target_tensor=net_out_new)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out_new)
        eps_t = net_out_new
        length_pred = 1. / alpha_t * (zt_l - sigma_t * eps_t)
        return length_pred

    def compute_angle_pred(self, a_out, zt_a, gamma_t):
        """Commputes angle predictions."""
        net_out_new = a_out
        sigma_t = self.sigma(gamma_t, target_tensor=net_out_new)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out_new)
        eps_t = net_out_new
        angle_pred = 1. / alpha_t * (zt_a - sigma_t * eps_t)
        return angle_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.atom_type_pred:
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]
     
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error    # [B]


    def compute_error_mp20(self, net_out, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out

        if self.atom_type_pred:
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]

        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            x_error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:   # test performance
            x_error = sum_except_batch((eps - eps_t) ** 2)

        return x_error
    
    def show_x_error(self, net_out, eps):
        eps_t = net_out
        if self.atom_type_pred:
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]

        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        print("x error is :", error)
        return error


    def show_l_error(self, l_out, eps_l):
        eps_t_l = l_out
        error = sum_except_batch((eps_l - eps_t_l) ** 2) / self.len_dim
        print("length error is :", error)
        return error


    def show_a_error(self, a_out, eps_a):
        eps_t_a = a_out
        error = sum_except_batch((eps_a - eps_t_a) ** 2) / self.angle_dim
        print("angle error is :", error)
        return error

    def log_constants_p_x_given_z0(self, batch_size, device, node_mask):
        """Computes p(x|z0)."""
        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))


    def sample_p_xh_lengths_angles_given_z0(self, z0, rl, ra, node_mask, edge_mask, context, fix_noise=False):
        """Samples x,h ~ p(x|z0)."""
        bs = z0.size(0)
        zeros = torch.zeros(size=(bs, 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        if self.property_pred:
            (net_out, pred) = self.phi_xh(z0, zeros, rl, ra, node_mask, edge_mask, context)
        else:
            net_out = self.phi_xh(z0, zeros, rl, ra, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        
        # Sample from Normal distribution
        if self.atom_type_pred:
            xh = self.sample_normal(mu=mu_x[:,:,:3], sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        else:
            xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        if self.atom_type_pred: # no include_charges
            h_cat = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
        else:
            h_cat = z0[:, :, self.n_dims:-1]

        # unnormalize x,h
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        if self.property_pred:
            return x, h, pred
        return x, h


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
    

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x) # L_zero first item

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z


    def log_p_xhla_given_z0_without_constants(self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        """
        计算 x, h 在给定 z0 下的log likelihood
        """
        # x, h 部分同log_pxh_given_z0_without_constants
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # x
        # x ~ N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0)
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)
        # print("log_p_x_given_z_without_constants: ", log_p_x_given_z_without_constants.sum(0))

        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]
        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()
        h_integer_centered = h_integer - estimated_h_integer
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)
        centered_h_cat = estimated_h_cat - 1
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)
        log_p_h_given_z = log_ph_integer + log_ph_cat
        # print("log_p_h_given_z: ", log_p_h_given_z.sum(0))

        # 合并所有对数似然
        log_p_xh_lengths_angles_given_z = log_p_x_given_z_without_constants + log_p_h_given_z
        return log_p_xh_lengths_angles_given_z


    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        gamma = gamma.reshape(-1, 1 ,1)
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma
    
    
    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        # return loss
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))
    
    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        # return loss
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))
    
    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # print("msx",mu,sigma,x)
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        # print(sigma.min(),sigma.max())
        # print(mu.min(),mu.max())

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        # if not torch.all(f_.isfinite()):
        #     print("f_", f_.min(), f_.max())
        #     print("mu", mu.min(), mu.max())
        #     print("sigma", sigma.min(), sigma.max())
        #     print("x", x.min(), x.max())
        #     print("flag_upper", flag_upper.min(), flag_upper.max())
        #     print("flag_lower", flag_lower.min(), flag_lower.max())
        #     raise ValueError("f_ is not finite")
        return f_
    
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment

        return x


    def compute_loss(self, x, h, lengths, angles, node_mask, edge_mask, context, t0_always, mask_indicator=None,
                     property_label=None, time_upperbond=-1, bond_info=None):
        """
        Computes an estimator for the variational lower bound, or the simple loss (MSE).

        :param x: [B, N, self.n_dims]
        :param h: [B, N, self.in_node_nf]
        :param node_mask: [B, N, 1]
        :param edge_mask: [B, N, N]
        :param context: Not used here.
        训练时的输入
        :param mask_indicator: 训练时是否使用mask
        :param property_label: 属性标签
        :param time_upperbond: 时间上限
        :param bond_info: 键值对

        :return: loss, loss_dict
        """

        batch_size = x.size(0)

        # This part is about whether to include loss term 0 always.
        if t0_always:
            lowest_t = 1
        else:
            lowest_t = 0

        # 采样时间步t,采样范围(int)为: [lowest_t, T], 采样结果变量t_int: [B, 1]
        if self.property_pred:
            # 0.5 rate use T+1, 0.5 rate use prediction_threshold_t
            random_number = torch.rand(1).item()
            # if random_number < -1:
            if self.unnormal_time_step:
                random_th = 0.5
            elif self.only_noisy_node:
                random_th = 1
            else:
                random_th = -1
            
            if random_number < random_th:
                random_prop = True
                t_int = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, 
                    size=(batch_size, 1), device=x.device).float()
                # lowest_t+1
                """一半的概率介于[0, T+1]，一半的概率介于[0, prediction_threshold_t]"""
            else:
                random_prop = False
                t_int = torch.randint(
                    lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()
                # lowest_t+1
            # print("t_int: ", t_int)
        else:
            t_int = torch.randint(
                    lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()
        
        if time_upperbond >= 0:
            t_int = torch.ones_like(t_int) * time_upperbond

        if self.half_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(batch_size - half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(lowest_t, self.T + 1, size=(half_batch_size, 1), device=x.device).float()
        
        if self.sep_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(batch_size - half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(self.prediction_threshold_t + 1, self.T + 1, 
                                                      size=(half_batch_size, 1), device=x.device).float()
        
        s_int = t_int - 1 
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t and s to [0, 1]. Note that the negative step 
        # of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1, t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)  
        # xh: [B, N, n_dims + num_classes + include_charges]
        fix_h = torch.ones_like(torch.cat([h['categorical'], h['integer']], dim=2))

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        if self.atom_type_pred:
            z_t = alpha_t * x + sigma_t * eps   
            z_t = torch.cat([z_t, fix_h], dim=2)
        else:
            z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        # phi: noise prediction model
        if self.property_pred:
            (net_out, property_pred) = self.phi(z_t, t, node_mask, edge_mask, 
                                                context, rl=lengths, ra=angles)
        else:
            # Neural net prediction.
            net_out = self.phi(z_t, t, node_mask, edge_mask, 
                               context, rl=lengths, ra=angles)

        # Compute the error.
        x_error = self.compute_error_mp20(net_out, eps)
        if self.training and self.loss_type == 'l2':
            SNR_weight_x = torch.ones_like(x_error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight_x = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert x_error.size() == SNR_weight_x.size()
        loss_t_larger_than_zero = 0.5 * (SNR_weight_x * x_error)

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(batch_size, x.device, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:   # test
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x,l,a,h for timestep t, from q(z_t | x,l,a,h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)

            if self.atom_type_pred:
                z_0 = alpha_0 * x + sigma_0 * eps_0
                z_0 = torch.cat([z_0, fix_h], dim=2)
            else:
                z_0 = alpha_0 * xh + sigma_0 * eps_0

            if self.property_pred:
                (net_out, property_pred) = self.phi(z_0, t_zeros, node_mask, edge_mask, context, 
                                                    rl=lengths, ra=angles)
            else:
                net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context, 
                                rl=lengths, ra=angles)

            # Compute the error for t = 0.            
            loss_term_0 = -self.log_p_xhla_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            # Combine all terms.
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
            # print("estimator_loss_terms: ", estimator_loss_terms.sum(0))

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_p_xhla_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask) 

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero
            # loss_term_0: main loss term

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size(), \
                print("kl_prior.size(), estimator_loss_terms.size(): ", kl_prior.size(), estimator_loss_terms.size())
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
         
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        loss_dict = {'t': t_int.squeeze(), 
                     'loss':loss.squeeze(),
                     'loss_t': loss_t.squeeze(),
                     'x_error': x_error.squeeze(),
                     'kl_prior': kl_prior.squeeze(),
                     'neg_log_constants': neg_log_constants.squeeze(),
                     'estimator_loss_terms': estimator_loss_terms.squeeze(),
                     'loss_term_0': loss_term_0.squeeze(),
                     'loss_t_larger_than_zero': loss_t_larger_than_zero.squeeze()}


        """至此loss已经计算完成, 下面mask掉不需要计算的loss, 即生长阶段loss"""

        # calc loss for prediction
        if self.property_pred:
            if self.target_property is not None:
                #loss for prediction use l1 loss
                loss_l1 = torch.nn.L1Loss(reduction='none')
                # print(property_pred.size(), property_label.size())
                assert property_pred.size() == property_label.size()
                pred_loss = loss_l1(property_pred, property_label)
                if pred_loss.dim() > 1 and pred_loss.size(1) == 53: # basic prob
                    pred_loss = pred_loss.mean(dim=1)
            else:
                # 0 loss for prediction
                pred_loss = torch.zeros_like(property_pred)

        if self.atom_type_pred:
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

            # l1 loss
            # l1_loss = torch.nn.L1Loss(reduction='none')
            # atom_type_loss = l1_loss(h_true, h_pred)
            # atom_type_loss = atom_type_loss * node_mask
            # atom_type_loss = atom_type_loss.mean(dim=2).mean(dim=1)

        if self.property_pred:
            pred_loss_mask = (t_int <= self.prediction_threshold_t).float()

            if self.use_prop_pred == 0:
                pred_loss_mask = torch.zeros_like(pred_loss_mask).to(pred_loss_mask.device)
            pred_rate = pred_loss_mask.sum() / pred_loss_mask.size(0)
            loss_dict["pred_rate"] = pred_rate
            pred_loss_mask = pred_loss_mask.squeeze(1)
            pred_loss = pred_loss * pred_loss_mask
            # pred_loss = pred_loss.squeeze(1)
            
            if not t0_always: # training mode
                if self.freeze_gradient and random_prop:
                    loss = 0 # do not generation
                    self.dynamics.eval() # freeze backbone, when do the property prediction.
                elif self.freeze_gradient:
                    pred_loss = 0 # do not do the property prediction when random seed is not less than 0.5
                    self.dynamics.train() # unfreeze the backbone
                else:
                    self.dynamics.train() # unfreeze the backbone
            
            # dynamic adjust the weight
            # pred_loss_weight = (error.mean() / pred_loss.mean()).item()
            pred_loss_weight = 1
            
            loss_dict['pred_loss'] = pred_loss * pred_loss_weight
            loss += pred_loss              

        if self.atom_type_pred: # True !
            pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
            pred_loss_mask = pred_loss_mask.squeeze(1)
            atom_type_loss = atom_type_loss * pred_loss_mask
            loss_dict["atom_type_loss"] = atom_type_loss
            loss += atom_type_loss
        else:
            loss_dict['loss'] = loss.squeeze()


        return loss, loss_dict   


    def evaluate_property(self, x, h, org_context, node_mask=None, edge_mask=None):
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        batch_size = x.size(0)
        t_int = torch.ones((batch_size, 1), device=x.device).float() # t_int all zero
        s_int = t_int - 1
        s_array = s_int / self.T
        t_array = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s_array), x)
        gamma_t = self.inflate_batch_array(self.gamma(t_array), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps
        
        new_context = None
        
        
        # perpare the context, z_t, keep unchanged, copy the sample method
        for s in reversed(range(0, self.T)):
            n_samples= batch_size

            s_array2 = torch.full((n_samples, 1), fill_value=s, device=x.device)
            t_array2 = s_array2 + 1
            s_array2 = s_array2 / self.T
            t_array2 = t_array2 / self.T
            
            # sample new_context
            if new_context is None:
                new_context = utils.sample_gaussian_with_mask(
                    size=(batch_size, 1, org_context.size(-1)), device=node_mask.device,
                    node_mask=node_mask)
            
            z, new_context = self.sample_p_zs_given_zt(
                    s_array, t_array, z_t, node_mask, edge_mask, new_context, fix_noise=False, 
                    yt=t_array2, ys=s_array2, force_t_zero=True) 
            # z_t and t keep unchanged
        
        # calcuate the mae between new_context and org_context
        mae = torch.mean(torch.abs(new_context - org_context))
        
        return new_context, mae
    

    def forward(self, *args, **kwargs):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        ################################################################

        # 如果 DataParallel 传进来的是一个 tuple/list
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        # 解包参数
        x, h, lengths, angles, node_mask, edge_mask, context = args[:7]
        mask_indicator = kwargs.get("mask_indicator", None)
        expand_diff = kwargs.get("expand_diff", False)
        property_label = kwargs.get("property_label", None)
        bond_info = kwargs.get("bond_info", None)

        if self.property_pred:
            assert property_label is not None, "property_label should not be None in training"

        ################################################################

        x, h, delta_log_px = self.normalize_cartesian_pos_with_h(x, h, node_mask)
        lengths, angles, delta_log_pl, delta_log_pa = self.normalize_lengths_angles(lengths, angles)
        delta_log_pl = torch.tensor(delta_log_pl, device=lengths.device, dtype=lengths.dtype)
        delta_log_pa = torch.tensor(delta_log_pa, device=angles.device, dtype=angles.dtype)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)
            delta_log_pl = torch.zeros_like(delta_log_pl)
            delta_log_pa = torch.zeros_like(delta_log_pa)

        ###############################################################
        # compute loss

        if self.training:
            loss, loss_dict = self.compute_loss(x, h, lengths, angles, node_mask, edge_mask, context, t0_always=False, 
                                                mask_indicator=mask_indicator, property_label=property_label, bond_info=bond_info)
        else:
            loss, loss_dict = self.compute_loss(x, h, lengths, angles, node_mask, edge_mask, context, t0_always=True, 
                                                mask_indicator=mask_indicator, property_label=property_label, bond_info=bond_info)

        neg_log_pxhla = loss
        delta_log_pxla = delta_log_px + delta_log_pl + delta_log_pa

        # Correct for normalization on x, l, a.
        assert neg_log_pxhla.size() == delta_log_pxla.size()
        neg_log_pxhla = neg_log_pxhla - delta_log_pxla
        
        return neg_log_pxhla, loss_dict
        
    
    def sample_p_zs_given_zt(self, s, t, zt, rl, ra, 
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
        if self.property_pred:
            (eps_t, pred) = self.phi_xh(zt, t, rl, ra, node_mask, edge_mask, context)
        else:
            eps_t = self.phi_xh(zt, t, rl, ra, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        if pesudo_context is not None and (t*1000)[0].item() < 100: # growth stage?
            with torch.enable_grad():
                loss_fn = torch.nn.L1Loss()
                zt = zt.clone().detach().requires_grad_(True)

                if (t*1000)[0].item() < 10: # growth stage?
                    its = 20
                    opt = torch.optim.SGD([zt], lr=0.001)
                else:
                    its = 5
                    opt = torch.optim.Adam([zt], lr=0.001)

                for i in range(its):
                    self.dynamics.zero_grad()
                    # Compute gamma_s and gamma_t via the network.
                    # Compute alpha_t and sigma_t from gamma.
                    gamma_s = self.inflate_batch_array(self.gamma(s), zt)
                    gamma_t = self.inflate_batch_array(self.gamma(t), zt)
                    alpha_t = self.alpha(gamma_t, zt)
                    sigma_t = self.sigma(gamma_t, zt)
                    
                    if zt.shape[-1] != eps_t.shape[-1]:
                        eps_tmp = eps_t[:,:,:3].clone().detach()
                    else:
                        eps_tmp = eps_t.clone().detach()
    
                    z0 = (zt * node_mask - sigma_t * eps_tmp) / alpha_t
                    t0 = torch.ones_like(t) * 0.001       
                    (_, pred) = self.phi_xh(z0, t0, rl, ra, node_mask, edge_mask, context)
                    
                    loss = loss_fn(pred, pesudo_context)

                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
        
        if self.atom_type_pred:
            atom_type_pred = eps_t[:, :, 3:]
            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t[:,:,0:3]
        else:
            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
       
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        if self.atom_type_pred:
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                atom_type_pred], dim=2
            )
        else:
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                zs[:, :, self.n_dims:]], dim=2
            )
        
        return zs

    
    def sample_p_zs_given_zt_annel_lang(self, s, t, zt, node_mask, edge_mask, context, 
                                        fix_noise=False, yt=None, ys=None, force_t_zero=False,
                                        force_t2_zero=False, T2=10, sigma_n=0.04):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # print(f't is {t[0].item()}, sigma: {sigma[0].item()}, z coeffient: 
        # {(1 / alpha_t_given_s)[0][0].item()}, nn output coeffient: 
        # {(sigma2_t_given_s / alpha_t_given_s / sigma_t)[0][0].item()}')
        
        # Project down to avoid numerical runaway of the center of gravity.
        
        # split the zs into two parts
        z_coord = zs[:, :, :self.n_dims]
        z_atomtype = zs[:, :, self.n_dims:]
        
        
        step = self.T // T2
        for i in range(T2):
            # caluate the mean
            gamma_idx = step * i
            gamma_t = torch.tensor(self.gamma_lst[gamma_idx], dtype=torch.float32, device=zs.device)
            eps_coord = eps_t[:,:,:self.n_dims]
            
            
            z_coord_mu = z_coord - (1 /sigma_n) * gamma_t * eps_coord
            # + torch.sqrt(2 * gamma_t) * zs_coords
            z_coord_sigma = torch.sqrt(2 * gamma_t).repeat(z_coord_mu.size(0), 1, 1)
            
            zs_coords = self.sample_normal(z_coord_mu, z_coord_sigma, node_mask, fix_noise, only_coord=True)
            
            # update the zs and the eps_t
            zs = torch.cat([zs_coords, z_atomtype], dim=2)
            eps_t = self.phi(zs, t, node_mask, edge_mask, context)
            
            # pass
        
        zs = torch.cat([zs[:, :, :self.n_dims], zs[:, :, self.n_dims:]], dim=2)
        
        return zs

    def sample_p_ys_given_yt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        if self.property_pred:
            eps_t, pred = self.phi(zt, t, node_mask, edge_mask, context)
        else:
            eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat([zs[:, :, :self.n_dims], zs[:, :, self.n_dims:]], dim=2)
        return zs


    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
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
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x


    @torch.no_grad()
    def sample(self, LatticeGenModel, n_samples, n_nodes, node_mask, edge_mask, context, 
               fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None):
        """
        Draw samples from the generative model.
        """
        print('use LatticeGenModel to sample l and a, beginning...')
        rl, ra = LatticeGenModel.sample(n_samples, device='cpu', fix_noise=fix_noise)
        rl = rl.to(node_mask.device)
        ra = ra.to(node_mask.device)
        print('sample lengths and angles done.')

        if fix_noise:
            print("using fixed noise......")
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        print('sample T', self.T)
        print('sample x and h beginning...')

        # for s in reversed(range(0, self.T)):
        for s in tqdm(reversed(range(0, self.T)), desc="Sampling diffusion steps"):
            try:
                s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                t_array = s_array + 1
                s_array = s_array / self.T
                t_array = t_array / self.T
                                
                if self.atom_type_pred:
                    z[:, :, self.n_dims:] = 1 # 默认 h_cat 全为 1，不起作用
                    z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], rl, ra, node_mask, edge_mask, 
                                                context, fix_noise=fix_noise, pesudo_context=pesudo_context)                    
                else:
                    z = self.sample_p_zs_given_zt(s_array, t_array, z, rl, ra, node_mask, edge_mask, 
                                                context, fix_noise=fix_noise)
                utils.safe_synchronize()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out Of Memory occurred at step {s}")
                    utils.print_memory("Before crash")
                    torch.cuda.empty_cache()
                    raise
                
        # Finally sample p(x, h | z_0).
        if self.property_pred:
            if self.atom_type_pred:
                z[:,:,self.n_dims:] = 1 # 默认 h_cat 全为 1，这一步预测
            x, h, pred= self.sample_p_xh_lengths_angles_given_z0(
                z, rl, ra, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        else:
            if self.atom_type_pred:
                z[:,:,self.n_dims:] = 1 # 默认 h_cat 全为 1，这一步预测
            x, h = self.sample_p_xh_lengths_angles_given_z0(
                z, rl, ra, node_mask, edge_mask, context, fix_noise=fix_noise
            )
        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)
        print('sample xh done.')

        if self.property_pred:
            return x, h, pred, rl, ra
        return x, h, rl, ra


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
