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


class VariationalDiffusion_L_another(torch.nn.Module):
    """
    The EquiTransformer Diffusion Module.
    """
    def __init__(
            self,
            in_node_nf: int, n_dims: int,
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
        self.prediction_threshold_t = prediction_threshold_t
        
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
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma_lengths = GammaNetwork()
            self.gamma_angles = GammaNetwork()
        else:
            self.gamma_lengths = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision, print_info=False)
            self.gamma_angles = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision, print_info=False)
            print("Using predefined noise schedule for gamma, l and a is same as x.")

        # The network that will predict the denoising.

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
        
        bins = 9
        k_c, self.k_l, self.k_r = self.get_k_params(bins)
        self.K_c = torch.tensor(k_c).to(device)
        
        self.t_min = 0.0001
        
        self.bfn_str = bfn_str
        self.str_loss_type = str_loss_type
        self.str_schedule_norm = str_schedule_norm
        self.temp_index = temp_index
        self.saved_grad = None

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
        gamma_0l = self.gamma_lengths(zeros)
        gamma_0a = self.gamma_angles(zeros)
        sigma_0l = self.sigma(gamma_0l, target_tensor=zeros).item()
        sigma_0a = self.sigma(gamma_0a, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[3], self.norm_values[4])

        if sigma_0l * num_stdevs > 1. / max_norm_value or sigma_0a * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0l:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, zl, za, t):
        """noise predict network"""   

        net_eps_l = self.length_mlp(zl, t)
        net_eps_a = self.angle_mlp(za, t)

        return net_eps_l, net_eps_a
    

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


    def normalize_lengths_angles(self, lengths, angles):
        B = lengths.size(0)
        lengths_norm = (lengths - self.norm_biases[3]) / self.norm_values[3]
        angles_norm = (angles - self.norm_biases[4]) / self.norm_values[4]
        delta_log_pl = -self.len_dim * np.log(self.norm_values[3]) * torch.ones(B, device=lengths.device)
        delta_log_pa = -self.angle_dim * np.log(self.norm_values[4]) * torch.ones(B, device=lengths.device)
        return lengths_norm, angles_norm, delta_log_pl, delta_log_pa
    

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


    def kl_prior_lengths_angles(self, lengths, angles, lambda_l=0.1, lambda_a=0.1):
        """
        计算KL散度
        """
        # lengths部分
        ones_length = torch.ones((lengths.size(0), 1), device=lengths.device)
        gamma_T_length = self.gamma_lengths(ones_length)
        alpha_T_length = self.alpha(gamma_T_length, lengths)
        mu_T_length = alpha_T_length * lengths
        sigma_T_length = self.sigma(gamma_T_length, mu_T_length)
        zeros_length = torch.zeros_like(mu_T_length)
        ones_length_tensor = torch.ones_like(sigma_T_length)
        # 这里直接对所有维度求和
        kl_distance_length = sum_except_batch(
            torch.log(ones_length_tensor / sigma_T_length)
            + 0.5 * (sigma_T_length ** 2 + (mu_T_length - zeros_length) ** 2) / (ones_length_tensor ** 2)
            - 0.5
        )
        
        # angles部分
        ones_angle = torch.ones((angles.size(0), 1), device=angles.device)
        gamma_T_angle = self.gamma_angles(ones_angle)
        alpha_T_angle = self.alpha(gamma_T_angle, angles)
        mu_T_angle = alpha_T_angle * angles
        sigma_T_angle = self.sigma(gamma_T_angle, mu_T_angle)
        zeros_angle = torch.zeros_like(mu_T_angle)
        ones_angle_tensor = torch.ones_like(sigma_T_angle)
        kl_distance_angle = sum_except_batch(
            torch.log(ones_angle_tensor / sigma_T_angle)
            + 0.5 * (sigma_T_angle ** 2 + (mu_T_angle - zeros_angle) ** 2) / (ones_angle_tensor ** 2)
            - 0.5
        )

        return kl_distance_length * lambda_l + kl_distance_angle * lambda_a

    
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


    def compute_error_mp20(self, lengths_out, angles_out, eps_l, eps_a, lambda_l=0.1, lambda_a=0.1):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t_lengths = lengths_out
        eps_t_angles = angles_out

        l_error = sum_except_batch((eps_l - eps_t_lengths) ** 2) * lambda_l / self.len_dim
        a_error = sum_except_batch((eps_a - eps_t_angles) ** 2) * lambda_a / self.angle_dim
        error = l_error + a_error
            
        return error, l_error, a_error
    

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


    def log_constants_p_la_given_z0(self, batch_size, device):
        degrees_of_freedom_l = self.len_dim
        degrees_of_freedom_a = self.angle_dim
        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0_l = self.gamma_lengths(zeros)
        gamma_0_a = self.gamma_angles(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_l = 0.5 * gamma_0_l.view(batch_size)
        log_sigma_a = 0.5 * gamma_0_a.view(batch_size)

        return degrees_of_freedom_l * (- log_sigma_l - 0.5 * np.log(2 * np.pi)) + \
               degrees_of_freedom_a * (- log_sigma_a - 0.5 * np.log(2 * np.pi))


    def sample_p_la_given_z0la(self, z0_l, z0_a, fix_noise=False):
        """Samples l,a ~ p(x|z0_l,z0_a)."""
        bs = z0_l.size(0)
        zeros = torch.zeros(size=(bs, 1), device=z0_l.device)
        gamma_0_l = self.gamma_lengths(zeros)
        gamma_0_a = self.gamma_angles(zeros)
        sigma_l = self.SNR(-0.5 * gamma_0_l)
        sigma_a = self.SNR(-0.5 * gamma_0_a)

        l_out, a_out = self.phi(z0_l, z0_a, zeros)

        # Compute mu for p(zs | zt).
        mu_l = self.compute_length_pred(l_out, z0_l, gamma_0_l)
        mu_a = self.compute_angle_pred(a_out, z0_a, gamma_0_a)
        
        # Sample lengths and angles
        lengths = self.sample_normal_length(mu=mu_l, sigma=sigma_l, fix_noise=fix_noise)
        angles = self.sample_normal_angle(mu=mu_a, sigma=sigma_a, fix_noise=fix_noise)

        # unnormalize l,a
        lengths, angles = self.unnormalize_lengths_angles(lengths, angles)

        return lengths, angles

    
    def sample_normal_length(self, mu, sigma, fix_noise=False):
        bs = 1 if fix_noise else mu.size(0)
        lengths_dim = mu.size(1)
        eps = self.sample_length_noise(bs, lengths_dim, mu.device)
        return mu + sigma * eps
    
    def sample_normal_angle(self, mu, sigma, fix_noise=False):
        bs = 1 if fix_noise else mu.size(0)
        angles_dim = mu.size(1)
        eps = self.sample_angle_noise(bs, angles_dim, mu.device)
        return mu + sigma * eps
    

    def log_p_la_given_z0_without_constants(
        self, lengths, angles, lengths_t, angles_t, gamma_0_length, gamma_0_angle,
        eps_l, eps_a, lengths_out, angles_out, lambda_l=0.1, lambda_a=0.1
    ):
        """
        计算 lengths, angles 在给定 z0 下的log likelihood
        """

        # lengths 部分
        net_l = lengths_out
        log_p_length_given_z = -0.5 * sum_except_batch((eps_l - net_l) ** 2)
        # print("log_p_length_given_z: ", log_p_length_given_z.sum(0))

        # angles 部分
        net_a = angles_out
        log_p_angle_given_z = -0.5 * sum_except_batch((eps_a - net_a) ** 2)
        # print("log_p_angle_given_z: ", log_p_angle_given_z.sum(0))

        # 合并所有对数似然
        log_p_lengths_angles_given_z = (
            log_p_length_given_z * lambda_l + log_p_angle_given_z * lambda_a
        )
        return log_p_lengths_angles_given_z


    def compute_loss(self, lengths, angles, t0_always):

        batch_size = lengths.size(0)
        device = lengths.device
        # whether to include loss term 0 always.
        if t0_always:
            lowest_t = 1
        else:
            lowest_t = 0

        # 时间步t采样范围: [lowest_t, T], 采样结果变量t_int: [B, 1]
        t_int = torch.randint(
                lowest_t, self.T + 1, size=(batch_size, 1), device=device).float()

        if self.half_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(batch_size - half_batch_size, 1), device=device).float()
            t_int[:half_batch_size,:] = torch.randint(lowest_t, self.T + 1, size=(half_batch_size, 1), device=device).float()
        
        if self.sep_noisy_node:
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(batch_size - half_batch_size, 1), device=device).float()
            t_int[:half_batch_size,:] = torch.randint(self.prediction_threshold_t + 1, self.T + 1, 
                                                      size=(half_batch_size, 1), device=device).float()
        
        s_int = t_int - 1 
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t and s to [0, 1]. Note that the negative step 
        # of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1, t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s_length = self.inflate_batch_array(self.gamma_lengths(s), lengths)
        gamma_s_angle = self.inflate_batch_array(self.gamma_angles(s), angles)
        gamma_t_length = self.inflate_batch_array(self.gamma_lengths(t), lengths)
        gamma_t_angle = self.inflate_batch_array(self.gamma_angles(t), angles)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t_length = self.alpha(gamma_t_length, lengths)
        alpha_t_angle = self.alpha(gamma_t_angle, angles)
        sigma_t_length = self.sigma(gamma_t_length, lengths)
        sigma_t_angle = self.sigma(gamma_t_angle, angles)
        
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_length = self.sample_length_noise(n_samples=batch_size, length_dim=lengths.size(1), device=lengths.device)
        eps_angle = self.sample_angle_noise(n_samples=batch_size, angle_dim=angles.size(1), device=angles.device)

        # Sample z_t
        z_t_length = alpha_t_length * lengths + sigma_t_length * eps_length
        z_t_angle = alpha_t_angle * angles + sigma_t_angle * eps_angle
      
        # Neural net prediction, the noisy zt is input
        lengths_out, angles_out = self.phi(z_t_length, z_t_angle, t)

        # Compute the error.
        error, l_error, a_error = self.compute_error_mp20(lengths_out, angles_out, \
            eps_length, eps_angle, self.lambda_l, self.lambda_a)

        if self.training and self.loss_type == 'l2':
            SNR_weight_l = torch.ones_like(l_error)
            SNR_weight_a = torch.ones_like(a_error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight_l = (self.SNR(gamma_s_length - gamma_t_length) - 1).squeeze(1)
            SNR_weight_a = (self.SNR(gamma_s_angle - gamma_t_angle) - 1).squeeze(1)
        assert l_error.size() == SNR_weight_l.size()
        assert a_error.size() == SNR_weight_a.size()
        loss_t_larger_than_zero = 0.5 * (SNR_weight_l * l_error + 
                                         SNR_weight_a * a_error)

        neg_log_constants = -self.log_constants_p_la_given_z0(batch_size, device)
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior_lengths_angles(lengths, angles, self.lambda_l, self.lambda_a)

        # Combining the terms
        if t0_always:   # test
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0_length = self.inflate_batch_array(self.gamma_lengths(t_zeros), lengths)
            gamma_0_angle = self.inflate_batch_array(self.gamma_angles(t_zeros), angles)
            alpha_0_length = self.alpha(gamma_0_length, lengths)
            alpha_0_angle = self.alpha(gamma_0_angle, angles)
            sigma_0_length = self.sigma(gamma_0_length, lengths)
            sigma_0_angle = self.sigma(gamma_0_angle, angles)

            # Sample z_0 given x,l,a,h for timestep t, from q(z_t | x,l,a,h)
            eps_0_length = self.sample_length_noise(n_samples=batch_size, length_dim=lengths.size(1), device=lengths.device)
            eps_0_angle = self.sample_angle_noise(n_samples=batch_size, angle_dim=angles.size(1), device=angles.device)

            z_0_length = alpha_0_length * lengths + sigma_0_length * eps_0_length
            z_0_angle = alpha_0_angle * angles + sigma_0_angle * eps_0_angle

            lengths_out, angles_out = self.phi(z_0_length, z_0_angle, t_zeros)

            # Compute the error for t = 0.            
            loss_term_0 = -self.log_p_la_given_z0_without_constants(
                lengths, angles, z_0_length, z_0_angle, gamma_0_length, gamma_0_angle,
                eps_0_length, eps_0_angle, lengths_out, angles_out, 
                lambda_l=self.lambda_l, lambda_a=self.lambda_a)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            # Combine all terms.
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_p_la_given_z0_without_constants(
                lengths, angles, lengths_out, angles_out, gamma_t_length, gamma_t_angle,
                eps_length, eps_angle, lengths_out, angles_out, 
                lambda_l=self.lambda_l, lambda_a=self.lambda_a)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

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
                     'total_error': error.squeeze(),
                     'l_error': l_error.squeeze(),
                     'a_error': a_error.squeeze(),
                     'kl_prior': kl_prior.squeeze(),
                     'neg_log_constants': neg_log_constants.squeeze(),
                     'estimator_loss_terms': estimator_loss_terms.squeeze(),
                     'loss_term_0': loss_term_0.squeeze(),
                     'loss_t_larger_than_zero': loss_t_larger_than_zero.squeeze()}

        return loss, loss_dict   


    def forward(self, *args, **kwargs):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # 如果 DataParallel 传进来的是一个 tuple/list
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        # 解包参数
        lengths, angles = args[:2]

        lengths, angles, delta_log_pl, delta_log_pa = self.normalize_lengths_angles(lengths, angles)
        delta_log_pl = torch.tensor(delta_log_pl, device=lengths.device, dtype=lengths.dtype)
        delta_log_pa = torch.tensor(delta_log_pa, device=angles.device, dtype=angles.dtype)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_pl = torch.zeros_like(delta_log_pl)
            delta_log_pa = torch.zeros_like(delta_log_pa)

        # compute loss
        if self.training:
            loss, loss_dict = self.compute_loss(lengths, angles, t0_always=False)
        else:
            loss, loss_dict = self.compute_loss(lengths, angles, t0_always=True)

        neg_log_pla = loss
        delta_log_pla = delta_log_pl + delta_log_pa

        # Correct for normalization on x, l, a.
        assert neg_log_pla.size() == delta_log_pla.size()
        neg_log_pla = neg_log_pla - delta_log_pla
        
        return neg_log_pla, loss_dict
        

    def lattice_sample_p_zs_given_zt(self, s, t, zt_l, zt_a, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s_length = self.gamma_lengths(s)
        gamma_s_angle = self.gamma_angles(s)
        gamma_t_length = self.gamma_lengths(t)
        gamma_t_angle = self.gamma_angles(t)

        sigma2_t_given_s_length, sigma_t_given_s_length, alpha_t_given_s_length = \
                self.sigma_and_alpha_t_given_s(gamma_t_length, gamma_s_length, zt_l)
        sigma2_t_given_s_angle, sigma_t_given_s_angle, alpha_t_given_s_angle = \
                self.sigma_and_alpha_t_given_s(gamma_t_angle, gamma_s_angle, zt_a)
        sigma_s_length = self.sigma(gamma_s_length, target_tensor=zt_l)
        sigma_t_length = self.sigma(gamma_t_length, target_tensor=zt_l)
        sigma_s_angle = self.sigma(gamma_s_angle, target_tensor=zt_a)
        sigma_t_angle = self.sigma(gamma_t_angle, target_tensor=zt_a)

        # Neural net prediction for l and a.
        eps_t_l, eps_t_a = self.phi(zt_l, zt_a, t)

        mu_l = zt_l / alpha_t_given_s_length - \
            (sigma2_t_given_s_length / alpha_t_given_s_length / sigma_t_length) * eps_t_l
        mu_a = zt_a / alpha_t_given_s_angle - \
            (sigma2_t_given_s_angle / alpha_t_given_s_angle / sigma_t_angle) * eps_t_a
 
        # Compute sigma for p(zs | zt).
        sigma_l = sigma_t_given_s_length * sigma_s_length / sigma_t_length
        sigma_a = sigma_t_given_s_angle * sigma_s_angle / sigma_t_angle
       
        # Sample zs given the paramters derived from zt.
        zs_l = self.sample_normal_length(mu_l, sigma_l, fix_noise)
        zs_a = self.sample_normal_angle(mu_a, sigma_a, fix_noise)

        return zs_l, zs_a
    

    def sample_combined_length_angle_noise(self, n_samples, length_dim, angle_dim, device):
        z_lengths = self.sample_length_noise(n_samples, length_dim, device)
        z_angles = self.sample_angle_noise(n_samples, angle_dim, device)
        return z_lengths, z_angles


    def sample_length_noise(self, n_samples, length_dim, device):
        """
        Samples mean-centered normal noise for lengths.
        """
        z_lengths = utils.sample_gaussian(
            size=(n_samples, length_dim), device=device)
        return z_lengths
    

    def sample_angle_noise(self, n_samples, angle_dim, device):
        """
        Samples mean-centered normal noise for angles.
        """
        z_angles = utils.sample_gaussian(
            size=(n_samples, angle_dim), device=device)
        return z_angles


    @torch.no_grad()
    def sample(self, n_samples, device, fix_noise=False):

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            print("using fixed noise......")
            z_l, z_a = self.sample_combined_length_angle_noise(1, self.len_dim, self.angle_dim, device) # [1,3]
            z_l = z_l.repeat(1, n_samples).view(n_samples, -1)
            z_a = z_a.repeat(1, n_samples).view(n_samples, -1)
        else:
            z_l, z_a = self.sample_combined_length_angle_noise(n_samples, self.len_dim, self.angle_dim, device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        print('sample T',self.T)
        print('sample l and a beginning...')

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z_l, z_a = self.lattice_sample_p_zs_given_zt(s_array, t_array, z_l, z_a, fix_noise)

        # sample l and a, to construct lattice first.
        rl, ra = self.sample_p_la_given_z0la(z_l, z_a, fix_noise)

        print('sample l and a done.')

        return rl, ra


    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0l = self.gamma_lengths(torch.zeros(1, device=self.buffer.device))
        gamma_1l = self.gamma_lengths(torch.ones(1, device=self.buffer.device))
        gamma_0a = self.gamma_angles(torch.zeros(1, device=self.buffer.device))
        gamma_1a = self.gamma_angles(torch.ones(1, device=self.buffer.device))

        log_SNR_maxl = -gamma_0l
        log_SNR_minl = -gamma_1l
        log_SNR_maxa = -gamma_0a
        log_SNR_mina = -gamma_1a

        info = {
            'log_SNR_maxl': log_SNR_maxl.item(),
            'log_SNR_minl': log_SNR_minl.item(),
            'log_SNR_maxa': log_SNR_maxa.item(),
            'log_SNR_mina': log_SNR_mina.item(),
        }
        print(info)

        return info
    