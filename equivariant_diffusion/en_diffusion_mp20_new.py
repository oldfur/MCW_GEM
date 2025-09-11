from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.mlp import DiffusionMLP
from torch_scatter import scatter_mean


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
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
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
    def __init__(self, noise_schedule, timesteps, precision):
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
        
        print("Predefined noise schedule:")
        # 设置打印选项，只显示头部和尾部
        np.set_printoptions(threshold=10, edgeitems=10)
        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

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


def cdf_standard_gaussian(x):   # 计算标准正态分布的累积分布函数（CDF）
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion_new(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_MP20, in_node_nf: int, n_dims: int,
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
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

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
        self.la_mlp = DiffusionMLP(input_dim=self.len_dim + self.angle_dim, 
                                   output_dim=self.len_dim + self.angle_dim, 
                                   time_emb_dim=16, hidden_dims=[128, 128, 256], 
                                   use_self_attn=False)
    
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

    def phi(self, x, lengths, angles, t, node_mask, edge_mask, context, t2=None, mask_y=None):
        # noise predict network
        # 噪声预测网络,用于diffusion的去噪过程
        # EGNN_dynamics_MP20 for diffusion

        if self.relay_sampling == 0:
            net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y)
            la_out = self.la_mlp(torch.cat([lengths, angles], dim=1), t)
        else:
            if t > self.sampling_threshold_t:
                net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y)
                la_out = torch.cat([lengths, angles], dim=1)
            else:
                print("relay_sampling t: ", t)
                assert isinstance(self.second_dynamics, models.EGNN_dynamics_MP20), \
                    "second_dynamics should be an instance of models.EGNN_dynamics_MP20"
                net_out = self.second_dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y)
                # lengths_out = self.second_dynamics.length_mlp(lengths)
                # angles_out = self.second_dynamics.angle_mlp(angles)


        lengths_out, angles_out = la_out[:, :self.len_dim], la_out[:, self.len_dim:]

        return net_out, lengths_out, angles_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.

        解释：
        inflate_batch_array 方法用于调整输入张量 array 的形状，使其能够与另一个张量 target 在后续操作中实现自动广播。具体来说，
        array 通常是一维的（形状为 (batch_size,)），而 target 可能有更多的维度（如 (batch_size, num_nodes, n_dims)）。为了让
        array 能在与 target 做逐元素运算时自动扩展到相同的 batch 维度，函数会将 array 重塑（view）为 (batch_size, 1, ..., 1)，
        其中 1 的个数等于 target 除 batch 维外的维度数。

        实现上，target_shape 通过 (array.size(0),) 加上 (1,) * (len(target.size()) - 1) 构造，确保 batch 维一致，其余
        维度为1。最后用 array.view(target_shape) 返回调整后的张量。这样，array 在与 target 做加减乘除等操作时，会自动在非 batch 
        维上广播，避免手动扩展维度，代码更简洁高效。这种技巧在深度学习中非常常见，尤其是在处理 batch-wise 的标量或向量与高维张量运算时。
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
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        # print("before norm h_cat", h['categorical'][0])
        # print("norm_bias", self.norm_biases[1])
        # print("norm_values", self.norm_values[1])
        # print("node_mask", node_mask[0])
        # print("after norm h_cat", h_cat[0])
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    
    def normalize_lengths_angles(self, lengths, angles):
        """
        对 lengths 和 angles 进行归一化，假设 norm_values[3]、norm_values[4] 和 norm_biases[3]、norm_biases[4] 已定义。
        输入:
            lengths: [batch_size, 3]
            angles: [batch_size, 3]
        输出:
            lengths_norm: 归一化后的 lengths
            angles_norm: 归一化后的 angles
        """
        lengths_norm = (lengths - self.norm_biases[3]) / self.norm_values[3]
        angles_norm = (angles - self.norm_biases[4]) / self.norm_values[4]
        return lengths_norm, angles_norm
    
    
    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_lengths_angles(self, lengths, angles):
        """
        对 lengths 和 angles 进行反归一化，假设 norm_values[3]、norm_values[4] 和 norm_biases[3]、norm_biases[4] 已定义。
        输入:
            lengths: [batch_size, 3]
            angles: [batch_size, 3]
        输出:
            lengths: 反归一化后的 lengths
            angles: 反归一化后的 angles
        """
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
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
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

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
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

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            kl_distance_h = 0.0
        return kl_distance_x + kl_distance_h


    def kl_prior_with_lengths_angles(self, xh, lengths, angles, node_mask):
        """
        计算 q(zT | x) 与 p(zT) = N(0, 1) 的KL散度，包含lengths和angles两个特征。
        """
        # x, h部分
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_h)
        zeros, ones_tensor = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones_tensor, node_mask)
        zeros, ones_tensor = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones_tensor, d=subspace_d)
        
        # lengths部分
        ones_length = torch.ones((lengths.size(0), 1), device=lengths.device)
        gamma_T_length = self.gamma(ones_length)
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
        gamma_T_angle = self.gamma(ones_angle)
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
        
        # print("kl_distance_x: ", kl_distance_x.sum(0))
        # print("kl_distance_length: ", kl_distance_length.sum(0))
        # print("kl_distance_angle: ", kl_distance_angle.sum(0))

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            kl_distance_h = 0.0

        return kl_distance_x + kl_distance_h + \
            self.lambda_l * kl_distance_length + \
            self.lambda_a * kl_distance_angle
    

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
        if self.dynamics.mode == "PAT" or self.atom_type_pred: # get x_only for PAT loss
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]
     
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error    # [B]


    def compute_error_mp20(self, net_out, lengths_out, angles_out, eps, eps_l, eps_a):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        eps_t_lengths = lengths_out
        eps_t_angles = angles_out
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dimsphi]

        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            x_error = sum_except_batch((eps - eps_t) ** 2) / denom
            l_error = sum_except_batch((eps_l - eps_t_lengths) ** 2) * self.lambda_l / self.len_dim
            a_error = sum_except_batch((eps_a - eps_t_angles) ** 2) * self.lambda_a / self.angle_dim
            error = x_error + l_error + a_error
            # print("x error is :", x_error.sum(0))
            # print("length error is :", l_error.sum(0))
            # print("angle error is :", a_error.sum(0))
        else:   # test performance
            x_error = sum_except_batch((eps - eps_t) ** 2)
            l_error = sum_except_batch((eps_l - eps_t_lengths) ** 2)
            a_error = sum_except_batch((eps_a - eps_t_angles) ** 2)
            error = x_error + l_error + a_error
            # print("x test error is :", x_error.sum(0))
            # print("length test error is :", l_error.sum(0))    
            # print("angle test error is :", a_error.sum(0))
            
        return error
    
    def show_x_error(self, net_out, eps):
        eps_t = net_out
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]

        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
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

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        
        if self.bond_pred:
            net_out, pred, edge_index_knn = self.phi(z0, zeros, node_mask, edge_mask, context)
        elif self.property_pred:
            net_out, pred = self.phi(z0, zeros, node_mask, edge_mask, context)
        else:
            net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            xh = self.sample_normal(mu=mu_x[:,:,:3], sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        else:
            xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            h_cat = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
            h_int = net_out[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        else:
            x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)
        # print(f"h_cat: {h_cat.shape}, h_int: {h_int.shape}")
        # print(f"h_cat: {h_cat}, h_int: {h_int}")
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}
        if self.bond_pred:
            bond_types = torch.argmax(pred[1], dim=1)
            
            nonzero_mask = bond_types != 0
            bond_types = bond_types[nonzero_mask].cpu().numpy()
            edge_index_knn = edge_index_knn[:,nonzero_mask].cpu().numpy()
            
            batch_size = h_int.shape[0]
            # parse knn_edge_index
            atoms_nums = node_mask.sum(dim=1).squeeze().tolist()
            acc_num_lst = []
            acc_num = 0
            
            edge_bond_lst = []
            iter_start = 0
            for i in range(len(atoms_nums)):
                acc_num_lst.append(acc_num)
                acc_num += atoms_nums[i]
                
                c_edge_lst = []
                c_bond_lst = []
                while iter_start < edge_index_knn.shape[1]:
                    if edge_index_knn[0,iter_start] >= acc_num or edge_index_knn[1,iter_start] >= acc_num:
                        break
                    
                    c_bond_type = bond_types[iter_start]
                    c_edge_index = edge_index_knn[:,iter_start] - acc_num_lst[i]
                    
                    if not [c_edge_index[0], c_edge_index[1]] in c_edge_lst and not [c_edge_index[1], c_edge_index[0]] in c_edge_lst:
                        c_edge_lst.append([c_edge_index[0], c_edge_index[1]])
                        c_bond_lst.append(c_bond_type)
                    iter_start += 1
                edge_bond_lst.append([c_edge_lst, c_bond_lst])
            h = [h, edge_bond_lst]    
        
        if self.property_pred:
            return x, h, pred
        return x, h
    
    def sample_p_xh_lengths_angles_given_z0(self, z0, z0_l, z0_a, node_mask, edge_mask, context, fix_noise=False):
        """Samples x,h,l,a ~ p(x|z0,z0_l,z0_a)."""
        bs = z0.size(0)
        zeros = torch.zeros(size=(bs, 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # 这里的sigma计算方式与compute_loss里的不同,后续有待研究？
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        sigma_l = self.SNR(-0.5 * gamma_0)
        sigma_a = self.SNR(-0.5 * gamma_0)

        if self.bond_pred:
            (net_out, pred, edge_index_knn), l_out, a_out = self.phi(z0, z0_l, z0_a, zeros, node_mask, edge_mask, context)
        elif self.property_pred:
            (net_out, pred), l_out, a_out = self.phi(z0, z0_l, z0_a, zeros, node_mask, edge_mask, context)
        else:
            net_out, l_out, a_out = self.phi(z0, z0_l, z0_a, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        mu_l = self.compute_length_pred(l_out, z0_l, gamma_0)
        mu_a = self.compute_angle_pred(a_out, z0_a, gamma_0)
        
        # Sample from Normal distribution
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            xh = self.sample_normal(mu=mu_x[:,:,:3], sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        else:
            xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            h_cat = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
            h_int = net_out[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        else:
            x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}
        if self.bond_pred:
            bond_types = torch.argmax(pred[1], dim=1)
            
            nonzero_mask = bond_types != 0
            bond_types = bond_types[nonzero_mask].cpu().numpy()
            edge_index_knn = edge_index_knn[:,nonzero_mask].cpu().numpy()
            
            batch_size = h_int.shape[0]
            # parse knn_edge_index
            atoms_nums = node_mask.sum(dim=1).squeeze().tolist()
            acc_num_lst = []
            acc_num = 0
            
            edge_bond_lst = []
            iter_start = 0
            for i in range(len(atoms_nums)):
                acc_num_lst.append(acc_num)
                acc_num += atoms_nums[i]
                
                c_edge_lst = []
                c_bond_lst = []
                while iter_start < edge_index_knn.shape[1]:
                    if edge_index_knn[0,iter_start] >= acc_num or edge_index_knn[1,iter_start] >= acc_num:
                        break
                    
                    c_bond_type = bond_types[iter_start]
                    c_edge_index = edge_index_knn[:,iter_start] - acc_num_lst[i]
                    
                    if not [c_edge_index[0], c_edge_index[1]] in c_edge_lst and not [c_edge_index[1], c_edge_index[0]] in c_edge_lst:
                        c_edge_lst.append([c_edge_index[0], c_edge_index[1]])
                        c_bond_lst.append(c_bond_type)
                    iter_start += 1
                edge_bond_lst.append([c_edge_lst, c_bond_lst])
            h = [h, edge_bond_lst]    
        
        # Sample lengths and angles
        lengths = self.sample_normal_length(mu=mu_l, sigma=sigma_l, fix_noise=fix_noise)
        angles = self.sample_normal_angle(mu=mu_a, sigma=sigma_a, fix_noise=fix_noise)
        lengths, angles = self.unnormalize_lengths_angles(lengths, angles)

        if self.property_pred:
            return x, h, pred, lengths, angles
        return x, h, lengths, angles

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
    
    def sample_normal2(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = utils.sample_gaussian_with_mask(
                size=(mu.size(0), 1, 1), device=node_mask.device,
                node_mask=node_mask)
        
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


    def log_pxh_lengths_angles_given_z0_without_constants(
        self, x, h, lengths, angles, z_t, lengths_t, angles_t, gamma_0, gamma_0_length, gamma_0_angle,
        eps, eps_l, eps_a, net_out, lengths_out, angles_out, node_mask, epsilon=1e-10
    ):
        """
        计算 x, h, lengths, angles 在给定 z0 下的对数似然（去掉常数项）。
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

        # lengths 部分
        net_l = lengths_out
        # sigma_0_length = self.sigma(gamma_0_length, target_tensor=lengths_t)
        # log_p_length_given_z = -0.5 * sum_except_batch(((eps_l - net_l) ** 2) / (sigma_0_length ** 2))
        log_p_length_given_z = -0.5 * sum_except_batch((eps_l - net_l) ** 2)
        # print("log_p_length_given_z: ", log_p_length_given_z.sum(0))

        # angles 部分
        net_a = angles_out
        # sigma_0_angle = self.sigma(gamma_0_angle, target_tensor=angles_t)
        # log_p_angle_given_z = -0.5 * sum_except_batch(((eps_a - net_a) ** 2) / (sigma_0_angle ** 2))
        log_p_angle_given_z = -0.5 * sum_except_batch((eps_a - net_a) ** 2)
        # print("log_p_angle_given_z: ", log_p_angle_given_z.sum(0))

        # 合并所有对数似然
        log_p_xh_lengths_angles_given_z = (
            log_p_x_given_z_without_constants + log_p_h_given_z +
            log_p_length_given_z * self.lambda_l + log_p_angle_given_z * self.lambda_a
        )
        return log_p_xh_lengths_angles_given_z


    def compute_loss_exp(self, x, h, lengths, angles, node_mask, edge_mask, context, t0_always, sigma_0=0.04):
        """expand first and add guassian secondly"""
        batch_size = x.size(0)
        # expand first
        lowest_t = 0
        t_int_c = torch.randint(
            lowest_t, self.T, size=(batch_size, 1), device=x.device).float()

        scale = t_int_c / self.T # 0-1
        
        # first step
        batch_size = x.shape[0]
        
        x_new = x.clone()
        for i in range(batch_size):
            x_new[i] = x[i] * scale[i] # scale molecular for each batch
        
        
        # the atom type part
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            # indicate the training, we can switch the mask indicator
            # if self.pre_training:
            #     self.mask_indicator = not self.mask_indicator
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()

        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1
        # t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        
        xh = torch.cat([x_new, h['categorical'], h['integer']], dim=2)
        
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)

        alpha_tr = alpha_t.repeat(1, xh.shape[1], xh.shape[2])
        sigma_tr = sigma_t.repeat(1, xh.shape[1], xh.shape[2])
        
        # z_t = alpha_tr * xh + sigma_tr * eps
        alpha_tr[:,:, :3] = 1
        sigma_tr[:,:, :3] = sigma_0
        
        z_t = alpha_tr * xh + sigma_tr * eps
        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)
        
        error = self.compute_error(net_out, gamma_t, eps)
        
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(batch_size, x.device, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0 # TODO fix this
            
            
            
            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()
            
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
        

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        

        
        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}        


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

        :param x: [B, N, D], x是坐标,D应为3
        :param h: [B, N, D]
        :param node_mask: [B, N, 1]
        :param edge_mask: [B, N, N]
        :param context: [B, N, D]
        :param t0_always: 是否总是包含loss_0项, 为了计算loss_t, loss_0
        :param mask_indicator: 训练时是否使用mask
        :param property_label: 属性标签
        :param time_upperbond: 时间上限
        :param bond_info: 键值对

        :return: loss, loss_dict
        """

        batch_size = x.size(0)

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            # indicate the training, we can switch the mask indicator
            # if self.pre_training:
            #     self.mask_indicator = not self.mask_indicator
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
        
        s_int = t_int - 1   # s为t的前一个时间步
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t and s to [0, 1]. Note that the negative step 
        # of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1, t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_s_length = self.inflate_batch_array(gamma_s, lengths)
        gamma_s_angle = self.inflate_batch_array(gamma_s, angles)

        gamma_t = self.inflate_batch_array(self.gamma(t), x)
        gamma_t_length = self.inflate_batch_array(gamma_t, lengths)
        gamma_t_angle = self.inflate_batch_array(gamma_t, angles)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        alpha_t_length = self.alpha(gamma_t_length, lengths)
        alpha_t_angle = self.alpha(gamma_t_angle, angles)
        sigma_t = self.sigma(gamma_t, x)
        sigma_t_length = self.sigma(gamma_t_length, lengths)
        sigma_t_angle = self.sigma(gamma_t_angle, angles)
        
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)
        eps_length = self.sample_length_noise(n_samples=batch_size, length_dim=lengths.size(1), device=lengths.device)
        eps_angle = self.sample_angle_noise(n_samples=batch_size, angle_dim=angles.size(1), device=angles.device)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        fix_h = torch.ones_like(torch.cat([h['categorical'], h['integer']], dim=2))

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            z_t = alpha_t * x + sigma_t * eps  
            z_t = torch.cat([z_t, fix_h], dim=2)
            z_t_length = alpha_t_length * lengths + sigma_t_length * eps_length
            z_t_angle = alpha_t_angle * angles + sigma_t_angle * eps_angle
        else:
            z_t = alpha_t * xh + sigma_t * eps
            z_t_length = alpha_t_length * lengths + sigma_t_length * eps_length
            z_t_angle = alpha_t_angle * angles + sigma_t_angle * eps_angle

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        
        # phi为每个时间步的去噪神经网络
        if self.property_pred:
            (net_out, property_pred), lengths_out, angles_out = self.phi(
                z_t, z_t_length, z_t_angle, t, node_mask, edge_mask, context)
        else:
            # Neural net prediction.
            net_out, lengths_out, angles_out = self.phi(z_t, z_t_length, z_t_angle, t, node_mask, edge_mask, context)


        # Compute the error.
        # error = self.compute_error(net_out, gamma_t, eps)
        error = self.compute_error_mp20(net_out, lengths_out, angles_out, eps, eps_length, eps_angle)
        # self.show_x_error(net_out, eps)
        # self.show_l_error(lengths_out, eps_length)
        # self.show_a_error(angles_out, eps_angle)
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(batch_size, x.device, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior_with_lengths_angles(xh, lengths, angles, node_mask)
        # 是否有归一化lengths和angles？

        # Combining the terms
        if t0_always:   # test
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            gamma_0_length = self.inflate_batch_array(gamma_0, lengths)
            gamma_0_angle = self.inflate_batch_array(gamma_0, angles)
            alpha_0 = self.alpha(gamma_0, x)
            alpha_0_length = self.alpha(gamma_0_length, lengths)
            alpha_0_angle = self.alpha(gamma_0_angle, angles)
            sigma_0 = self.sigma(gamma_0, x)
            sigma_0_length = self.sigma(gamma_0_length, lengths)
            sigma_0_angle = self.sigma(gamma_0_angle, angles)

            # Sample z_0 given x,l,a,h for timestep t, from q(z_t | x,l,a,h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)
            eps_0_length = self.sample_length_noise(n_samples=batch_size, length_dim=lengths.size(1), device=lengths.device)
            eps_0_angle = self.sample_angle_noise(n_samples=batch_size, angle_dim=angles.size(1), device=angles.device)

            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z_0 = alpha_0 * x + sigma_0 * eps_0
                z_0 = torch.cat([z_0, fix_h], dim=2)
                z_0_length = alpha_0_length * lengths + sigma_0_length * eps_0_length
                z_0_angle = alpha_0_angle * angles + sigma_0_angle * eps_0_angle
            else:
                z_0 = alpha_0 * xh + sigma_0 * eps_0
                z_0_length = alpha_0_length * lengths + sigma_0_length * eps_0_length
                z_0_angle = alpha_0_angle * angles + sigma_0_angle * eps_0_angle

            if self.property_pred:
                (net_out, property_pred), lengths_out, angles_out = self.phi(z_0, z_0_length, z_0_angle, t_zeros, node_mask, edge_mask, context)
            else:
                net_out, lengths_out, angles_out = self.phi(z_0, z_0_length, z_0_angle, t_zeros, node_mask, edge_mask, context)

            # Compute the error for t = 0.            
            loss_term_0 = -self.log_pxh_lengths_angles_given_z0_without_constants(
                x, h, lengths, angles, z_0, z_0_length, z_0_angle, gamma_0, gamma_0_length, gamma_0_angle,
                eps_0, eps_0_length, eps_0_angle, net_out, lengths_out, angles_out, node_mask
            , lambda_l=self.lambda_l, lambda_a=self.lambda_a)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            # Combine all terms.
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
            # print("kl_prior: ", kl_prior.sum(0))
            # print("estimator_loss_terms: ", estimator_loss_terms.sum(0)) # main loss term!!!!!
            ## print("neg_log_constants: ", neg_log_constants.sum(0))
            # print("loss_term_0: ", loss_term_0.sum(0)) 


        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_lengths_angles_given_z0_without_constants(
                x, h, lengths, angles, z_t, lengths_out, angles_out, gamma_t, gamma_t_length, gamma_t_angle,
                eps, eps_length, eps_angle, net_out, lengths_out, angles_out, node_mask
            , lambda_l=self.lambda_l, lambda_a=self.lambda_a)   # main loss term

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero
            # print("loss_term_0: ", loss_term_0.sum(0))  # main loss term
            # print("loss_t_larger_than_zero: ", loss_t_larger_than_zero.sum(0))

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
            # print("kl_prior: ", kl_prior.sum(0))
            # print("estimator_loss_terms: ", estimator_loss_terms.sum(0))    # main loss term
            ## print("neg_log_constants: ", neg_log_constants.sum(0))
         
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        loss_dict = {'t': t_int.squeeze(), 
                     'loss':loss.squeeze(),
                     'loss_t': loss_t.squeeze(),
                     'error': error.squeeze(),
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
                # print("property_label: ", property_label)
                # print("property_pred: ", property_pred)
                # print("pred_loss", pred_loss)
            else:
                # 0 loss for prediction
                pred_loss = torch.zeros_like(property_pred)

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            # calculate the loss for atom type
            h_true = torch.cat([h['categorical'], h['integer']], 
                               dim=2).clone().detach().requires_grad_(True).to(torch.float32).to(x.device)
            h_pred = net_out[:, :, 3:]
            # 保留第0维度不计算loss，因为第0维度是batch维度
            l1_loss = torch.nn.L1Loss(reduction='none')
            atom_type_loss = l1_loss(h_true, h_pred)
            
            # atom_type_loss = l1_loss(h_true, h_pred)
            atom_type_loss = atom_type_loss * node_mask
            atom_type_loss = atom_type_loss.mean(dim=2).mean(dim=1)

        if self.property_pred:
            assert self.dynamics.mode == "DGAP", "only DGAP mode support property prediction"
            # TODO check the pred_mask and pred_loss
            # mask the loss if the threshold is reached
            # Set a tensor with the same dimension as t_int, 1 means that t_int is less than or 
            # equal to prediction_threshold_t, and 0 means that t_int is greater than prediction_threshold_t.
            pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
            """
            ************
            如果t_int小于等于prediction_threshold_t, 则pred_loss_mask为1, 否则为0
            为1时处于nucleation time(成核阶段,有x,h,context的loss), 为0时处于growth time(生长阶段,只有x的loss)
            这里的prediction_threshold_t是一个超参数,表示成核时间的阈值
            论文中的成核与生长之间的分野, 其具体实现就是在这里!!!!!!(实际上有一部分loss在上面一段代码中已经计算了)
            这里loss是预测loss, 也就是property_pred与property_label之间的loss, pred_loss_mask用于mask掉不需要计算的loss
            ************ 
            """
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
                    self.dynamics.egnn.eval() # freeze backbone, when do the property prediction.
                elif self.freeze_gradient:
                    pred_loss = 0 # do not do the property prediction when random seed is not less than 0.5
                    self.dynamics.egnn.train() # unfreeze the backbone
                else:
                    self.dynamics.egnn.train() # unfreeze the backbone
            
            # dynamic adjust the weight
            # pred_loss_weight = (error.mean() / pred_loss.mean()).item()
            pred_loss_weight = 1
            
            
            loss_dict['pred_loss'] = pred_loss * pred_loss_weight
            loss += pred_loss
            
            # loss_dict = {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
            #       'error': error.squeeze(), "pred_loss": pred_loss, "pred_rate": pred_rate}                

        if self.atom_type_pred: # True !
            pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
            pred_loss_mask = pred_loss_mask.squeeze(1)
            atom_type_loss = atom_type_loss * pred_loss_mask
            loss_dict["atom_type_loss"] = atom_type_loss
            loss += atom_type_loss

        elif self.dynamics.mode == "PAT":
            pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
            pred_rate = pred_loss_mask.sum() / pred_loss_mask.size(0)
            pred_loss_mask = pred_loss_mask.squeeze(1)   
            atom_type_loss = atom_type_loss * pred_loss_mask
            loss += atom_type_loss
            loss_dict["atom_type_loss"] = atom_type_loss
            loss_dict["pred_rate"] = pred_rate
            loss_dict["loss"] = loss.squeeze()

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

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
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
                    s_array, t_array, z_t, node_mask, edge_mask, new_context, fix_noise=False, yt=t_array2, ys=s_array2, force_t_zero=True) 
            # z_t and t keep unchanged
        
        # calcuate the mae between new_context and org_context
        mae = torch.mean(torch.abs(new_context - org_context))
        
        return new_context, mae
    
    def evaluate_property_mp20(self, x, l, a, h, org_context, node_mask=None, edge_mask=None):
        # Normalize data, take into account volume change in x.
        batch_size = x.size(0)
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        l,a = self.normalize_lengths_angles(l, a)
        
        t_int = torch.ones((batch_size, 1), device=x.device).float() # t_int all zero
        s_int = t_int - 1
        s_array = s_int / self.T
        t_array = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s_array), x)
        gamma_t = self.inflate_batch_array(self.gamma(t_array), x)
        gamma_s_l = self.inflate_batch_array(self.gamma_length(s_array), l)
        gamma_t_l = self.inflate_batch_array(self.gamma_length(t_array), l)
        gamma_s_a = self.inflate_batch_array(self.gamma_angle(s_array), a)
        gamma_t_a = self.inflate_batch_array(self.gamma_angle(t_array), a)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        alpha_t_l = self.alpha(gamma_t_l, l)
        sigma_t_l = self.sigma(gamma_t_l, l)
        alpha_t_a = self.alpha(gamma_t_a, a)
        sigma_t_a = self.sigma(gamma_t_a, a)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=x.size(1), node_mask=node_mask)
        eps_l, eps_a = self.sample_combined_length_angle_noise(
            n_samples=batch_size,length_dim=l.size(1), angle_dim=a.size(1), device=l.device)


        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps
        z_t_l = alpha_t_l * l + sigma_t_l * eps_l
        z_t_a = alpha_t_a * a + sigma_t_a * eps_a

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
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
                
            # 参考Unigem中uni_diffusion为true时sample_p_zs_given_zt的返回值，目前不用这个函数    
            _, _, new_context = self.sample_p_zs_given_zt(
                    s_array, t_array, z_t, z_t_l, z_t_a, node_mask, 
                    edge_mask, new_context, fix_noise=False, 
                    yt=t_array2, ys=s_array2, force_t_zero=True) 
            # z_t and t keep unchanged
            # 更新的是new_context

        # calcuate the mae between new_context and org_context
        mae = torch.mean(torch.abs(new_context - org_context))
        
        return new_context, mae

    def forward(self, x, h, lengths, angles, node_mask=None, edge_mask=None, context=None, 
                mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        if self.property_pred:
            assert property_label is not None, "property_label should not be None in training"
        
        # Normalize data.
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        lengths, angles = self.normalize_lengths_angles(lengths, angles)
        # print("x.shape: ", x.shape, "x[0]", x[0])
        # print("h[categorical].shape: ", h["categorical"].shape, "h[0]", h["categorical"][0])
        # print("h[integer].shape: ", h["integer"].shape, "h[0]", h["integer"][0])

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            if expand_diff:
                loss, loss_dict = self.compute_loss_exp(x, h, lengths, angles, node_mask, edge_mask, context, t0_always=False)
            else:
                loss, loss_dict = self.compute_loss(x, h, lengths, angles, node_mask, edge_mask, context, t0_always=False, 
                                                    mask_indicator=mask_indicator, property_label=property_label, bond_info=bond_info)
        else:
            loss, loss_dict = self.compute_loss(x, h, lengths, angles, node_mask, edge_mask, context, t0_always=True, 
                                                mask_indicator=mask_indicator, property_label=property_label, bond_info=bond_info)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px
        
        return neg_log_pxh, loss_dict
        

    def sample_p_zs_given_zt(self, s, t, zt, zt_l, zt_a, 
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

        sigma2_t_given_s_length, sigma_t_given_s_length, alpha_t_given_s_length = \
                self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_l)
        sigma2_t_given_s_angle, sigma_t_given_s_angle, alpha_t_given_s_angle = \
                self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_a)
        sigma_s_length = self.sigma(gamma_s, target_tensor=zt_l)
        sigma_t_length = self.sigma(gamma_t, target_tensor=zt_l)
        sigma_s_angle = self.sigma(gamma_s, target_tensor=zt_a)
        sigma_t_angle = self.sigma(gamma_t, target_tensor=zt_a)
        alpha_s_length = self.alpha(gamma_s, target_tensor=zt_l)
        alpha_t_length = self.alpha(gamma_t, target_tensor=zt_l)
        alpha_s_angle = self.alpha(gamma_s, target_tensor=zt_a)
        alpha_t_angle = self.alpha(gamma_t, target_tensor=zt_a)

        # Neural net prediction.
        if self.bond_pred:
            (eps_t, property_pred, edge_index_knn), eps_t_l, eps_t_a = self.phi(zt, zt_l, zt_a, t, 
                                                                                node_mask, edge_mask, context)
            pred = property_pred[0]
        elif self.property_pred:
            # if pesudo_context is not None:
            #     zt = zt.clone().detach().requires_grad_(True)
                # zt.requires_grad = True
            (eps_t, pred), eps_t_l, eps_t_a = self.phi(zt, zt_l, zt_a, t, node_mask, edge_mask, context)
        else:
            eps_t, eps_t_l, eps_t_a = self.phi(zt, zt_l, zt_a, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)

        if pesudo_context is not None and (t*1000)[0].item() < 100: # growth stage
            with torch.enable_grad():
                loss_fn = torch.nn.L1Loss()
                zt = zt.clone().detach().requires_grad_(True)
                zt_l = zt_l.clone().detach().requires_grad_(True)
                zt_a = zt_a.clone().detach().requires_grad_(True)
                if (t*1000)[0].item() < 10: # growth stage?
                    its = 20
                    # opt = torch.optim.Adam([zt], lr=0.001)
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

                    gamma_s_length = self.inflate_batch_array(gamma_s, zt_l)
                    gamma_t_length = self.inflate_batch_array(gamma_t, zt_l)
                    alpha_t_length = self.alpha(gamma_t_length, zt_l)
                    sigma_t_length = self.sigma(gamma_t_length, zt_l)
                    gamma_s_angle = self.inflate_batch_array(gamma_s, zt_a)
                    gamma_t_angle = self.inflate_batch_array(gamma_t, zt_a)
                    alpha_t_angle = self.alpha(gamma_t_angle, zt_a)
                    sigma_t_angle = self.sigma(gamma_t_angle, zt_a)
                    
                    if zt.shape[-1] != eps_t.shape[-1]:
                        eps_tmp = eps_t[:,:,:3].clone().detach()
                    else:
                        eps_tmp = eps_t.clone().detach()

                    eps_tmp_l = eps_t_l.clone().detach()    
                    eps_tmp_a = eps_t_a.clone().detach()
    
                    z0 = (zt * node_mask - sigma_t * eps_tmp) / alpha_t
                    z0 = diffusion_utils.remove_mean_with_mask(z0, node_mask)

                    z0_l = (zt_l - sigma_t_length * eps_tmp_l) / alpha_t_length
                    z0_a = (zt_a - sigma_t_angle * eps_tmp_a) / alpha_t_angle
                    
                    t0 = torch.ones_like(t) * 0.001 
                                       
                    (_, pred), _, _ = self.phi(z0, z0_l, z0_a, t0, node_mask, edge_mask, context)
                    
                    loss = loss_fn(pred, pesudo_context)

                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
        
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            atom_type_pred = eps_t[:, :, 3:]

            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t[:,:,0:3]
            mu_l = zt_l / alpha_t_given_s_length - \
                (sigma2_t_given_s_length / alpha_t_given_s_length / sigma_t_length) * eps_t_l
            mu_a = zt_a / alpha_t_given_s_angle - \
                (sigma2_t_given_s_angle / alpha_t_given_s_angle / sigma_t_angle) * eps_t_a
            
        else:
            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
            mu_l = zt_l / alpha_t_given_s_length - \
                (sigma2_t_given_s_length / alpha_t_given_s_length / sigma_t_length) * eps_t_l
            mu_a = zt_a / alpha_t_given_s_angle - \
                (sigma2_t_given_s_angle / alpha_t_given_s_angle / sigma_t_angle) * eps_t_a
 
        ## Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        sigma_l = sigma_t_given_s_length * sigma_s_length / sigma_t_length
        sigma_a = sigma_t_given_s_angle * sigma_s_angle / sigma_t_angle
       
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        zs_l = self.sample_normal_length(mu_l, sigma_l, fix_noise)
        zs_a = self.sample_normal_angle(mu_a, sigma_a, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                    node_mask),
                atom_type_pred], dim=2
            )
        else:
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                    node_mask),
                zs[:, :, self.n_dims:]], dim=2
            )
        
        return zs, zs_l, zs_a

    
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
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
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
        
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        
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
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
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
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            return z_x
        return z
    

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


    def sample_combined_position_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x


    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, 
               fix_noise=False, condition_generate_x=False, annel_l=False, pesudo_context=None):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            print("using fixed noise......")
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)

            z_l, z_a = self.sample_combined_length_angle_noise(1, self.len_dim, self.angle_dim, z.device) # [1,3]
            z_l = z_l.repeat(1, n_samples).view(n_samples, -1)
            z_a = z_a.repeat(1, n_samples).view(n_samples, -1)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            z_l, z_a = self.sample_combined_length_angle_noise(n_samples, self.len_dim, self.angle_dim, z.device)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        
        # self.T = 50 ## optimal sampling change timestep
        print('sample T',self.T)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
                            
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z[:, :, self.n_dims:] = 1 # set the atom type to 1 for PAT
                z, z_l, z_a = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], z_l, z_a, node_mask, edge_mask, 
                                              context, fix_noise=fix_noise, pesudo_context=pesudo_context)
            else:
                z, z_l, z_a = self.sample_p_zs_given_zt(s_array, t_array, z, z_l, z_a, node_mask, edge_mask, 
                                              context, fix_noise=fix_noise)
                
        # Finally sample p(x, h | z_0).
        if self.property_pred:
            if self.atom_type_pred:
                z[:,:,self.n_dims:] = 1    
            x, h, pred, length, angle = self.sample_p_xh_lengths_angles_given_z0(
                z, z_l, z_a, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        else:
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z[:,:,self.n_dims:] = 1 # set the atom type to 1 for PAT
            x, h, length, angle = self.sample_p_xh_lengths_angles_given_z0(
                z, z_l, z_a, node_mask, edge_mask, context, fix_noise=fix_noise
            )

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        if self.property_pred:
            return x, h, pred, length, angle
        return x, h, length, angle

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None, annel_l=False):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        逐步逆向采样实现了扩散模型的生成过程，同时保留了中间状态以便可视化
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            fix_h = torch.ones([n_samples, n_nodes, self.in_node_nf], device=z.device)
            z = torch.cat([z, fix_h], dim=2)
 
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        # print("z shape: ", z.size())
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        # print("chain original size", chain.size())
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):    # 逆向去噪过程,正式开始采样,保存在chain中
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            

            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], node_mask, edge_mask, context)

            else:
                z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context)


            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            z = torch.cat([z[:,:, :self.n_dims], fix_h], dim=2)
        # Finally sample p(x, h | z_0).
        if self.property_pred:
            x, h, pred = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)
        else:
            x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

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
