import torch
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models_mp20 import EGNN_dynamics_MP20

from equivariant_diffusion.en_diffusion_mp20 import EnVariationalDiffusion
from equivariant_diffusion.en_diffusion_mp20_new import EnVariationalDiffusion_new
from equivariant_diffusion.en_diffusion_another import EnVariationalDiffusion_another 
from egnn.EGNN_MP20_another2 import EGNN_dynamics_MP20_another2
from egnn.Equiformer_dynamics import EquiformerV2Dynamics
from equivariant_diffusion.en_diffusion_trans import EquiTransVariationalDiffusion
from equivariant_diffusion.en_diffusion_pure_x import EnVariationalDiffusion_pure_x
from equivariant_diffusion.en_diffusion_concat import EnVariationalDiffusion_concat
from mp20.utils import extract_attribute_safe, extract_property_safe


def get_model(args, device, dataset_info, dataloader_train, 
              uni_diffusion=False, use_basis=False, decoupling=False, pretrain=False, finetune=False):
    histogram = dataset_info['n_nodes']
    in_node_nf = max(dataset_info['atom_encoder'].values()) + int(args.include_charges)
    num_classes = max(dataset_info['atom_encoder'].values())
    
    nodes_dist = DistributionNodes(histogram)
    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    if hasattr(args, 'condition_decoupling'):
        condition_decoupling = args.condition_decoupling
    else:
        condition_decoupling = False
    if not hasattr(args, 'property_pred'):
        args.property_pred = False
    if not hasattr(args, 'prediction_threshold_t'):
        args.prediction_threshold_t = 10
    if not hasattr(args, 'target_property'):
        args.target_property = None

    if args.probabilistic_model == 'diffusion' or args.probabilistic_model == 'diffusion_new' or \
        args.probabilistic_model == 'diffusion_pure_x':
        net_dynamics = EGNN_dynamics_MP20(
            in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            condition_decoupling=condition_decoupling, uni_diffusion=uni_diffusion, use_basis=use_basis, 
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding, decoupling=decoupling, 
            pretraining=pretrain, finetune=finetune, normalization_factor=args.normalization_factor, 
            aggregation_method=args.aggregation_method, property_pred=args.property_pred, 
            target_property=args.target_property, freeze_gradient=args.freeze_gradient, 
            prediction_threshold_t=args.prediction_threshold_t, 
            basic_prob=args.basic_prob if "basic_prob" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            branch_layers_num=args.branch_layers_num if "branch_layers_num" in args else 0,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,        
            sample_steps=args.sample_steps if 'sample_steps' in args else 1000,
            use_get=args.use_get if 'use_get' in args else False,
            bond_pred=args.bond_pred if 'bond_pred' in args else False,)
    # with open("qm9_model.txt", "w") as f:
    #     f.write(str(net_dynamics))
    # print(net_dynamics)
    elif args.probabilistic_model == 'diffusion_another':
        net_dynamics = EGNN_dynamics_MP20_another2(
            in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            condition_decoupling=condition_decoupling, uni_diffusion=uni_diffusion, use_basis=use_basis, 
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding, decoupling=decoupling, 
            pretraining=pretrain, finetune=finetune, normalization_factor=args.normalization_factor, 
            aggregation_method=args.aggregation_method, property_pred=args.property_pred, 
            target_property=args.target_property, freeze_gradient=args.freeze_gradient, 
            prediction_threshold_t=args.prediction_threshold_t, 
            basic_prob=args.basic_prob if "basic_prob" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            branch_layers_num=args.branch_layers_num if "branch_layers_num" in args else 0,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,        
            sample_steps=args.sample_steps if 'sample_steps' in args else 1000,
            use_get=args.use_get if 'use_get' in args else False,
            bond_pred=args.bond_pred if 'bond_pred' in args else False,)
    elif args.probabilistic_model == 'diffusion_concat':
        net_dynamics = EGNN_dynamics_MP20(
            in_node_nf=dynamics_in_node_nf + 6, context_node_nf=args.context_node_nf,
            n_dims=3, device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            condition_decoupling=condition_decoupling, uni_diffusion=uni_diffusion, use_basis=use_basis, 
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding, decoupling=decoupling, 
            pretraining=pretrain, finetune=finetune, normalization_factor=args.normalization_factor, 
            aggregation_method=args.aggregation_method, property_pred=args.property_pred, 
            target_property=args.target_property, freeze_gradient=args.freeze_gradient, 
            prediction_threshold_t=args.prediction_threshold_t, 
            basic_prob=args.basic_prob if "basic_prob" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            branch_layers_num=args.branch_layers_num if "branch_layers_num" in args else 0,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,        
            sample_steps=args.sample_steps if 'sample_steps' in args else 1000,
            use_get=args.use_get if 'use_get' in args else False,
            bond_pred=args.bond_pred if 'bond_pred' in args else False,)
    elif args.probabilistic_model == 'diffusion_transformer':
        net_dynamics = EquiformerV2Dynamics(hidden_dim=128, latent_dim=0, 
            max_neighbors=12, radius=12., condition_time='embed', 
            time_dim=128, embed_noisy_types=False, regress_energy=False, 
            regress_forces=True, regress_atoms=True, regress_lattices=True,
            embed_lattices=True, embed_coord=False,
            condition_dim=0, # 128
            is_decode=False,
            lmax_list=[4],mmax_list=[2],
            use_pbc=False, # only konw the noise, can not use pbc
            otf_graph=False, # on-the-fly graph
            )
    else:
        raise ValueError(args.probabilistic_model)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            )
        
        ##  在这里打印一些模型参数
        # print("optimal_sampling:", args.optimal_sampling)
        
        return vdm, nodes_dist, prop_dist

    elif args.probabilistic_model == 'diffusion_new':
        vdm = EnVariationalDiffusion_new(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            )
        
        return vdm, nodes_dist, prop_dist
        
    elif args.probabilistic_model == 'diffusion_another':
        vdm = EnVariationalDiffusion_another(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            )
        
        return vdm, nodes_dist, prop_dist
    

    elif args.probabilistic_model == 'diffusion_pure_x':
        vdm = EnVariationalDiffusion_pure_x(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            )
        return vdm, nodes_dist, prop_dist
    
    elif args.probabilistic_model == 'diffusion_concat':
        vdm = EnVariationalDiffusion_concat(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            num_classes=num_classes,
            )
        return vdm, nodes_dist, prop_dist
    
    elif args.probabilistic_model == 'diffusion_transformer':
        vdm = EquiTransVariationalDiffusion(
            n_dims=3, device=device,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            pre_training=pretrain,
            uni_diffusion=uni_diffusion,
            timesteps=args.diffusion_steps,
            property_pred=args.property_pred,
            freeze_gradient=args.freeze_gradient,
            target_property=args.target_property,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            norm_biases= args.normalize_biases,
            include_charges=args.include_charges,
            prediction_threshold_t=args.prediction_threshold_t,
            use_prop_pred=args.use_prop_pred if hasattr(args, 'use_prop_pred') else 1,
            unnormal_time_step=args.unnormal_time_step if "unnormal_time_step" in args else False,
            only_noisy_node=args.only_noisy_node if "only_noisy_node" in args else False,
            half_noisy_node=args.half_noisy_node if "half_noisy_node" in args else False,
            sep_noisy_node=args.sep_noisy_node if "sep_noisy_node" in args else False,
            atom_type_pred=args.atom_type_pred if "atom_type_pred" in args else False,
            bfn_schedule=args.bfn_schedule if "bfn_schedule" in args else False,
            bond_pred=args.bond_pred if "bond_pred" in args else False,
            atom_types=len(dataset_info['atom_decoder']),
            bfn_str=args.bfn_str if "bfn_str" in args else False,
            optimal_sampling=args.optimal_sampling if "optimal_sampling" in args else False,
            str_loss_type=args.str_loss_type if "str_loss_type" in args else "denoise_loss",
            str_sigma_x=args.str_sigma_x if "str_sigma_x" in args else 0.05,
            str_sigma_h=args.str_sigma_h if "str_sigma_h" in args else 0.05,
            str_schedule_norm=args.str_schedule_norm if "str_schedule_norm" in args else False,
            temp_index=args.temp_index if "temp_index" in args else 0,
            lambda_l=args.lambda_l, lambda_a=args.lambda_a,
            )
        
        # 假设你的模型变量名为 model
        total_params = sum(p.numel() for p in vdm.parameters())
        trainable_params = sum(p.numel() for p in vdm.parameters() if p.requires_grad)
        print(f"总参数量: {total_params}")
        print(f"可训练参数量: {trainable_params}")
        return vdm, nodes_dist, prop_dist
    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            all_num_atoms = extract_attribute_safe(dataloader.dataset, 'num_atoms')
            all_props = extract_property_safe(dataloader.dataset, prop)
            self._create_prob_dist(all_num_atoms,
                                   all_props,
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs.clone().detach())
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
