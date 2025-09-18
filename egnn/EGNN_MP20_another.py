import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from egnn.torchmd_et import TorchMD_ET, AccumulatedNormalization
from egnn.output_modules import EquivariantVectorOutput, EquivariantScalar
from torch_scatter import scatter_add, scatter_mean

from egnn.utils import GaussianSmearing
from egnn.egnn_new import EquivariantBlock, coord2diff
import torch.nn.functional as F

from egnn.torchmd_etf2d import TorchMD_ETF2D
from torch_scatter import scatter

class EGNN_dynamics_MP20_another(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum', condition_decoupling=False,
                 uni_diffusion=False, use_basis=False, decoupling=False, 
                 freeze_gradient=False, # if freeze_gradient, the gradient of property prediciton don't influence the generation model
                 atom_type_pred = False, # if atom_type_pred, the model will predict the atom type, even under the DGAP mode
                 branch_layers_num = 0, # the number of layers for the second branch
                 bfn_schedule=False, # if bfn_schedule, the model will use the bfn schedule
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.n_dims = n_dims
        
        self.uni_diffusion = uni_diffusion
        self.use_basis = use_basis
        
        self.decoupling = decoupling
        
        self.hidden_nf = hidden_nf
        
        self.property_pred = kwargs.get('property_pred', False)
        self.prediction_threshold_t=kwargs.get('prediction_threshold_t', 10)
        
        self.pretraining = kwargs.get('pretraining', False)
        
        # Get the finetune option
        self.finetune = kwargs.get('finetune', False)
        
        self.use_get = kwargs.get('use_get', False)
        self.bond_pred = kwargs.get('bond_pred', False)
        
        if mode == 'egnn_dynamics':
            
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
                
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)

            if self.decoupling:
                self.egnn2 = EGNN(
                    in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                    hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                    n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                    inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)
            
            
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
        elif mode == 'torchmdnet':
            print("uni_diffusion: ", self.uni_diffusion)
            print("self.finetune: ", self.finetune)
            print("in_node_nf: ", in_node_nf)
            print("context_node_nf: ", context_node_nf)
            if self.uni_diffusion:
                context_node_nf = 53 + 1 # 53 is the property numbers, 1 is the time
                self.mask_embedding = nn.Embedding(1, context_node_nf) # for masking the property
            
            # if self.finetune: #TODO: check if this is correct
            context_node_nf = 53 + 1
            self.property_fc = nn.Linear(1, context_node_nf - 1) # fc emb + property
            
            shared_args = {'hidden_channels': 256, 'num_layers': 8, 'num_rbf': 64, 'rbf_type': 'expnorm', 
                           'trainable_rbf': False, 'activation': 'silu', 'neighbor_embedding': True, 
                           'cutoff_lower': 0.0, 'cutoff_upper': 5.0, 'max_z': in_node_nf + context_node_nf, 'max_num_neighbors': 32}
            self.gnn = TorchMD_ET(
                attn_activation="silu",
                num_heads=8,
                distance_influence="both",
                layernorm_on_vec='whitened',
                md17=False,
                seperate_noise=False,
                **shared_args,
            )
            self.node_out = nn.Sequential(
                nn.Linear(shared_args['hidden_channels'], shared_args['hidden_channels']),
                nn.SiLU(),
                nn.Linear(shared_args['hidden_channels'], in_node_nf + context_node_nf),
            )
            self.noise_out = EquivariantVectorOutput(shared_args['hidden_channels'], shared_args['activation'])

            if self.pretraining:
                # for denoise
                self.noise_out2 = EquivariantVectorOutput(shared_args['hidden_channels'], shared_args['activation'])
                # for nomalizer
                # normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
                
        elif mode == "DGAP":
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
            self.in_node_nf = in_node_nf # here the in_node_nf contains time, didn't contain the conditional dim
            self.atom_type_pred = atom_type_pred
            # if self.atom_type_pred:
            #     assert branch_layers_num > 0, "branch_layers_num should be larger than 0 when the atom_type_pred is True"
            if self.use_get:
                shared_args = {'hidden_channels': 256, 'num_layers': 9, 'num_rbf': 64, 'rbf_type': 'expnorm', 'trainable_rbf': False, 'activation': 'silu', 'neighbor_embedding': True, 'cutoff_lower': 0.0, 'cutoff_upper': 5.0, 'max_z': in_node_nf + context_node_nf, 'max_num_neighbors': 32}
                shared_args['bond_pred'] = self.bond_pred
                self.egnn = TorchMD_ETF2D(
                        attn_activation="silu",
                        num_heads=8,
                        distance_influence="both",
                        layernorm_on_vec='whitened',
                        md17=False,
                        seperate_noise=False,
                        **shared_args,
                    )
                self.noise_out = EquivariantVectorOutput(shared_args['hidden_channels'], shared_args['activation'])
                self.prop_pred = EquivariantScalar(shared_args['hidden_channels'], shared_args['activation'])
            else:
                self.egnn = EGNN(
                    in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                    hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                    n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                    inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim, branch_layers_num=branch_layers_num, condition=context_node_nf, bfn_schedule=bfn_schedule, 
                    prediction_threshold_t=kwargs.get('prediction_threshold_t', 10),
                    sample_steps=kwargs.get('sample_steps', 1000))
            
            self.freeze_gradient = freeze_gradient
            if self.freeze_gradient:
                self.property_layers = nn.ModuleList()
                p_layers = 2
                for _ in range(p_layers):
                    self.property_layers.append(EquivariantBlock(hidden_nf, edge_feat_nf=2, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=True, tanh=tanh,
                                                               coords_range=15, norm_constant=norm_constant,
                                                               sin_embedding=None,
                                                               normalization_factor=1,
                                                               aggregation_method='sum'))
            
            
            self.node_dec = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf,  hidden_nf),
            )
            
            
            if self.atom_type_pred:
                self.node_dec2 = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf,  in_node_nf + context_node_nf), 
                    # in_node_nf = 6 contains time; context_node_nf: conditional generation
                ) # for atom prediction 
            
            if self.bond_pred:
                self.node_dec3 = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf,  5), # bond type prediction
                )
            
            basic_prob = kwargs.get('basic_prob', 0)
            if basic_prob:
                pred_dim = 53
            else:
                pred_dim = 1
            
            self.graph_dec = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, pred_dim),
            )


            self.x_embed = nn.Sequential(
                nn.Linear(self.n_dims, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf),
            )

            self.h_embed = nn.Sequential(
                nn.Linear(self.in_node_nf - 1, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf),
            )

            self.lattice_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, 6)
            )
            
        elif mode == "PAT":
            '''
            PAT = DGAP - graph_dec
            input_dim = 3
            change the dim of the node_dec to h_dims
            '''
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
            self.in_node_nf = in_node_nf
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)
            #7 0 3
            #in_node_nf: 7 = 5 + 1 + 1(one-hot + charges + time)
            self.node_dec = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, in_node_nf),
            )

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time
        
        
        # TODO
        # self.egnn.requires_grad_(False) 
        
        if self.uni_diffusion:
            # init the guassian layer for the property
            # get start, end, num_gaussians, trainable from kwargs
            start = kwargs.get('start', -5)
            end = kwargs.get('end', 5)
            num_gaussians = kwargs.get('num_gaussians', 100)
            trainable = kwargs.get('trainable', True)
            property_latent = kwargs.get('property_latent', 64)
            
            self.gaussian_layer = GaussianSmearing(start, end, num_gaussians, trainable)
            
            self.property_emb = nn.Sequential(
                nn.Linear(num_gaussians, property_latent),
                nn.SiLU(),
                nn.Linear(property_latent, property_latent),
            )
            
            # define the property output layer
            
            if mode == 'torchmdnet':
               self.property_out = nn.Sequential(
                        nn.Linear(hidden_nf, hidden_nf),
                        nn.SiLU(),
                        nn.Linear(hidden_nf, context_node_nf - 1), # substract time
                    )
            else:
                self.property_out = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.SiLU(),
                    nn.Linear(hidden_nf, 1),
                )
        
        if self.finetune and mode == 'torchmdnet':
            self.property_out = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.SiLU(),
                    nn.Linear(hidden_nf, 1),
                )
            
            
    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward
    
    @torch.no_grad()
    def DGAP_prop_pred(self, t, xh, node_mask, edge_mask, atom_type_pred=False):
        '''
        use node decoder and graph decoder to predict the property
        '''
        assert self.mode == "DGAP", "DGAP mode should be used"
        assert (t == 0).all(), "t should be 0"
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        assert h_dims == 6, "h_dims should be 6"
        h = xh[:, self.n_dims:].clone()
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            
            if atom_type_pred:
                h = torch.ones_like(h) * node_mask # if atom prediction loss, h should be all in 1
            
            h = torch.cat([h, h_time], dim=1)

        # print("h_shape: ", h.shape)
        
        
        if self.use_basis:
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context_basis=context_basis)
        else:
            # print("h_shape: ", h.shape)
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, batch_size=bs, n_nodes=n_nodes)
                
        if self.decoupling:
            _, _, org_h = self.egnn2(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
                
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        #TODO check the output dim
        node_dec = self.node_dec(org_h)
        node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
        node_dec = torch.sum(node_dec, dim=1)
        pred = self.graph_dec(node_dec)
        pred = pred.squeeze(1)
        
        return pred
    
    def PAT_forward(self, t, xh, node_mask, edge_mask, context, t2=None, mask_y=None):
        '''
        input x: (bs, n_nodes, 3), no h
        '''
        bs, n_nodes, dims = xh.shape
        # print("xh_shape: ", xh.shape)
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 6).to(self.device)
            h = h * node_mask
            xh = torch.cat([x, h], dim=1)
        else:
            h = xh[:, self.n_dims:].clone()
        #assert h is all in 1
        assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should be all in 1"

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)
        # print("h_shape after time in PAT: ", h.shape)
        if context is not None:
            # We're conditioning, awesome!
            #一行完成assert not true
            assert not (self.mode == "DGAP" or self.mode == "PAT"), "unconditional generation not supported have context"
            
        assert self.mode == "PAT", "PAT mode should be used"
        h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
                
                
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        #TODO check the output dim
        node_dec = self.node_dec(org_h)
        #将node_dec作为h的预测
        # print("node_dec_dim: ", node_dec.size())
        # print("bs, n_nodes: ", bs, n_nodes)
        h_final = node_dec.view(bs*n_nodes, -1)
        
        if context is not None and self.context_node_nf > 0:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        # print(f"h_dims: {h_dims}, self.uni_diffusion: {self.uni_diffusion}")
        h_final = h_final.view(bs, n_nodes, -1)
        return torch.cat([vel, h_final], dim=2)
    

    def _bfnforward(self, t, x, h, node_mask, edge_mask, context):
        bs, n_nodes, dims = x.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        # xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        # x = xh[:, 0:self.n_dims].clone()
        x = x.view(bs*n_nodes, -1).clone() * node_mask
        
        
        if h is None:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        
        # if self.atom_type_pred:
        #     assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        h = h.view(bs*n_nodes, -1).clone() * node_mask
        # h means charge
        mu_charge_t = h
        h = self.gaussian_basis(mu_charge_t) # bfn shape: X x 16
        if self.atom_type_pred:
            if self.condition_time:
                time = h[:, -1].unsqueeze(1)
                h = torch.cat([torch.zeros_like(h)[:, :-1], time], dim=1)
            else:
                h = torch.zeros_like(h)
        
        h = h * node_mask
        
        
        
        
        h_time = t.view(bs, 1).repeat(1, n_nodes)
        h_time = h_time.view(bs * n_nodes, 1)
        h = torch.cat([h, h_time], dim=1)
        
        if context is not None and self.context_node_nf > 0:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
        
        h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, batch_size=bs, n_nodes=n_nodes) # TODO, org_h maybe used for the property prediction
        
        if self.property_pred:
            node_dec = self.node_dec(org_h)
            node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
            node_dec = node_dec * node_mask.reshape(bs, n_nodes, 1)
            node_dec = torch.sum(node_dec, dim=1)
            pred = self.graph_dec(node_dec)
            pred = pred.squeeze(1)
            return h_final, x_final, org_h, pred
        return h_final, x_final, org_h 
        
        # pass

    def gaussian_basis(self, x): # Form bfn
        """
        input x: [batch_size, ...]
        output: [batch_size, ..., 16]
        这段代码定义了一个名为 gaussian_basis 方法，用于将输入张量 x 转换为高斯基函数
        (Gaussian Basis Functions, BFN)的表示形式。高斯基函数是一种常用的特征变换方法,
        尤其在处理连续值输入时，用于将其映射到高维空间以捕获更多的特征信息
        """
        # x = torch.unsqueeze(x, dim=-1)  # [batch_size, ..., 1]
        start, end = -2, 2
        self.in_node_nf_bfn = 16
        
        self.width = (end - start) / self.in_node_nf_bfn
        
        self.centers =  torch.linspace(
            start, end, self.in_node_nf_bfn, device="cuda:0", dtype=torch.float32,
        ) 
        out = (x - self.centers) / self.width
        ret = torch.exp(-0.5 * out**2)

        return F.normalize(ret, dim=-1, p=1) * 2 - 1  # 将归一化后的值从 [0, 1] 映射到 [-1, 1]，为适配后续模型的输入范围




    def _forward(self, t, xh, node_mask, edge_mask, context, t2=None, mask_y=None):
        '''
        时间特征处理
        如果 self.condition_time 为真，将时间步长 t 添加到特征 h 中：
        如果 t 是标量，则为每个节点复制相同的时间值。
        如果 t 是张量，则根据批量大小和节点数调整形状
        
        上下文特征处理
        如果 context 不为空，表示模型需要条件上下文：
        self.uni_diffusion 模式:
            对上下文进行形状调整，并根据需要添加时间步长 t2。
            如果 mask_y 为真，使用掩码嵌入替代上下文信息。
        self.finetune 模式:
            使用全连接层对上下文进行嵌入，并与特征拼接。
        self.context_node_nf > 0 模式:
            将上下文直接拼接到特征中
        '''
        # print("xh_shape: ", xh.shape)
        # print("xh[0]", xh[0])
        if self.mode == 'PAT':
            return self.PAT_forward(t, xh, node_mask, edge_mask, context, t2, mask_y)
        

        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        # if self.mode == "DGAP":
        #     assert h_dims == 6, "h_dims should be 6"


        if h_dims == 0:
            if self.atom_type_pred:
                h = torch.ones(bs*n_nodes, self.in_node_nf - 1).to(self.device) # erase the time, self in_node_nf contains the time
                h = h * node_mask
                xh = torch.cat([x, h], dim=1)
            else:
                h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.atom_type_pred:
            assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        else:
            assert not (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            if self.uni_diffusion:
                
                context = context.view(bs*n_nodes, -1)
                
                if self.use_basis:
                    context_basis = self.gaussian_layer(context.squeeze())
                    context_basis = self.property_emb(context_basis)
                
                # concat t2:
                c_time = t2.view(bs, 1).repeat(1, n_nodes)
                c_time = c_time.view(bs * n_nodes, 1)
                
                context_with_t = torch.cat([context, c_time], dim=1)
                if mask_y is not None and mask_y: # masking the property
                    context_with_t = self.mask_embedding(torch.zeros(1).long().to(t.device)).repeat(bs*n_nodes, 1)
                
                h = torch.cat([h, context_with_t], dim=1)
            elif self.finetune: 
                context_emb = self.property_fc(context)
                context = context.view(bs*n_nodes, -1)
                context_emb = context_emb.view(bs*n_nodes, -1)
                h = torch.cat([h, context_emb, context], dim=1)
            elif self.context_node_nf > 0: # TODO eval
                context = context.view(bs*n_nodes, self.context_node_nf)
            
                h = torch.cat([h, context], dim=1)


        """目前使用DGAP"""
        if self.use_basis:
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context_basis=context_basis)
        else:
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, batch_size=bs, n_nodes=n_nodes)            
        
        if self.decoupling:
            _, _, org_h = self.egnn2(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        
        if self.freeze_gradient:
            distances, _ = coord2diff(x, edges)
            org_h_pred = org_h
            tmp_x = x_final
            for module in self.property_layers:
                org_h_pred, tmp_x = module(org_h_pred, tmp_x, edges, node_mask, edge_mask, distances)
        else:
            org_h_pred = org_h
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        
        if self.atom_type_pred:
            h_final = self.node_dec2(org_h)
            h_final = h_final.view(bs*n_nodes, -1)
        
        node_dec = self.node_dec(org_h_pred)
        node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
        node_dec = node_dec * node_mask.reshape(bs, n_nodes, 1) # add mask
        
        # property prediction
        new_node_dec = torch.sum(node_dec, dim=1)   # (bs, hidden)
        pred = self.graph_dec(new_node_dec)
        pred = pred.squeeze(1)

        # lattice prediction
        lx_final, lh_final = x_final.view(bs*n_nodes, -1), h_final.view(bs*n_nodes, -1)
        lh_final = lh_final[:, :self.in_node_nf - 1] # remove time
        lattice_input = torch.cat([self.x_embed(lx_final), 
                                   self.h_embed(lh_final)], dim=1) # (bs*n_nodes, hidden*2)
        lattice_input = lattice_input.view(bs, n_nodes, self.hidden_nf * 2) * node_mask.view(bs, n_nodes, 1)
        lattice_dec = torch.mean(lattice_input, dim=1) # (bs, 2* hidden)
        out = self.lattice_mlp(lattice_dec)  # (bs,6)
        lengths = F.softplus(out[:, :3])  # >0
        angles = torch.sigmoid(out[:, 3:]) * 180.0  # (0,180)
        
        if context is not None and self.context_node_nf > 0:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = h_final.view(bs, n_nodes, -1)

        if self.uni_diffusion:
            return torch.cat([vel, h_final], dim=2), pred, lengths, angles 
        if self.property_pred:
            return (torch.cat([vel, h_final], dim=2), pred), lengths, angles
        return torch.cat([vel, h_final], dim=2), lengths, angles 

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
                # 返回一个包含两个张量的列表 [rows, cols]，分别表示边的起点和终点索引
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
    
    def predict_lattice(self, xh, node_mask):
        '''
        use the graph_dec to predict the lattice
        '''
        assert self.mode == "DGAP", "DGAP mode should be used"
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        node_mask = node_mask.view(bs*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        assert h_dims == 6, "h_dims should be 6"
        h = xh[:, self.n_dims:].clone()
        
        if self.condition_time:
            h = h[:, :-1] # remove time
        
        h = torch.cat([h, node_mask], dim=1) # add mask info
        
        node_dec = self.node_dec(h)
        node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
        node_dec = node_dec * node_mask.reshape(bs, n_nodes, 1)
        node_dec = torch.sum(node_dec, dim=1)
        out = self.crystal_mlp(node_dec)  # (B,6)
        lengths = F.softplus(out[:, :3])  # >0
        angles = torch.sigmoid(out[:, 3:]) * 180.0  # (0,180)
        
        return lengths, angles
