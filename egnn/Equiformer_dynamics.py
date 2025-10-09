import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT
from torch_scatter import scatter
try:
    from e3nn import o3
except ImportError:
    pass


# in crystalgrw package
from crystalgrw.utils import (
    compute_neighbors,
    conditional_grad,
)
from crystalgrw.data_utils import get_pbc_distances, radius_graph_pbc, lattice_params_from_matrix
from crystalgrw.data_utils import frac_to_cart_coords, lattice_params_to_matrix_torch
from crystalgrw.registry import registry
from crystalgrw.smearing import (
    GaussianSmearing,
)
from crystalgrw.gaussian_rbf import GaussianRadialBasisLayer
from crystalgrw.edge_rot_mat import init_edge_rot_mat
from crystalgrw.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from crystalgrw.so2_ops import RadialFunction
from crystalgrw.layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from crystalgrw.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from crystalgrw.input_block import EdgeDegreeEmbedding


# constants
MAX_ATOMIC_NUM = 89  # MP20 dataset 1-89, initial 100
# Statistics of IS2RE 100K 
_AVG_NUM_NODES  = 77.81317
_AVG_DEGREE     = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        pos,
        natoms,
        lengths,
        angles,
        batch,
        cell_offsets=None,
        edge_index=None,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        # cutoff = cutoff or self.cutoff
        # max_neighbors = max_neighbors or self.max_neighbors
        # use_pbc = use_pbc or self.use_pbc
        # otf_graph = otf_graph or self.otf_graph
        
        if use_pbc:
            if otf_graph:
                # edge_index, cell_offsets, neighbors = radius_graph_pbc(
                #     pos, natoms, cell, cutoff, max_neighbors
                # )
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    pos, (lengths, angles), natoms, cutoff, max_neighbors,
                    device=pos.device
                )


            # out = get_pbc_distances(
            #    pos,
            #    edge_index,
            #    cell,
            #    cell_offsets,
            #    neighbors,
            #    return_offsets=True,
            #    return_distance_vec=True,
            # )
            out = get_pbc_distances(
                pos,
                edge_index,
                lengths,
                angles,
                cell_offsets,
                natoms,
                neighbors,
                coord_is_cart=True,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
            
        else:
            if otf_graph:
                edge_index = radius_graph(
                    pos,
                    r=cutoff,
                    batch=batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = pos[j] - pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=pos.device
            )
            neighbors = compute_neighbors(pos, natoms, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None):
        super().__init__(modules)
        self.info_str = str(info_str)
    

    def __repr__(self): 
        return self.info_str


@registry.register_model("equiformer_v2")
class EquiformerV2(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        # num_atoms,      # not used
        # bond_feat_dim,  # not used
        # num_targets,    # not used
        use_pbc=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=12.0,
        max_num_elements=MAX_ATOMIC_NUM,

        num_layers=8,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',
        
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=None, 

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal',

        regress_energy=False,
        regress_atoms=True,
        regress_forces=True,
        regress_lattices=True,

        latent_dim=0,
        time_dim=128,
        extra_dim=128,
        atom_readout="so2",
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.gaussian_cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.regress_energy = regress_energy
        self.regress_atoms = regress_atoms
        self.regress_forces = regress_forces
        self.regress_lattices = regress_lattices
        self.atom_readout = atom_readout

        # self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)

        # Embedding block
        total_emb_size_atom = self.sphere_channels_all + latent_dim + time_dim + extra_dim

        if total_emb_size_atom > self.sphere_channels_all:
            self.extra_atom_embedding = nn.Sequential(
                nn.Linear(total_emb_size_atom, total_emb_size_atom),
                nn.ReLU(),
                nn.Linear(total_emb_size_atom, self.sphere_channels_all)
            )
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.gaussian_cutoff,
                600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        if self.regress_energy:
            self.energy_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )
        if self.regress_atoms:
            if self.atom_readout == "ffn":
                self.atom_block = FeedForwardNetwork(
                    self.sphere_channels,
                    self.ffn_hidden_channels*2,
                    MAX_ATOMIC_NUM,
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act
                )
            elif self.atom_readout == "so2":
                self.atom_block = SO2EquivariantGraphAttention(
                    self.sphere_channels,
                    self.attn_hidden_channels,
                    self.num_heads,
                    self.attn_alpha_channels,
                    self.attn_value_channels,
                    MAX_ATOMIC_NUM,
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_rotation,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.max_num_elements,
                    self.edge_channels_list,
                    self.block_use_atom_edge_embedding,
                    self.use_m_share_rad,
                    self.attn_activation,
                    self.use_s2_act_attn,
                    self.use_attn_renorm,
                    self.use_gate_act,
                    self.use_sep_s2_act,
                    alpha_drop=0.0
                )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation, 
                self.mappingReduced, 
                self.SO3_grid, 
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )
        if self.regress_lattices:
            # For equivariant lattice
            # self.lattice_block = SO2EquivariantGraphAttention(
            #    self.sphere_channels,
            #    self.attn_hidden_channels,
            #    self.num_heads,
            #    self.attn_alpha_channels,
            #    self.attn_value_channels,
            #    3,
            #    self.lmax_list,
            #    self.mmax_list,
            #    self.SO3_rotation,
            #    self.mappingReduced,
            #    self.SO3_grid,
            #    self.max_num_elements,
            #    self.edge_channels_list,
            #    self.block_use_atom_edge_embedding,
            #    self.use_m_share_rad,
            #    self.attn_activation,
            #    self.use_s2_act_attn,
            #    self.use_attn_renorm,
            #    self.use_gate_act,
            #    self.use_sep_s2_act,
            #    alpha_drop=0.0
            # )

            # For invariant lattice
            self.lattice_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels*2,
                9,
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )

            # For tensor decomposition trick
            # self.lattice_block = FeedForwardNetwork(
            #     self.sphere_channels,
            #     self.ffn_hidden_channels,
            #     1,
            #     self.lmax_list,
            #     self.mmax_list,
            #     self.SO3_grid,
            #     self.ffn_activation,
            #     self.use_gate_act,
            #     self.use_grid_mlp,
            #     self.use_sep_s2_act
            # )
            # self.lattice_block2 = FeedForwardNetwork(
            #     self.sphere_channels,
            #     self.ffn_hidden_channels,
            #     3,
            #     self.lmax_list,
            #     self.mmax_list,
            #     self.SO3_grid,
            #     self.ffn_activation,
            #     self.use_gate_act,
            #     self.use_grid_mlp,
            #     self.use_sep_s2_act
            # )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)


    @conditional_grad(torch.enable_grad())
    def forward(self,  pos, atomic_numbers, natoms,
                lengths, angles, edge_index, to_jimages, nbonds,
                node_feats=None, z=None, lat_mat=None, batch=None):
        self.batch_size = len(natoms)
        self.dtype = pos.dtype

        pos = frac_to_cart_coords(pos, lengths, angles, natoms)
        if lat_mat is None:
            lat_mat = lattice_params_to_matrix_torch(lengths, angles)

        num_atoms = len(atomic_numbers) # atomic_numbers = atom_types - 1, start from zero

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(pos=pos, 
                                natoms=natoms, 
                                # cell=lat_mat,
                                lengths=lengths,
                                angles=angles,
                                batch=batch,
                                cutoff=self.max_radius,
                                max_neighbors=self.max_neighbors,
                                use_pbc=self.use_pbc,
                                otf_graph=self.otf_graph,
                                )
        
        if edge_distance_vec.numel() == 0:
            outs = {}
            if self.regress_energy:
                outs["energy"] = torch.zeros(self.batch_size, device=pos.device, dtype=pos.dtype)
            outs["atoms"] = torch.zeros(num_atoms, MAX_ATOMIC_NUM, device=pos.device, dtype=pos.dtype)
            outs["forces"] = torch.zeros(num_atoms, 3, device=pos.device, dtype=pos.dtype)
            outs["lattices"] = torch.zeros(self.batch_size, 9, device=pos.device, dtype=pos.dtype)
            return outs

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            None, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            pos.device,
            self.dtype,
        )
        
        # Embedding block
        h = self.sphere_embedding(atomic_numbers)
        # Merge z, time embedding, and atom embedding
        if node_feats:
            h = torch.cat([h, *node_feats], dim=1)
            h = self.extra_atom_embedding(h)

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                # x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
                x.embedding[:, offset_res, :] = h
            else:
                # x.embedding[:, offset_res, :] = self.sphere_embedding(
                #     atomic_numbers
                #     )[:, offset : offset + self.sphere_channels]
                x.embedding[:, offset_res, :] = h[:, offset: offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        outs = {}

        ###############################################################
        # Energy estimation
        ###############################################################
        if self.regress_energy:
            node_energy = self.energy_block(x)
            node_energy = node_energy.embedding.narrow(1, 0, 1)
            energy = torch.zeros(len(natoms), device=node_energy.device, dtype=node_energy.dtype)
            energy.index_add_(0, batch, node_energy.view(-1))
            outs["energy"] = energy / _AVG_NUM_NODES

        ###############################################################
        # Atom estimation
        ###############################################################
        if self.regress_atoms:
            if self.atom_readout == "ffn":
                A_t = self.atom_block(x)
            elif self.atom_readout == "so2":
                A_t = self.atom_block(x,
                                      atomic_numbers,
                                      edge_distance,
                                      edge_index)
            outs["atoms"] = A_t.embedding.narrow(1, 0, 1).squeeze(1)

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            F_t = self.force_block(x,
                atomic_numbers,
                edge_distance,
                edge_index)
            F_t = F_t.embedding.narrow(1, 1, 3)
            outs["forces"] = F_t.view(-1, 3)

        ###############################################################
        # Lattice force estimation
        ###############################################################
        if self.regress_lattices:
            # Equivariant lattice
            #L_t = self.lattice_block(x,
            #    atomic_numbers,
            #    edge_distance,
            #    edge_index)
            #L_t = L_t.embedding.narrow(1, 1, 3)

            # Invariant lattice
            L_t = self.lattice_block(x)
            L_t = L_t.embedding.narrow(1, 0, 1)
            L_t = scatter(L_t,
                          torch.arange(natoms.shape[0]).to(natoms.device).repeat_interleave(natoms),
                          dim=0,
                          reduce="max"
                          )

            # Tensor decomposition trick
            # L_t = self.lattice_block(x)
            # L_t = L_t.embedding.narrow(1, 0, 4)
            # L_t = scatter(L_t,
            #               torch.arange(natoms.shape[0]).to(natoms.device).repeat_interleave(natoms),
            #               dim=0,
            #               reduce="mean"
            #               )
            # trace = L_t.narrow(1, 0, 1)
            # trace = trace.view(-1, 1).repeat(1, 3).diag_embed()
            # antisym = L_t.narrow(1, 1, 3)
            # antisym = (self.lct(3, natoms.device)[None] * antisym[:, :, None]).sum(dim=1)
            # sym = self.lattice_block2(x)
            # sym = sym.embedding.narrow(1, 1, 3)
            # sym = scatter(sym,
            #               torch.arange(natoms.shape[0]).to(natoms.device).repeat_interleave(natoms),
            #               dim=0,
            #               reduce="mean"
            #               )
            # L_t = trace + antisym + sym
            outs["lattices"] = L_t.view(-1, 9)

        return outs

    @staticmethod
    def lct(n, device):
        assert n > 2
        e = torch.zeros([n] * n)
        indices = torch.tensor([range(n)])
        indices = indices.repeat(n, 1)
        perm_indices = torch.cartesian_prod(*indices)
        for perm in perm_indices:
            perm = perm.tolist()
            sign = torch.tensor(1)
            for i in range(n):
                for j in range(i + 1, n):
                    if perm[i] > perm[j]:
                        sign *= -1
            e[tuple(perm)] = sign
        return e.to(device)


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


# def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
#     """
#     From https://github.com/yang-song/score_sde_pytorch
#     """
#     half_dim = embedding_dim // 2
#     # magic number 10000 is from transformers
#     emb = math.log(max_positions) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
#     emb = timesteps.to(torch.get_default_dtype())[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = nn.functional.pad(emb, (0, 1), mode='constant')
#     assert emb.shape == (timesteps.shape[0], embedding_dim)
#     return emb

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    timesteps = timesteps.to(torch.get_default_dtype())
    half_dim = embedding_dim // 2
    # 指数衰减因子
    emb_scale = math.log(max_positions) / (half_dim - 1)
    freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=timesteps.device, dtype=torch.get_default_dtype()))
    freqs = torch.clamp(freqs, 1e-8, 1e4)  # 防止极端值
    args = timesteps[:, None] * freqs[None, :]
    args = torch.clamp(args, min=-1e4, max=1e4)  # 防止 sin/cos overflow
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')

    return emb


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.get_default_dtype(),
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
            "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """
    The same initialization used in DDPM.
    From https://github.com/yang-song/score_sde_pytorch
    """
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class BaseDynamics(nn.Module):
    def __init__(
            self,
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.,
            scale_file=None,
            condition_time=None,
            num_targets=1,
            regress_logvars=False,
            time_dim=128,
            embed_noisy_types=False,
            embed_lattices=True,
            embed_coord=False,
            condition_dim=0,
            regress_energy=False,
            regress_atoms=True,
            regress_forces=True,
            regress_lattices=True,
            is_decode=True,
    ):
        super(BaseDynamics, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.regress_logvars = regress_logvars
        self.condition_time = condition_time
        self.embed_noisy_types = embed_noisy_types
        self.regress_energy = regress_energy
        self.regress_forces = regress_forces
        self.regress_atoms = regress_atoms
        self.regress_lattices = regress_lattices
        self.embed_lattices = embed_lattices
        self.embed_coord = embed_coord
        self.is_decode = is_decode
        self.keys = {"forces": "frac_coords", "atoms": "atom_types",
                     "lattices": "lattices"}

        if is_decode:
            assert latent_dim != 0
        else:
            assert latent_dim == 0

        if condition_time == 'None':
            self.time_dim = 0
        elif condition_time == 'constant':
            self.time_dim = 1
        elif condition_time == 'embed':
            self.time_dim = time_dim
            # Condition on noise levels.
            # self.fc_time = nn.Embedding(self.timesteps, self.time_dim)
            # self.fc_time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim * 4),
            #                              nn.ReLU(),
            #                              nn.Linear(self.time_dim * 4, self.time_dim)
            #                             )
            self.fc_time = nn.Sequential(nn.LayerNorm(self.time_dim),                      # 🔹 normalize before MLP
                                        nn.Linear(self.time_dim, self.time_dim * 4),
                                        nn.SiLU(),                                        # 🔹 smoother than ReLU
                                        nn.Linear(self.time_dim * 4, self.time_dim),
                                        nn.LayerNorm(self.time_dim)                       # 🔹 optional output norm
                                        )
            # for i in [0, 2]:
            #     self.fc_time[i].weight.data = default_init()(self.fc_time[i].weight.data.shape)
            #     nn.init.zeros_(self.fc_time[i].bias)            
            for m in self.fc_time:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0.0)


        if self.embed_noisy_types:
            noisy_atom_dim = hidden_dim
            self.noisy_atom_emb = nn.Sequential(nn.Linear(MAX_ATOMIC_NUM - 1, noisy_atom_dim * 4),
                                                nn.ReLU(),
                                                nn.Linear(noisy_atom_dim * 4, noisy_atom_dim)
                                                )
            for i in [0, 2]:
                nn.init.xavier_uniform_(self.noisy_atom_emb[i].weight.data)
                nn.init.zeros_(self.noisy_atom_emb[i].bias)
        else:
            noisy_atom_dim = 0

        if self.embed_lattices:
            lattice_dim = hidden_dim
            self.lattice_emb = nn.Sequential(nn.Linear(9, lattice_dim),
                                             nn.ReLU(),
                                             nn.Linear(lattice_dim, lattice_dim))
        else:
            lattice_dim = 0

        if self.embed_coord:
            coord_dim = hidden_dim
            self.coord_emb = nn.Sequential(nn.Linear(3, coord_dim),
                                           nn.ReLU(),
                                           nn.Linear(coord_dim, coord_dim))
        else:
            coord_dim = 0

        self.extra_dim = noisy_atom_dim + lattice_dim + coord_dim + condition_dim

        self.gnn = nn.Module() # Need to be EquiformerV2
        self.gnn.forward = lambda *args, **kwargs: None

        # if regress_atoms:
        #     atom_hidden_dim = hidden_dim + latent_dim + self.time_dim + self.extra_dim
        #     self.fc_atom = nn.Linear(atom_hidden_dim, MAX_ATOMIC_NUM)

    def bundle_feats(self, z, t, noisy_atom_types,
                     noisy_lattices, cond_feat, natoms):
        node_feats = []

        if z is not None:
            node_feats.append(z.repeat_interleave(natoms, dim=0))

        if t is not None:
            if self.condition_time == "embed":
                assert len(t.shape) == 1
                time_emb = get_timestep_embedding(t, self.time_dim)
                time_emb = self.fc_time(time_emb)
            elif self.condition_time == "constant":
                time_emb = t
            elif self.condition_time == "neglect":
                time_emb = None
            else:
                raise NotImplementedError
            time_emb = time_emb.repeat_interleave(natoms, dim=0)
            node_feats.append(time_emb)

        if self.embed_noisy_types:
            node_feats.append(self.noisy_atom_emb(noisy_atom_types))

        if self.embed_lattices:
            lattice_feats = noisy_lattices.view(-1, 9)
            lattice_feats = self.lattice_emb(lattice_feats)
            node_feats.append(lattice_feats.repeat_interleave(natoms, dim=0))

        if cond_feat is not None:
            node_feats.append(cond_feat)

        return node_feats

    def key_map(self, outs):
        for k in list(outs.keys()):
            if k in self.keys:
                outs[self.keys[k]] = outs.pop(k)
            else:
                outs.pop(k)
        return outs

    def forward(self, t, frac_coords, atom_types, natoms, lattices=None,
                noisy_atom_types=None, lengths=None, angles=None,
                z=None, cond_feat=None, batch=None):

        if batch is None:
            batch = torch.arange(
                natoms.size(0), device=natoms.device
            ).repeat_interleave(natoms, dim=0)

        node_feats = self.bundle_feats(z, t, noisy_atom_types,
                                       lattices, cond_feat, natoms)

        if lattices is not None:
            assert lattices.shape[-1] == 3
            lengths, angles = lattice_params_from_matrix(lattices)

        outs = self.gnn(
            node_feats=node_feats,
            pos=frac_coords,
            atomic_numbers=atom_types - 1,  # set an atom index to start from zero.
            natoms=natoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            nbonds=None,
            batch=batch,
        )

        outs = self.key_map(outs)

        if self.regress_atoms:
            outs["atom_types"] = torch.softmax(outs["atom_types"], dim=1)
        return outs


class EquiformerV2Dynamics(BaseDynamics):

    def __init__(
            self,
            hidden_dim=128,
            latent_dim=0,
            max_neighbors=20,
            radius=12.,
            scale_file=None,
            condition_time=None,
            num_targets=1,
            regress_logvars=False,
            time_dim=128,
            embed_noisy_types=False,
            regress_energy=False,
            regress_forces=True,
            regress_atoms=True,
            regress_lattices=True,
            embed_lattices=True,
            embed_coord=False,
            condition_dim=0,
            is_decode=True,
            atom_readout="so2",

            use_pbc=True,
            otf_graph=True,
            max_num_elements=MAX_ATOMIC_NUM,

            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,

            norm_type='rms_norm_sh',

            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=None,

            num_sphere_samples=128,

            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=600,

            attn_activation='scaled_silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='scaled_silu',
            use_gate_act=False,
            use_grid_mlp=False,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,

            weight_init='normal',
    ):
        super(EquiformerV2Dynamics, self).__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_neighbors=max_neighbors,
            radius=radius,
            scale_file=scale_file,
            condition_time=condition_time,
            num_targets=num_targets,
            regress_logvars=regress_logvars,
            time_dim=time_dim,
            embed_noisy_types=embed_noisy_types,
            regress_energy=regress_energy,
            regress_forces=regress_forces,
            regress_atoms=regress_atoms,
            regress_lattices=regress_lattices,
            embed_lattices=embed_lattices,
            embed_coord=embed_coord,
            condition_dim=condition_dim,
            is_decode=is_decode,
        )

        self.gnn = EquiformerV2(
            # num_targets=num_targets,
            # emb_size_atom=hidden_dim,
            # emb_size_edge=hidden_dim,
            regress_energy=self.regress_energy,
            regress_atoms=self.regress_atoms,
            regress_forces=self.regress_forces,
            regress_lattices=self.regress_lattices,
            latent_dim=latent_dim,
            time_dim=self.time_dim,
            extra_dim=self.extra_dim,
            atom_readout=atom_readout,

            use_pbc=use_pbc,
            otf_graph=otf_graph,
            max_neighbors=self.max_num_neighbors,
            max_radius=self.cutoff,
            max_num_elements=max_num_elements,

            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,

            norm_type=norm_type,

            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=grid_resolution,

            num_sphere_samples=num_sphere_samples,

            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,

            attn_activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,

            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,

            weight_init=weight_init,
        )

    def forward(self, t, frac_coords, atom_types, natoms, 
                lattices=None, noisy_atom_types=None, 
                lengths=None, angles=None, z=None, 
                cond_feat=None, batch=None):
        # t: [B,1]
        t = t.squeeze(-1) # [B]
        frac_coords = frac_coords % 1.0
        
        if batch is None:
            batch = torch.arange(
                natoms.size(0), device=natoms.device
            ).repeat_interleave(natoms, dim=0)

        if lattices is None and self.embed_lattices and \
            lengths is not None and angles is not None:
            lattices = lattice_params_to_matrix_torch(lengths, angles)


        node_feats = self.bundle_feats(z, t, noisy_atom_types,
                                       lattices, cond_feat, natoms)

        if lattices is not None:
            assert lattices.shape[-1] == 3
            lengths, angles = lattice_params_from_matrix(lattices)

        outs = self.gnn(
            node_feats=node_feats,
            pos=frac_coords,
            atomic_numbers=atom_types - 1,  # set an atom index to start from zero.
            natoms=natoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            nbonds=None,
            batch=batch,
        )

        outs = self.key_map(outs)

        if self.regress_atoms:
            outs["atom_types"] = torch.softmax(outs["atom_types"], dim=1)
        return outs


