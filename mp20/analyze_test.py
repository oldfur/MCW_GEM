from mp20.sample_epoch import sample
from mp20.crystal import lattice_matrix, cart_to_frac, array_dict_to_crystal
from mp20.utils import RankedLogger, joblib_map
from mp20.ase_tools.viewer import AseView
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from typing import Dict
from functools import partial   # 固定某个函数的一部分参数，返回一个新的函数

import torch
import wandb
import numpy as np
import tqdm
import pandas as pd
import os

log = RankedLogger(__name__)  # 代码输出日志
ase_view = AseView(
    rotations="45x,45y,45z",
    atom_font_size=16,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)


def analyze_and_save(args, epoch, model_sample, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    batch_size = args.sample_batch_size
    device = args.device
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"/all.csv")
            )["cif"].tolist()
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    if args.property_pred:
        one_hot, charges, x, node_mask, pred, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
    else:
        one_hot, charges, x, node_mask, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy()

    for i in range(int(batch_size)):
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 0], angle[i, 0])
        mask = node_mask[i].squeeze(-1).bool()
        x_valid = x[i][mask].detach().cpu().numpy()
        frac_coords = cart_to_frac(x_valid, lattice)
        atom_types = charges[i][mask].detach().cpu().numpy()

        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords,
                    "lengths": length[i].detach().cpu().numpy(),
                    "angles": angle[i].detach().cpu().numpy(),
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )


    for k, v in metrics_dict.items():
        print(f"{k}: {v.item() if isinstance(v, torch.Tensor) else v}")

    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"].sum()/batch_size, 
               'Novelty': metrics_dict["novel_rate"].sum()/batch_size})
    
    print({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"].sum()/batch_size, 
               'Novelty': metrics_dict["novel_rate"].sum()/batch_size})

    return metrics_dict



#######################################################################################################

class CrystalGenerationEvaluator:
    """Evaluator for crystal generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        dataset_cif_list,
        stol=0.5,
        angle_tol=10,
        ltol=0.3,
        device="cpu",
        compute_novelty=False,
    ):
        self.dataset_cif_list = dataset_cif_list
        self.dataset_struct_list = None  # loader first time it is required
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_crys_list = []
        self.device = device
        self.compute_novelty = compute_novelty

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_crys_list = []

    def _arrays_to_crystals(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        self.pred_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=save_dir,
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Crystal",
            total=len(self.pred_arrays_list),
        )

    def _dataset_cif_to_struct(self):
        """Convert dataset CIFs to Structure objects for novelty evaluation."""
        if self.dataset_struct_list is None:
            self.dataset_struct_list = joblib_map(
                partial(Structure.from_str, fmt="cif"),
                self.dataset_cif_list,
                n_jobs=-4,
                inner_max_num_threads=1,
                desc="    Load dataset CIFs (one time)",
                total=len(self.dataset_cif_list),
            )

    def _get_novelty(self, struct):
        # matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        for other_struct in self.dataset_struct_list:
            if self.matcher.fit(struct, other_struct, skip_structure_reduction=True):
                return False
        return True

    # !!!!!!
    def get_metrics(self, save: bool = False, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_crystals(save, save_dir)

        # Compute validity metrics
        metrics_dict = {
            "valid_rate": torch.tensor([c.valid for c in self.pred_crys_list], device=self.device),
            "comp_valid_rate": torch.tensor(
                [c.comp_valid for c in self.pred_crys_list], device=self.device
            ),
            "struct_valid_rate": torch.tensor(
                [c.struct_valid for c in self.pred_crys_list], device=self.device
            ),
        }

        # Compute uniqueness
        valid_structs = [c.structure for c in self.pred_crys_list if c.valid]
        unique_struct_groups = self.matcher.group_structures(valid_structs)
        if len(valid_structs) > 0:
            metrics_dict["unique_rate"] = torch.tensor(
                len(unique_struct_groups) / len(valid_structs), device=self.device
            )
        else:
            metrics_dict["unique_rate"] = torch.tensor(0.0, device=self.device)

        # Compute novelty (slow to compute)
        if self.compute_novelty:
            self._dataset_cif_to_struct()
            struct_is_novel = []
            for struct in tqdm(
                [group[0] for group in unique_struct_groups],
                desc="    Novelty",
                total=len(unique_struct_groups),
            ):
                struct_is_novel.append(self._get_novelty(struct))

            metrics_dict["novel_rate"] = torch.tensor(
                sum(struct_is_novel) / len(struct_is_novel), device=self.device
            )
        else:
            metrics_dict["novel_rate"] = torch.tensor(-1.0, device=self.device)

        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log crystal structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Comp valid?",
                "Struct valid?",
                "Pred atom types",
                "Pred lengths",
                "Pred angles",
                "Pred 2D",
            ]
        )

        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.pred_crys_list[idx].atom_types)

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])

            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_crys_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                pred_atom_types,
                pred_lengths,
                pred_angles,
                pred_2d,
            )

        return pred_table
    

"""
self.val_metrics = ModuleDict(
            {
                "mp20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                )
            }


########################################################################################################


    generation_evaluator.device = metrics["loss"].device

    t_start = time.time()
    for samples_so_far in tqdm(
        range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
        desc=f"    Sampling",
    ):
        # Perform sampling and decoding to crystal structures

        # ......采样

        # Save predictions for metrics and visualisation
        start_idx = 0
        for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
            _atom_types = (
                out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
            )  # take argmax
            _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
            _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
            _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
            _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                1 / 3
            )  # unscale lengths
            _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
            generation_evaluator.append_pred_array(
                {
                    "atom_types": _atom_types.detach().cpu().numpy(),
                    "pos": _pos.detach().cpu().numpy(),
                    "frac_coords": _frac_coords.detach().cpu().numpy(),
                    "lengths": _lengths.detach().cpu().numpy(),
                    "angles": _angles.detach().cpu().numpy(),
                    "sample_idx": samples_so_far
                    + self.global_rank * len(batch["num_atoms"])
                    + idx_in_batch,
                }
            )
            start_idx = start_idx + num_atom
    t_end = time.time()

    # Compute generation metrics
    gen_metrics_dict = generation_evaluator.get_metrics(
        save=self.hparams.sampling.visualize,
        save_dir=self.hparams.sampling.save_dir + f"/{stage}_{self.global_rank}",
    )
    gen_metrics_dict["sampling_time"] = t_end - t_start

    # 输出评估结果到日志
    for k, v in gen_metrics_dict.items():
        metrics[k](v)
        self.log(
            f"{stage}/{k}",
            metrics[k],
            on_step=False,
            on_epoch=True,
            prog_bar=False if k != "valid_rate" else True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
        pred_table = generation_evaluator.get_wandb_table(
            current_epoch=self.current_epoch,
            save_dir=self.hparams.sampling.save_dir
            + f"/{stage}_{self.global_rank}",
        )
        self.logger.experiment.log(
            {f"{stage}_samples_table_device{self.global_rank}": pred_table}
        )
"""


###############################################################################################################
# 分子评估与测试实现，仅作参考！
   
# def analyze_stability_for_molecules(molecule_list, dataset_info, bfn_schedule=False, bond_lst=None):
#     one_hot = molecule_list['one_hot']
#     x = molecule_list['x']
#     node_mask = molecule_list['node_mask']

#     if isinstance(node_mask, torch.Tensor):
#         atomsxmol = torch.sum(node_mask, dim=1)
#     else:
#         atomsxmol = [torch.sum(m) for m in node_mask]

#     n_samples = len(x)
#     molecule_stable = 0
#     nr_stable_bonds = 0
#     n_atoms = 0
#     processed_list = []

#     for i in range(n_samples):
#         atom_type = one_hot[i].argmax(1).cpu().detach()
#         pos = x[i].cpu().detach()

#         atom_type = atom_type[0:int(atomsxmol[i])]
#         pos = pos[0:int(atomsxmol[i])]
#         processed_list.append((pos, atom_type))
    
#     for idx, mol in enumerate(processed_list):
#         pos, atom_type = mol
#         validity_results = check_stability(pos, atom_type, dataset_info)

#         molecule_stable += int(validity_results[0])
#         nr_stable_bonds += int(validity_results[1])
#         n_atoms += int(validity_results[2])

#     # Validity
#     fraction_mol_stable = molecule_stable / float(n_samples)
#     fraction_atm_stable = nr_stable_bonds / float(n_atoms)
#     validity_dict = {
#         'mol_stable': fraction_mol_stable,
#         'atm_stable': fraction_atm_stable,
#     }

#     metrics = BasicMolecularMetrics(dataset_info)
#     rdkit_metrics = metrics.evaluate(processed_list)
#     #print("Unique molecules:", rdkit_metrics[1])
#     return validity_dict, rdkit_metrics
    

# def check_stability(positions, atom_type, dataset_info, debug=False, bond_info=None):
#     assert len(positions.shape) == 2
#     assert positions.shape[1] == 3
#     atom_decoder = dataset_info['atom_decoder']
#     x = positions[:, 0]
#     y = positions[:, 1]
#     z = positions[:, 2]

#     nr_bonds = np.zeros(len(x), dtype='int')

#     for i in range(len(x)):
#         for j in range(i + 1, len(x)):
#             p1 = np.array([x[i], y[i], z[i]])
#             p2 = np.array([x[j], y[j], z[j]])
#             dist = np.sqrt(np.sum((p1 - p2) ** 2))
#             atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
#             pair = sorted([atom_type[i], atom_type[j]])
#             if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_second_half2' or dataset_info['name'] == 'qm9_first_half':
#                 order = bond_analyze.get_bond_order(atom1, atom2, dist, check_exists=True)
#             elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'pcq':
#                 order = bond_analyze.geom_predictor(
#                     (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
#             nr_bonds[i] += order
#             nr_bonds[j] += order
#     nr_stable_bonds = 0
#     for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
#         possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
#         if type(possible_bonds) == int:
#             is_stable = possible_bonds == nr_bonds_i
#         else:
#             is_stable = nr_bonds_i in possible_bonds
#         if not is_stable and debug:
#             print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
#         nr_stable_bonds += int(is_stable)

#     molecule_stable = nr_stable_bonds == len(x)
#     return molecule_stable, nr_stable_bonds, len(x)


# class BasicMolecularMetrics(object):
#     def __init__(self, dataset_info, dataset_smiles_list=None):
#         self.atom_decoder = dataset_info['atom_decoder']
#         self.dataset_smiles_list = dataset_smiles_list
#         self.dataset_info = dataset_info

#         # Retrieve dataset smiles only for qm9 currently.
#         if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
#             self.dataset_smiles_list = retrieve_qm9_smiles(
#                 self.dataset_info)

#     def compute_validity(self, generated):
#         """ generated: list of couples (positions, atom_types)"""
#         valid = []

#         for graph in generated:
#             mol = build_molecule(*graph, self.dataset_info)
#             smiles = mol2smiles(mol)
#             if smiles is not None:
#                 mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
#                 largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
#                 smiles = mol2smiles(largest_mol)
#                 valid.append(smiles)

#         return valid, len(valid) / len(generated)

#     def compute_uniqueness(self, valid):
#         """ valid: list of SMILES strings."""
#         return list(set(valid)), len(set(valid)) / len(valid)

#     def compute_novelty(self, unique):
#         num_novel = 0
#         novel = []
#         for smiles in unique:
#             if smiles not in self.dataset_smiles_list:
#                 novel.append(smiles)
#                 num_novel += 1
#         return novel, num_novel / len(unique)

#     # !!!!!!
#     def evaluate(self, generated):
#         """ generated: list of pairs (positions: n x 3, atom_types: n [int])
#             the positions and atom types should already be masked. """
#         valid, validity = self.compute_validity(generated)
#         print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
#         if validity > 0:
#             unique, uniqueness = self.compute_uniqueness(valid)
#             print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

#             if self.dataset_smiles_list is not None:
#                 _, novelty = self.compute_novelty(unique)
#                 print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
#             else:
#                 novelty = 0.0
#         else:
#             novelty = 0.0
#             uniqueness = 0.0
#             unique = None
#         return [validity, uniqueness, novelty], unique
    

# def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', uni_diffusion=False):
#     eval_model.eval()
#     with torch.no_grad():
#         nll_epoch = 0
#         n_samples = 0

#         n_iterations = len(loader)

#         for i, data in enumerate(loader):
#             x = data['positions'].to(device, dtype)
#             batch_size = x.size(0)
#             node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
#             edge_mask = data['edge_mask'].to(device, dtype)
#             one_hot = data['one_hot'].to(device, dtype)
#             charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
#             if args.bond_pred:
#                 edge_index = data['edge_index'].to(device, dtype)
#                 edge_attr = data['edge_attr'].to(device, dtype)
#                 bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
#             else:
#                 bond_info = None

#             if args.augment_noise > 0:
#                 # Add noise eps ~ N(0, augment_noise) around points.
#                 eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
#                                                                     x.device,
#                                                                     node_mask)
#                 x = x + eps * args.augment_noise

#             x = remove_mean_with_mask(x, node_mask)
#             check_mask_correct([x, one_hot, charges], node_mask)
#             assert_mean_zero_with_mask(x, node_mask)

#             h = {'categorical': one_hot, 'integer': charges}

#             if len(args.conditioning) > 0:
#                 context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
#                 assert_correctly_masked(context, node_mask)
#             elif 'property' in data:
#                 context = data['property']
#                 context = context.unsqueeze(1)
#                 context = context.repeat(1, x.shape[1], 1).to(device, dtype)
#                 context = context * node_mask
#             else:
#                 context = None

#             # transform batch through flow
#             if uni_diffusion:
#                 nll, _, _, _ = compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
#                                                     node_mask, edge_mask, context, uni_diffusion=uni_diffusion)
#             else:
#                 nll, _, _, _ = compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
#                                                     node_mask, edge_mask, context, uni_diffusion=uni_diffusion
#                                                     , property_label=data[args.target_property].to(device, dtype) if args.target_property in data else None, bond_info=bond_info)
#             # standard nll from forward KL

#             nll_epoch += nll.item() * batch_size
#             n_samples += batch_size
#             if i % args.n_report_steps == 0:
#                 print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
#                       f"NLL: {nll_epoch/n_samples:.2f}")

#     return nll_epoch/n_samples
 

def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, uni_diffusion=False, 
                         mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
    """
    负对数似然（NLL）和正则化项的计算
    Args:
        args: 参数对象，包含模型配置和训练参数
        generative_model: 生成模型，用于计算NLL
        nodes_dist: 节点分布，用于计算节点数的对数概率
        x: 输入数据，通常是分子图的节点特征
        h: 辅助信息，通常是分子图的边特征
        node_mask: 节点掩码，标记哪些节点是有效的
        edge_mask: 边掩码，标记哪些边是有效的
        context: 上下文信息，用于条件生成
        uni_diffusion: 是否使用单一扩散模型
        mask_indicator: 掩码指示器，用于处理不同类型的掩码
        expand_diff: 是否扩展扩散模型
        property_label: 属性标签，用于条件生成
        bond_info: 键信息，用于条件生成
    Returns:
        nll: 负对数似然
        reg_term: 正则化项
        mean_abs_z: 平均绝对值
        loss_dict: 损失字典，包含不同类型的损失
    """
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        
        
        if uni_diffusion:
            nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context, mask_indicator=mask_indicator)
            # 默认的loss_dict是一个字典里面有很多个loss,此处调用了forward函数
        else:
            nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context, mask_indicator=mask_indicator, 
                                              expand_diff=args.expand_diff, property_label=property_label, bond_info=bond_info)

        if args.bfn_schedule:
            return nll, torch.tensor([0], device=nll.device), torch.tensor([0], device=nll.device), loss_dict

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    
    return nll, reg_term, mean_abs_z, loss_dict
    
    # if uni_diffusion:
    #     return nll, reg_term, mean_abs_z, loss_dict
    
    # return nll, reg_term, mean_abs_z


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'
