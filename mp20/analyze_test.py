# 确保这是文件的第一行
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mp20.sample_epoch import sample, sample_pure_x
from mp20.crystal import lattice_matrix, cart_to_frac, frac_to_cart, array_dict_to_crystal
from mp20.utils import RankedLogger, joblib_map, prepare_context_test, compute_loss_and_nll,\
    assert_correctly_masked, remove_mean_with_mask, assert_mean_zero_with_mask, check_mask_correct,\
    compute_loss_and_nll_pure_x
from mp20.ase_tools.viewer import AseView
from mp20.batch_reshape import reshape
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
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist()
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    if args.property_pred:
        if args.frac_coords_mode:
            one_hot, charges, frac_coords, node_mask, pred, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
        else:
            one_hot, charges, x, node_mask, pred, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, dataset_info=dataset_info)
    else:
        if args.frac_coords_mode:
            one_hot, charges, frac_coords, node_mask, length, angle = sample(args, device, model_sample, prop_dist=prop_dist,
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

        if args.frac_coords_mode:
            frac_coords_valid = frac_coords[i][mask].detach().cpu().numpy()
            x_valid = frac_to_cart(frac_coords_valid, lattice)
        else: 
            x_valid = x[i][mask].detach().cpu().numpy()
            frac_coords_valid = cart_to_frac(x_valid, lattice)

        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1)  # convert one-hot to atom types
        # charges = charges[i][mask].detach().cpu().numpy()

        if i <= 3:
            # print("sampled frac_coords:", frac_coords_valid)
            print("sampled x", x_valid)
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)

        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords_valid,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!


    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")

    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    
    print({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})

    return metrics_dict


def analyze_and_save_pure_x(args, epoch, model_sample, nodes_dist, dataset_info, 
                     prop_dist, evaluate_condition_generation, lattice_pred_model):
    print(f'Analyzing crystal validity at epoch {epoch}...')
    batch_size = args.sample_batch_size
    device = args.device
    mp20_evaluator = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(args.dataset_folder_path, f"all.csv")
            )["cif"].tolist()
        )

    # sample the crystal structures
    nodesxsample = nodes_dist.sample(batch_size)
    if args.property_pred:
        one_hot, charges, x, node_mask, pred= sample_pure_x(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
    else:
        one_hot, charges, x, node_mask= sample_pure_x(args, device, model_sample, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)

    
    x = remove_mean_with_mask(x, node_mask)
    check_mask_correct([x, one_hot, charges], node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    h = {'categorical': one_hot, 'integer': charges}
    xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
    atom_mask = (h['integer'].squeeze(-1)) > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask.to(edge_mask.device)
    
    # predict the length, angle with the lattice_pred_model
    lattice_pred_model.eval()
    with torch.no_grad():
        length, angle = lattice_pred_model.lattice_pred(xh, node_mask, edge_mask)
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy() 

    for i in range(int(batch_size)):
        
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 0], angle[i, 0])
        mask = node_mask[i].squeeze(-1).bool()

        if args.frac_coords_mode:
            frac_coords = frac_coords[i][mask].detach().cpu().numpy()
            x_valid = frac_to_cart(frac_coords, lattice)
        else: 
            x_valid = x[i][mask].detach().cpu().numpy()
            frac_coords = cart_to_frac(x_valid, lattice)

        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1)  # convert one-hot to atom types
        # charges = charges[i][mask].detach().cpu().numpy()

        if i <=2:
            print("sampled lengths:", length[i])
            print("sampled angles:", angle[i])
            # print("sampled atom types:", atom_types)

        mp20_evaluator.append_pred_array(
                {
                    "atom_types": atom_types,
                    "pos": x_valid,
                    "frac_coords": frac_coords,
                    "lengths": length[i],
                    "angles": angle[i],
                    "sample_idx": f"epoch_{epoch}_sample_{i}"
                }
            )

    # Compute generation metrics
    metrics_dict = mp20_evaluator.get_metrics(
        save=args.visualize,
        save_dir=args.save_dir + f"/epoch_{epoch}",
    )   # warning!


    for k, v in metrics_dict.items():
        print(f"{k}: {v.tolist() if isinstance(v, torch.Tensor) else v}")

    wandb.log(metrics_dict)
    wandb.log({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})
    
    print({'Validity': metrics_dict["valid_rate"].sum()/batch_size, 
               'Uniqueness': metrics_dict["unique_rate"], 
               'Novelty': metrics_dict["novel_rate"]})

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
    
    

def test(args, loader, info, epoch, eval_model, property_norms, nodes_dist, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            props = data.propertys # 理化性质, a list of dict
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            frac_coords = data['frac_coords'].to(device, dtype)
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            x = remove_mean_with_mask(x, node_mask) # 后续暂时不给x加噪声
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context_test(args.conditioning, data, props, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            # print(x.shape, h['categorical'].shape, h['integer'].shape, lengths.shape, angles.shape)
            if args.frac_coords_mode:
                # print("using frac_coords to compute loss")
                nll, _, _, loss_dict = compute_loss_and_nll(args, eval_model, nodes_dist, 
                                                frac_coords, h, lengths, angles, node_mask, edge_mask, context,
                                                property_label=props[args.target_property].to(device, dtype) \
                                                    if args.target_property in props else None,)
            else:
                nll, _, _, loss_dict = compute_loss_and_nll(args, eval_model, nodes_dist, 
                                                x, h, lengths, angles, node_mask, edge_mask, context, 
                                                bond_info=bond_info, property_label=props[args.target_property].to(device, dtype) \
                                                    if args.target_property in props else None)
        
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                if args.probabilistic_model == 'diffusion_transformer' or args.probabilistic_model == 'diffusion_Lfirst':
                    if 'total_error' in loss_dict:
                        print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, " 
                              f"NLL: {nll_epoch/n_samples:.2f}", end=', ')
                        print(f"denoise x: {loss_dict['x_error'].mean().item():.3f}, " 
                              f"denoise l: {loss_dict['l_error'].mean().item():.3f}, "
                              f"denoise a: {loss_dict['a_error'].mean().item():.3f} ",
                              f"total xla denoise: {loss_dict['total_error'].mean().item():.3f}", 
                              end = '')
                        wandb.log({f"{partition}_denoise_x": loss_dict['x_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_l": loss_dict['l_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_a": loss_dict['a_error'].mean().item()}, commit=True)
                        wandb.log({f"{partition}_denoise_xla": loss_dict['total_error'].mean().item()}, commit=True)
                    if 'atom_type_loss' in loss_dict:
                        print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.3f}', end='\n')
                        wandb.log({f"{partition}_denoise_atom_type_loss": loss_dict['atom_type_loss'].mean().item()}, commit=True)
                    if args.property_pred:
                        if not isinstance(loss_dict['pred_loss'], int):
                            print(f", pred_loss: {loss_dict['pred_loss'].mean().item():.3f}", end='')
                        print(f", pred_rate: {loss_dict['pred_rate'].mean().item():.3f}")

                else: # other models
                    print(f"\r {partition} \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"NLL: {nll_epoch/n_samples:.2f}")
                    print(f"error: {loss_dict['error'].mean().item():.3f}, ", end='')
                    if 'lattice_loss' in loss_dict:
                        print(f"lattice_loss: {loss_dict['lattice_loss'].mean().item():.3f}, ", end='')
                    print(f"kl_prior: {loss_dict['kl_prior'].mean().item():.3f}, "
                        f"loss_term_0: {loss_dict['loss_term_0'].mean().item():.2f}, "
                        f"neg_log_constants: {loss_dict['neg_log_constants'].mean().item():.3f}, "
                        f"estimator_loss_terms: {loss_dict['estimator_loss_terms'].mean().item():.3f}, ",
                        f"loss: {loss_dict['loss'].mean().item():.3f}, ",
                        f"loss_t: {loss_dict['loss_t'].mean().item():.3f}, "
                        f"loss_t_larger_than_zero: {loss_dict['loss_t_larger_than_zero'].mean().item():.3f}, ",
                        f"atom_type_loss: {loss_dict['atom_type_loss'].mean().item():.3f}"
                        )

    return nll_epoch/n_samples
 

def test_pure_x(args, loader, info, epoch, eval_model, property_norms, nodes_dist, partition='Test'):
    print(f"Testing {partition} at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            props = data.propertys # 理化性质, a list of dict
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = prepare_context_test(args.conditioning, data, props, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _, loss_dict = compute_loss_and_nll_pure_x(args, eval_model, nodes_dist, x, h,
                                            node_mask, edge_mask, context, bond_info=bond_info,
                                            property_label=props[args.target_property].to(device, dtype) \
                                                if args.target_property in props else None)
        
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}", end='\n')
                print(f"error: {loss_dict['error'].mean().item():.3f}, "
                      f"kl_prior: {loss_dict['kl_prior'].mean().item():.3f}, "
                      f"loss_term_0: {loss_dict['loss_term_0'].mean().item():.2f}, "
                      f"neg_log_constants: {loss_dict['neg_log_constants'].mean().item():.3f}, "
                      f"estimator_loss_terms: {loss_dict['estimator_loss_terms'].mean().item():.3f}, ",
                      f"loss: {loss_dict['loss'].mean().item():.3f}, ",
                      f"loss_t: {loss_dict['loss_t'].mean().item():.3f}, "
                      f"loss_t_larger_than_zero: {loss_dict['loss_t_larger_than_zero'].mean().item():.3f}, ",
                      f"atom_type_loss: {loss_dict['atom_type_loss'].mean().item():.3f}"
                      )

    return nll_epoch/n_samples
