from equivariant_diffusion.utils import assert_correctly_masked, remove_mean_with_mask\
, sample_center_gravity_zero_gaussian_with_mask, assert_mean_zero_with_mask
from mp20.utils import *
from mp20.sample_epoch import sample_sweep_conditional, sample
import utils
import time
import numpy as np
import torch
import wandb
import datetime
from mp20.batch_reshape import reshape

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def train_epoch_pure_x(args, model, model_dp, model_ema, ema, dataloader, dataset_info, property_norms, 
                nodes_dist, gradnorm_queue, optim, epoch, prop_dist):
    # model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(dataloader)
    mask_indicator = False
    device = args.device
    dtype = args.dtype
    one_hot_shape = max(dataset_info['atom_encoder'].values())

    if args.denoise_pretrain:
        mask_indicator = 2
    for i, data in enumerate(dataloader):
        # if i >= 3:    # 调试用
        #     break

        batch_props = data.propertys # 理化性质, a dict of lists with property's name as key
        # propertys : a list of dict, has length B, each dict has keys: 
        #     'formation_energy_per_atom', 'band_gap', 'e_above_hull'
        data = reshape(data, device, dtype, include_charges=True)
        
        x = data['positions'].to(device, dtype)
        # batch_size, 简写B;  num_atoms, 简写N: 1~20都有可能 
        # x shape: torch.Size([B, N, 3]), 
        # lattices = data['lattices'].to(device, dtype)

        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        # node_mask shape: torch.Size([B, N, 1]), 
        edge_mask = data['edge_mask'].to(device, dtype)
        # edge_mask shape: torch.Size([B*N*N, 1]),
        one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
        # one_hot shape: torch.Size([B, N, maxnum_atom_type])
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        # charges shape: torch.Size([B, N, 1])


        if args.bond_pred:
            edge_index = data['edge_index'].to(device, dtype)
            edge_attr = data['edge_attr'].to(device, dtype)
            bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
        else:
            bond_info = None

        x = remove_mean_with_mask(x, node_mask) # erase mean value

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0 :
            context = prepare_context_train(args.conditioning, data, batch_props, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        # use model_dp
        if args.target_property is not None:    
            if args.target_property in batch_props: # 暂时只预测一个属性，后续可以扩展!!!
                property_label = batch_props[args.target_property].to(device, dtype)
            if property_norms is not None:
                property_label = (property_label - property_norms[args.target_property]['mean']) / property_norms[args.target_property]['mad']
        else:
            property_label = None

        # 计算晶体结构的loss
        nll, reg_term, mean_abs_z, loss_dict = compute_loss_and_nll_pure_x(args, model_dp, nodes_dist,
                                                            x, h, node_mask, edge_mask, context,
                                                            property_label=property_label, bond_info=bond_info)
        
        if 'error' in loss_dict:
            wandb.log({"denoise_coords": loss_dict['error'].mean().item()}, commit=True)
        if 'lattice_loss' in loss_dict:
            wandb.log({"lattice_loss": loss_dict['lattice_loss'].mean().item()}, commit=True)
        if 'pred_loss' in loss_dict:
            if isinstance(loss_dict['pred_loss'], torch.Tensor):
                wandb.log({"pred_loss": loss_dict['pred_loss'].mean().item(), "pred_rate": loss_dict['pred_rate'].mean().item()}, commit=True)
        if 'atom_type_loss' in loss_dict:
            wandb.log({"atom_type_loss": loss_dict['atom_type_loss'].mean().item()}, commit=True)
        if 'posloss' in loss_dict:
            wandb.log({"posloss": loss_dict['posloss'].mean().item()}, commit=True)
        if 'charge_loss' in loss_dict:
            wandb.log({"charge_loss": loss_dict['charge_loss'].mean().item()}, commit=True)
        if 'bond_loss' in loss_dict:
            wandb.log({"bond_loss": loss_dict['bond_loss'].mean().item()}, commit=True)
            nll += loss_dict['bond_loss'].mean()

        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        # loss.backward()

        try:
            loss.backward()

            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = 0.

        except Exception as e:
            grad_norm = 0.
            print('Error in backward pass(may occure loss zero), skipping batch')

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            if 'error' in loss_dict:
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    # f"RegTerm: {reg_term.item():.1f}, "
                    f"GradNorm: {grad_norm:.1f}, "
                    f"denoise x: {loss_dict['error'].mean().item():.3f} ", 
                    end = '')
            else:  # BFN
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    # f"RegTerm: {reg_term.item():.1f}, "
                    f"posloss: {loss_dict['posloss'].mean().item():.3f}, "
                    f"charge_loss: {loss_dict['charge_loss'].mean().item():.3f}, "
                    f"GradNorm: {grad_norm:.1f}", end='' if args.property_pred or args.model == "PAT" else '\n')
            if 'lattice_loss' in loss_dict:
                print(f", lattice_loss: {loss_dict['lattice_loss'].mean():.3f}", end='\n')
            if args.bond_pred:
                print(f", bond_loss: {loss_dict['bond_loss'].mean():.3f}", end='')
            if args.property_pred:
                if not isinstance(loss_dict['pred_loss'], int):
                    print(f", pred_loss: {loss_dict['pred_loss'].mean():.3f}", end='')
                print(f", pred_rate: {loss_dict['pred_rate'].mean():.3f}")
            if args.model == "PAT":
                print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.3f}', end='')
                print(f", pred_rate: {loss_dict['pred_rate'].mean():.3f}")

        nll_epoch.append(nll.item())
        wandb.log({"Batch NLL": nll.item()}, commit=True)

        if args.break_train_epoch:
            break

    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


