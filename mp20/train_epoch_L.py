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
from mp20.crystal import array_dict_to_crystal, lattice_matrix, cart_to_frac

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)



def train_epoch_L(args, model_dp, model_ema, ema, dataloader, dataset_info, property_norms, 
                nodes_dist, gradnorm_queue, optim, epoch, prop_dist):
    model_dp.train()
    nll_epoch = []
    n_iterations = len(dataloader)
    device = args.device
    dtype = args.dtype

    for i, data in enumerate(dataloader):
        data = reshape(data, device, dtype, include_charges=True)
        lengths = data['lengths'].to(device, dtype)
        angles = data['angles'].to(device, dtype)

        if i <= 1 and epoch <= 1:
          print("lengths for training: ", lengths[:2])
          print("angles for training: ", angles[:2])

        optim.zero_grad()

        # transform batch through flow
        # use model_dp

        nll, reg_term, mean_abs_z, loss_dict = compute_loss_and_nll_L(args, model_dp, lengths, angles)

        if args.probabilistic_model == 'diffusion_L':    
            if 'l_error' in loss_dict:
                wandb.log({"denoise_lengths": loss_dict['l_error'].mean().item()}, commit=True)
            if 'a_error' in loss_dict:
                wandb.log({"denoise_angles": loss_dict['a_error'].mean().item()}, commit=True)  
        else:
            raise ValueError(args.probabilistic_model)

        loss = nll + args.ode_regularization * reg_term 

        try:
            loss.backward()
            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model_dp, gradnorm_queue)
                if isinstance(model_dp, torch.nn.DataParallel):
                    base_model = model_dp.module
                else:
                    base_model = model_dp
                params = [p for p in base_model.parameters() if p.requires_grad] # collect params to operate on
                torch.nn.utils.clip_grad_value_(params, clip_value=1.0) # 对极端分量裁剪  
            else:
                grad_norm = 0.
        except Exception as e:
            grad_norm = 0.
            print('Error in backward pass(may occure loss zero), skipping batch')

        # 检查 loss 是否有效
        if not torch.isfinite(loss):
            print("⚠️ Detected NaN in loss, skipping batch and resetting optimizer state")
            optim.zero_grad(set_to_none=True)
            continue

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model_dp)

        # print training stats
        if i % args.n_report_steps == 0:
            if args.probabilistic_model == 'diffusion_L':
                if 'total_error' in loss_dict:
                    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                        # f"RegTerm: {reg_term.item():.1f}, "
                        f"GradNorm: {grad_norm:.1f}, "
                        f"denoise l: {loss_dict['l_error'].mean().item():.3f}, "
                        f"denoise a: {loss_dict['a_error'].mean().item():.3f} ",
                        f"total la denoise: {loss_dict['total_error'].mean().item():.3f}", 
                        end = '')
            else:
                raise ValueError(args.probabilistic_model)

        nll_epoch.append(nll.item())


        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

