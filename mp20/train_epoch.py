from equivariant_diffusion.utils import assert_correctly_masked, remove_mean_with_mask\
, sample_center_gravity_zero_gaussian_with_mask, assert_mean_zero_with_mask
from mp20.utils import *
from mp20.sample_epoch import sample_sweep_conditional, sample
import utils
import time
import numpy as np
import torch
import wandb
from mp20.batch_reshape import reshape
from mp20.crystal import array_dict_to_crystal

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def train_epoch(args, model, model_dp, model_ema, ema, dataloader, dataset_info, property_norms, 
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
        data = reshape(data, device, dtype, include_charges=True)

        x = data['positions'].to(device, dtype)
        # frac_coords = data['frac_coords'].to(device, dtype)
        # lattices = data['lattices'].to(device, dtype)
        lengths = data['lengths'].to(device, dtype)
        angles = data['angles'].to(device, dtype)
        # print(f"lengths: {lengths.shape}, angles: {angles.shape}")
        # lengths shape: torch.Size([B, 3]), angles shape: torch.Size([B, 3])
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        # print(x.shape, node_mask.shape, edge_mask.shape, one_hot.shape, charges.shape)
        # batch_size, 简写B;  num_atoms, 简写N: 1~20都有可能 
        # x shape: torch.Size([B, N, 3]), 
        # node_mask shape: torch.Size([B, N, 1]), 
        # edge_mask shape: torch.Size([B*N*N, 1]),
        # one_hot shape: torch.Size([B, N, maxnum_atom_type]),
        # charges shape: torch.Size([B, N, 1])
        # propertys : a list of dict, has length B, each dict has keys: 
        #     'formation_energy_per_atom', 'band_gap', 'e_above_hull'


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

        x = remove_mean_with_mask(x, node_mask)
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

        # 只需坐标、晶胞长度、角度，就可以计算晶体结构的loss
        nll, reg_term, mean_abs_z, loss_dict = compute_loss_and_nll(args, model_dp, nodes_dist,
                                                            x, h, lengths, angles, node_mask, edge_mask, context,
                                                            property_label=property_label, bond_info=bond_info)
        
        if 'error' in loss_dict:
            wandb.log({"denoise_x": loss_dict['error'].mean().item()}, commit=True)
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

            optim.step()
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
                    f"RegTerm: {reg_term.item():.1f}, "
                    f"GradNorm: {grad_norm:.1f}, "
                    f"denoise x: {loss_dict['error'].mean().item():.3f} ", 
                    end='' if args.property_pred or args.model == "PAT" else '\n')
            else:  # BFN
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    f"RegTerm: {reg_term.item():.1f}, "
                    f"posloss: {loss_dict['posloss'].mean().item():.3f}, "
                    f"charge_loss: {loss_dict['charge_loss'].mean().item():.3f}, "
                    f"GradNorm: {grad_norm:.1f}", end='' if args.property_pred or args.model == "PAT" else '\n')
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

 
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            print(f"Visualizing at epoch {epoch}, batch {i}")
        # if epoch == 0: # for test
            # 采样
            start = time.time()
            if len(args.conditioning) > 0 and not args.uni_diffusion:
                # 条件化采样

                one_hot, charges, x = save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
                print(f"one_hot: {one_hot.shape}, charges: {charges.shape}, x: {x.shape}")
                """
                打印结果(100个样本 20个原子 88种原子类型):
                one_hot: torch.Size([100, 20, 88]), charges: torch.Size([100, 20, 1]), x: torch.Size([100, 20, 3])
                """

            # save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
            #                       batch_id=str(i))

            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            # if len(args.conditioning) > 0 and not args.uni_diffusion:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=wandb, mode='conditional')
            print("应该在这里采样")

        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    if args.property_pred:
        one_hot, charges, x, node_mask, pred = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)
    else:
        one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    # # Save the sampled data
    # vis.save_xyz_file(
    #     'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
    #     id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    """从生成模型中采样不同大小的晶体结构，并将结果保存为.cif文件格式"""
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        if args.bfn_schedule:
            theta_traj, segment_ids = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, dataset_info=dataset_info)
        elif args.property_pred:
            one_hot, charges, x, node_mask, pred = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        else:
            one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        if not args.bfn_schedule:
            print(f"Generated crystal: Positions {x[:-1, :, :]}")
        
        if args.bfn_schedule:
            frame_num = len(theta_traj)
            charges = []
            one_hot = []
            x = []
            for i in range(frame_num):
                x.append(theta_traj[i][0].cpu().numpy())
                h = theta_traj[i][1].cpu()
                one_hot.append(charge_decode(h, dataset_info))
        
        # vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
        #                   batch_size * counter, name='molecule', bfn_schedule=args.bfn_schedule)
        save_file_name = f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/'
        # 需要：分数坐标、晶胞长度、角度、样本索引
        # 根据欧式空间的3D坐标x，晶胞长度lengths，晶胞角度angles，可以计算出分数坐标frac_coords
        """
        crys = Crystal(
            {
                "frac_coords": np.zeros_like(x["frac_coords"]),
                "atom_types": np.zeros_like(x["atom_types"]),
                "lengths": 100 * np.ones_like(x["lengths"]),
                "angles": np.ones_like(x["angles"]) * 90,
                "sample_idx": x["sample_idx"],
            }
        )
        """
        
