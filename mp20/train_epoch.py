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
        # propertys : a list of dict, has length B, each dict has keys: 
        #     'formation_energy_per_atom', 'band_gap', 'e_above_hull'
        data = reshape(data, device, dtype, include_charges=True)
        
        x = data['positions'].to(device, dtype)
        # batch_size, 简写B;  num_atoms, 简写N: 1~20都有可能 
        # x shape: torch.Size([B, N, 3]), 
        # frac_coords = data['frac_coords'].to(device, dtype)
        # lattices = data['lattices'].to(device, dtype)
        lengths = data['lengths'].to(device, dtype)
        angles = data['angles'].to(device, dtype)
        # if i == 0:
        #   print("lengths for training: ", lengths[:2])
        #   print("angles for training: ", angles[:2])
        """
        输出:
        lengths for training:  tensor([[3.6849, 5.5324, 5.5324],
                                    [4.9460, 4.9460, 7.4759]])
        angles for training:  tensor([[120.0000,  90.0000,  90.0000],
                                    [ 90.0000,  90.0000, 120.0000]])
        """
        # lengths shape: torch.Size([B, 3]), angles shape: torch.Size([B, 3])
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
        # print(x.shape, h['categorical'].shape, h['integer'].shape, lengths.shape, angles.shape)
        nll, reg_term, mean_abs_z, loss_dict = compute_loss_and_nll(args, model_dp, nodes_dist,
                                                            x, h, lengths, angles, node_mask, edge_mask, context,
                                                            property_label=property_label, bond_info=bond_info)
        
        if 'error' in loss_dict:
            wandb.log({"denoise_x_l_a": loss_dict['error'].mean().item()}, commit=True)
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
                    # f"RegTerm: {reg_term.item():.1f}, "
                    f"GradNorm: {grad_norm:.1f}, "
                    f"denoise x,l,a: {loss_dict['error'].mean().item():.3f} ", 
                    end='' if args.property_pred or args.model == "PAT" else '\n')
            else:  # BFN
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                    # f"RegTerm: {reg_term.item():.1f}, "
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

 
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and (not epoch == 0) and (not i == 0):
            print(f"Visualizing at epoch {epoch}, batch {i}")
        # if epoch == 0: # for test
            # 采样
            start = time.time()
            if len(args.conditioning) > 0 and not args.uni_diffusion:
                # 条件化采样, 暂时no saving

                one_hot, charges, x, length, angle = save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
                # print(f"one_hot: {one_hot.shape}, charges: {charges.shape}, x: {x.shape}, length: {length.shape}, angle: {angle.shape}")
                """
                打印结果(10个样本 20个原子 88种原子类型):
                one_hot: torch.Size([10, 20, 88]), charges: torch.Size([10, 20, 1]), x: torch.Size([10, 20, 3]), 
                length: torch.Size([10, 3]), angle: torch.Size([10, 3])
                """

            # save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
            #                       batch_id=str(i))
            # chain用来可视化，目前暂时不用
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch, batch_size=args.batch_size, batch_id=str(i))
            print(f'Sampling took {time.time() - start:.2f} seconds')

            # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            # if len(args.conditioning) > 0 and not args.uni_diffusion:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=wandb, mode='conditional')

        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    if args.property_pred:
        one_hot, charges, x, node_mask, pred, length, angle = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)
    else:
        one_hot, charges, x, node_mask, length, angle = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)
    ## Save the sampled data
    # vis.save_xyz_file(
    #     'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
    #     id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x, length, angle


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=10, batch_id=''):
    """从生成模型中采样不同大小的晶体结构，并将结果保存为.cif文件格式"""
    batch_size = min(batch_size, n_samples)
    print(f"sampling {batch_size} samples with different sizes and save to outputs/{args.exp_name}/epoch_{epoch}_{batch_id}...")
    nodesxsample = nodes_dist.sample(batch_size)
    # print("n_samples in sample_different_sizes_and_save: ", n_samples)
    # print("batch_size in sample_different_sizes_and_save: ", batch_size)
    # print(f"Sampled nodesxsample: {nodesxsample}")
    """
    n_samples in sample_different_sizes_and_save:  5
    batch_size in sample_different_sizes_and_save:  5
    Sampled nodesxsample: tensor([10, 20,  6, 12, 18])
    """
    if args.bfn_schedule:
        theta_traj, segment_ids, length, angle = sample(args, device, model, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample, dataset_info=dataset_info)
    elif args.property_pred:
        one_hot, charges, x, node_mask, pred, length, angle = sample(args, device, model, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample,
                                            dataset_info=dataset_info)
    else:
        one_hot, charges, x, node_mask, length, angle = sample(args, device, model, prop_dist=prop_dist,
                                            nodesxsample=nodesxsample,
                                            dataset_info=dataset_info)
    
    if args.bfn_schedule:
        frame_num = len(theta_traj)
        charges = []
        one_hot = []
        x = []
        for i in range(frame_num):
            x.append(theta_traj[i][0].cpu().numpy())
            h = theta_traj[i][1].cpu()
            one_hot.append(charge_decode(h, dataset_info))
    
    # 需要：分数坐标、晶胞长度、角度、样本索引、原子类型
    # 根据欧式空间的3D坐标x，晶胞长度lengths，晶胞角度angles，可以计算出分数坐标frac_coords
    length = length.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy()

    for i in range(batch_size):
        save_file_name = f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}_{i}'
        lattice = lattice_matrix(length[i, 0], length[i, 1], length[i, 2],
                                    angle[i, 0], angle[i, 0], angle[i, 0])
        mask = node_mask[i].squeeze(-1).bool()
        x_valid = x[i][mask].detach().cpu().numpy()
        frac_coords = cart_to_frac(x_valid, lattice)
        one_hot_valid = one_hot[i][mask].detach().cpu().numpy()
        atom_types = np.argmax(one_hot_valid, axis=-1) 

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%m-%d-%H-%M-%S")
        sample_idx = f"{formatted_time}_{epoch}_{batch_id}_{i}"
        # charges = charges[i][mask].detach().cpu().numpy()
        
        # print("generated angles: ", angle[i])
        # print("generated lengths: ", length[i])
        # print("generated atom_types: ", atom_types)
        # print("generated frac_coords shape: ", frac_coords.shape)

        crys = array_dict_to_crystal(
            {
                "frac_coords": frac_coords,
                "atom_types": atom_types,
                "lengths": length[i],
                "angles": angle[i],
                "sample_idx": sample_idx,
            },
            save=True,
            save_dir_name=save_file_name
        )
        # print(f"save to {save_file_name}!!")

        
