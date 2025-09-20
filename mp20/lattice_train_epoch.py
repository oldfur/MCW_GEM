from equivariant_diffusion.utils import assert_correctly_masked, remove_mean_with_mask\
, sample_center_gravity_zero_gaussian_with_mask, assert_mean_zero_with_mask
from mp20.utils import *
import utils
import torch
import wandb
from mp20.batch_reshape import reshape

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def lattice_train_epoch(args, model, model_dp, model_ema, ema, dataloader, dataset_info, property_norms, 
                nodes_dist, gradnorm_queue, optim, epoch, prop_dist):
    # model_dp.train()
    model.train()
    device = args.device
    dtype = args.dtype
    one_hot_shape = max(dataset_info['atom_encoder'].values())
    n_iterations = len(dataloader)

    for i, data in enumerate(dataloader):
        # if i >= 3:    # 调试用
        #     break
        data = reshape(data, device, dtype, include_charges=True)
        
        x = data['positions'].to(device, dtype) # [B, N, 3] 
        # frac_coords = data['frac_coords'].to(device, dtype)
        # lattices = data['lattices'].to(device, dtype)
        lengths = data['lengths'].to(device, dtype)
        angles = data['angles'].to(device, dtype)
        # lengths shape: torch.Size([B, 3]), angles shape: torch.Size([B, 3])
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        # node_mask shape: torch.Size([B, N, 1]), 
        edge_mask = data['edge_mask'].to(device, dtype)
        # edge_mask shape: torch.Size([B*N*N, 1]),
        one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
        # one_hot shape: torch.Size([B, N, maxnum_atom_type])
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        # charges shape: torch.Size([B, N, 1])

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

        optim.zero_grad()

        loss, loss_dict = lattice_compute_loss(args, model_dp, x, h, lengths, angles, node_mask, edge_mask)
        wandb.log({"lattice_loss": loss_dict['lattice_loss'].mean().item()}, commit=True)

        try:
            loss.backward()
            optim.step()
        except Exception as e:
            print('Error in backward pass(may occure loss zero), skipping batch')

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, ", end='')
            print(f"lattice_loss: {loss_dict['lattice_loss'].mean():.3f}", end='\n')


def lattice_val(args, loader, info, epoch, eval_model):
    print(f"Validating at epoch {epoch}...")
    one_hot_shape = max(info['atom_encoder'].values())
    device = args.device
    dtype = args.dtype
    eval_model.eval()
    with torch.no_grad():

        n_iterations = len(loader)
        epoch_loss = 0

        for i, data in enumerate(loader):
            data = reshape(data, device, dtype, include_charges=True)
            x = data['positions'].to(device, dtype) 
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'][:,:,:one_hot_shape].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)
            h = {'categorical': one_hot, 'integer': charges}

            # transform batch through flow
            loss, loss_dict = lattice_compute_loss(args, eval_model, x, h, lengths, angles, node_mask, edge_mask)
            loss = loss.mean().item()
            epoch_loss += loss
            # standard nll from forward KL

            if i % args.n_report_steps == 0:
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, ", end='')
                print(f"lattice_loss: {loss_dict['lattice_loss'].mean():.3f}", end='\n')
    
    return epoch_loss / n_iterations


def lattice_compute_loss(args, model_dp, x, h, true_lengths, true_angles, node_mask, edge_mask):
    # x, h -> xh
    xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
    pred_lengths, pred_angles = model_dp.lattice_pred(xh, node_mask, edge_mask)
    loss = model_dp.compute_lattice_loss(pred_lengths, pred_angles, true_lengths, true_angles)
    loss_dict = {'lattice_loss': loss}

    return loss, loss_dict
