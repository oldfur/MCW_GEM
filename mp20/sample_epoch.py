import torch
import numpy as np
import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=torch.tensor([10]), # 默认节点数
           context=None, fix_noise=False, evaluate_condition_generation=False, pesudo_context=None, sample_steps=1000):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':        
        print(f'sample with evaluate_condition_generation: [{evaluate_condition_generation}]')
        args.expand_diff = 0
        if hasattr(args, 'bfn_schedule') and args.bfn_schedule:
            # sample_steps = 50 # optimal sampling debug set timesteps
            print('T=',sample_steps,' BFN')
            if args.bfn_str:
                theta_traj, segment_ids = generative_model.sample_bfn_str(batch_size, max_n_nodes, node_mask, edge_mask, context, sample_steps=sample_steps)
            # theta_traj, segment_ids = generative_model.sample_bfn(batch_size, max_n_nodes, node_mask, edge_mask, context, sample_steps=sample_steps)
            elif args.optimal_sampling:                
                theta_traj, segment_ids = generative_model.sample_bfn_optimal_sampling(batch_size, max_n_nodes, node_mask, edge_mask, context, sample_steps=sample_steps)
            else:
                theta_traj, segment_ids = generative_model.sample_bfn(batch_size, max_n_nodes, node_mask, edge_mask, context, sample_steps=sample_steps)
            return theta_traj, segment_ids
        elif args.property_pred:
            x, h, pred = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise, condition_generate_x=evaluate_condition_generation, annel_l=args.expand_diff, pesudo_context=pesudo_context)        
        else:
            x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise, condition_generate_x=evaluate_condition_generation, annel_l=args.expand_diff)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        if isinstance(h, list):
            h, bond_lst = h[0], h[1]
            x = [x, bond_lst]
        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    # print("sample type: ", type(x))
    # print("sample x: ", x)
    if args.property_pred:
        return one_hot, charges, x, node_mask, pred
    else:
        return one_hot, charges, x, node_mask




def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    if args.property_pred:
        one_hot, charges, x, node_mask, pred = sample(args, device, generative_model, dataset_info, prop_dist, 
                                                      nodesxsample=nodesxsample, context=context, fix_noise=True, 
                                                      evaluate_condition_generation=True)
        return one_hot, charges, x, node_mask, pred
    else:
        one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, 
                                                nodesxsample=nodesxsample, context=context, fix_noise=True)
        return one_hot, charges, x, node_mask




def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, evaluate_condition_generation=False):
    """需要修改！！！！！！"""
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    if args.bfn_schedule:
        molecules = []
    else:
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        if args.bfn_schedule:
            theta_traj, segment_ids = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)
        elif args.property_pred:
            one_hot, charges, x, node_mask, pred = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)
        else:
            one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)

        # processed_list.append((pos, atom_type))
        if args.bfn_schedule:
            segment_ids = segment_ids.cpu()
            x, h = theta_traj[-1] # x: N x 3
            if h.shape[-1] == 1:
                atom_types = charge_decode(h, dataset_info) # N x 1 --> N x 5


###########################################


            else:
                atom_types = h #  atom type prediction: h is N x 5 logits
            bz = segment_ids.max().item()
            for i in range(bz):
                pos = x[segment_ids == i].detach().cpu()
                sub_atom_type = atom_types[segment_ids == i] # n x 5
                sub_atom_type = sub_atom_type.argmax(1).cpu()
                molecules.append((pos, sub_atom_type))
        else:
            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())

    if not args.bfn_schedule:
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info, bfn_schedule=args.bfn_schedule)


###################################


    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict
