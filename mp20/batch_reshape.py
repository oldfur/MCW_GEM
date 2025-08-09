import torch
from collections import defaultdict

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    elif props.shape[-1] == 53:
        return props # basic properties
    else:
        return props[:, to_keep, ...]


def collate_fn(batch, load_charges=True):
    """
    Collation function for the MP20 dataset
    """
    new_batch = {}
    for prop in batch[0].keys():
        if prop not in ['edge_index', 'edge_attr']:
            new_batch[prop] = batch_stack([mol[prop] for mol in batch])
        if prop == 'edge_attr':
            new_batch[prop] = torch.cat([mol[prop] for mol in batch], dim=1)
        # if prop == 'edge_index':
        #     # edge_index: list of tensors, be concatenated along the first dim
        #     edge_index = []
        #     offset = 0
        #     for mol in batch:
        #         edge_index.append(mol[prop] + offset)
        #         offset += mol['num_atoms']
        #     new_batch[prop] = torch.cat(edge_index, dim=1)
                
    # batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (new_batch['charges'].sum(0) > 0)

    for key, prop in new_batch.items():
        if key not in ['edge_index', 'edge_attr','lengths','angles','lattices']:
            new_batch[key] = drop_zeros(prop, to_keep)

    atom_mask = new_batch['charges'] > 0

    new_batch['atom_mask'] = atom_mask

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask.to(edge_mask.device)

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    new_batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    if load_charges:
        new_batch['charges'] = new_batch['charges'].unsqueeze(2)
    else:
        new_batch['charges'] = torch.zeros(0)
        
    return new_batch


def reshape(data, device, dtype, include_charges=True):
    num_atoms = data.num_atoms.to(device, dtype)  # [bs], number of atoms for per sample 
    x = data.pos.to(device, dtype)  # [num_atoms_all, 3]
    frac_coords = data.frac_coords.to(device, dtype)  # [num_atoms_all, 3]
    lengths = data.lengths.to(device, dtype)  # [bs, 3]
    angles = data.angles.to(device, dtype)  # [bs, 3]
    lattices = data.lattices.to(device, dtype)  # [bs, 3, 3]
    belong_to = data.batch.to(device, dtype)  # [num_atoms_all]
    atom_types = data.atom_types.to(device, dtype)  # [num_atoms_all]
    edge_index = data.edge_index.to(device, dtype)  # [2, num_edges_all]
    atom_types_onehot = data.atom_types_onehot.to(device, dtype)  # [num_atoms_all, 100]
    batch = [] # a list of dict, the dict contains the index of each atom in the batch
    
    index_dict = defaultdict(list)
    for idx, val in enumerate(belong_to.tolist()):
        index_dict[val].append(idx)
    index_dict = dict(index_dict)
        
    for j in range(len(index_dict)):
        cur_dict = {}
        index_j = index_dict[j]
        cur_dict['positions'] = x[index_j]
        cur_dict['frac_coords'] = frac_coords[index_j]
        cur_dict['lengths'] = lengths[j]
        cur_dict['angles'] = angles[j]
        cur_dict['lattices'] = lattices[j]
        cur_dict['num_atoms'] = num_atoms[j]  
        cur_dict['charges'] = atom_types[index_j]
        cur_dict['one_hot'] = atom_types_onehot[index_j]
        batch.append(cur_dict)
    batch = collate_fn(batch, load_charges=include_charges)
    batch['edge_index'] = edge_index

    return batch


def reshape_minibatch(data, include_charges=True):
    num_atoms = data.num_atoms# [bs], number of atoms for per sample 
    x = data.pos # [num_atoms_all, 3]
    frac_coords = data.frac_coords # [num_atoms_all, 3]
    lengths = data.lengths # [bs, 3]
    angles = data.angles # [bs, 3]
    lattices = data.lattices # [bs, 3, 3]
    belong_to = data.batch # [num_atoms_all]
    num_atoms = data.num_atoms # [bs]
    atom_types = data.atom_types # [num_atoms_all]
    edge_index = data.edge_index # [2, num_edges_all]
    atom_types_onehot = data.atom_types_onehot # [num_atoms_all, 100]
    batch = [] # a list of dict, the dict contains the index of each atom in the batch
    
    index_dict = defaultdict(list)
    for idx, val in enumerate(belong_to.tolist()):
        index_dict[val].append(idx)
    index_dict = dict(index_dict)
        
    for j in range(len(index_dict)):
        cur_dict = {}
        index_j = index_dict[j]
        cur_dict['positions'] = x[index_j]
        cur_dict['frac_coords'] = frac_coords[index_j]
        cur_dict['lengths'] = lengths[j]
        cur_dict['angles'] = angles[j]
        cur_dict['lattices'] = lattices[j]
        cur_dict['num_atoms'] = num_atoms[j]  
        cur_dict['charges'] = atom_types[index_j]
        cur_dict['one_hot'] = atom_types_onehot[index_j]
        batch.append(cur_dict)
    batch = collate_fn(batch, load_charges=include_charges)
    batch['edge_index'] = edge_index

    return batch