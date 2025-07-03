import torch
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import wandb

from mp20.batch_reshape import reshape_minibatch


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'mp20':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    elif dataset_name == 'mp20_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataloader, properties):
    """Compute mean and mad for each property in the dataset."""
    property_norms = {}
    all_propertys = [sample.propertys for sample in dataloader.dataset]

    for property_key in properties:
        # values = dataloader.dataset.data[property_key]
        values = torch.tensor([sample[property_key] for sample in all_propertys], dtype=torch.float32)
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms


def get_adj_matrix(n_nodes, batch_size, device):
    edges_dic = {}
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    """
    根据输入的条件conditioning、小批量数据minibatch和属性归一化参数property_norms
    生成上下文张量context。上下文张量通常用于深度学习模型中,作为节点或全局特征的补充信息。
    """
    # with open(file="minibatch_output.txt", mode='w') as file:
    #     file.write(f"type of minibatch: {type(minibatch)}\n")
    #     file.write(str(minibatch))
    # print("data_dim", minibatch['positions'].size())

    minibatch_props = minibatch.propertys
    minibatch = reshape_minibatch(minibatch, include_charges=True)
    batch_size, n_nodes, _ = minibatch['positions'].size()
    node_mask = minibatch['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch_props[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # print(f"{key} is global feature")
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # print(f"{key} is node feature")
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    # print("property_norms", property_norms)
    # print("context_size", context.size())
    return context


def prepare_context_train(conditioning, data, batch_props, property_norms):
    """ Prepare context for training data."""
    batch_size, n_nodes, _ = data['positions'].size()
    node_mask = data['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = batch_props[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # print(f"{key} is global feature")
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # print(f"{key} is node feature")
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    # print("property_norms", property_norms)
    # print("context_size", context.size())
    return context


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, uni_diffusion=False, mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
    """
    负对数似然NLL和正则化项的计算
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
            nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context, mask_indicator=mask_indicator, expand_diff=args.expand_diff, property_label=property_label, bond_info=bond_info)

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


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def evaluate_properties(args, loader, epoch, eval_model, device, 
                        dtype, property_norms, nodes_dist, partition='Test', wandb=None): 
    """node properties evaluation"""
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)
        
        gts = []
        preds = []
        
        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}
            
            if 'property' in data:
                context = data['property']
                context = context.unsqueeze(1)
                context = context.repeat(1, x.shape[1], 1).to(device, dtype)
                org_context = context * node_mask
            else:
                org_context = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(org_context, node_mask)
            if isinstance(eval_model, torch.nn.DataParallel):
                pred_properties, batch_mae = eval_model.module.evaluate_property(x, h, org_context, node_mask, edge_mask)
            else:
                pred_properties, batch_mae = eval_model.evaluate_property(x, h, org_context, node_mask, edge_mask)
            
            preds.append(pred_properties)
            gts.append(org_context)
            
            print(f'batch mae is {batch_mae}')
            
            break # for test speed up
        
        # calculate the mean absolute error between preds and gts
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
        preds = preds[:, 0, 0]
        gts = gts[:, 0, 0]
        mae = torch.mean(torch.abs(preds - gts))
        
        if wandb is not None:
            wandb.log({'Properties Mean Absolute Error': mae.item()})
        
        print(f'Epoch {epoch}: properties Mean Absolute Error is {mae}')


def extract_attribute_safe(subset, attribute_name):
    """Extracts an attribute from each sample in a subset, handling missing attributes gracefully."""
    values = []
    for sample in subset:
        try:
            values.append(getattr(sample, attribute_name))
        except AttributeError:
            print(f"Attribute {attribute_name} not found in sample {sample}")
            continue
    return torch.tensor(values, dtype=torch.float32)


def extract_property_safe(subset, property_name):
    """Extracts the property value from each sample in a subset, handling missing attributes gracefully."""
    values = []
    for sample in subset:
        try:
            values.append(sample.propertys[property_name])
        except KeyError:
            print(f"Property {property_name} not found in sample {sample}")
            continue
    return torch.tensor(values, dtype=torch.float32)


def charge_decode(charge, dataset_info, remove_h=False):
    atomic_nb = dataset_info['atomic_nb']
    atom_type_num = len(atomic_nb[remove_h:])
    anchor = torch.tensor(
        [
            (2 * k - 1) / max(atomic_nb) - 1
            for k in atomic_nb[remove_h :]
        ],
        dtype=torch.float32,
        device=charge.device,
    )
    atom_type = (charge - anchor).abs().argmin(dim=-1)
    one_hot = torch.zeros(
        [charge.shape[0], atom_type_num], dtype=torch.float32
    )
    one_hot[torch.arange(charge.shape[0]), atom_type] = 1
    return one_hot
