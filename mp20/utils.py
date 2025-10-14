import torch
import logging
from typing import Mapping, Optional
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import wandb
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only
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
    device = data['positions'].device
    batch_size, n_nodes, _ = data['positions'].size()
    node_mask = data['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = batch_props[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1: # 事实上目前只有全局特征
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
    context = torch.cat(context_list, dim=2).to(device)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    # print("property_norms", property_norms)
    # print("context_size", context.size())
    return context

def prepare_context_test(conditioning, data, batch_props, property_norms):
    """ Prepare context for training data."""
    device = data['positions'].device
    batch_size, n_nodes, _ = data['positions'].size()
    node_mask = data['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = batch_props[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1: # 事实上目前只有全局特征
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
    context = torch.cat(context_list, dim=2).to(device)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf

    return context


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, lengths, angles,
                         node_mask, edge_mask, context, uni_diffusion=False, mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
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

    if args.probabilistic_model == 'diffusion' or args.probabilistic_model == 'diffusion_new' \
        or args.probabilistic_model == 'diffusion_another' or args.probabilistic_model == 'diffusion_concat' \
        or args.probabilistic_model == 'diffusion_transformer':
        
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        assert_correctly_masked(x, node_mask)
        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        
        inputs = (x, h, lengths, angles, node_mask, edge_mask, context)
        nll, loss_dict = generative_model(
            *inputs,
            mask_indicator=mask_indicator,
            expand_diff=args.expand_diff,
            property_label=property_label,
            bond_info=bond_info
        )


        # nll, loss_dict = generative_model(x, h, lengths, angles, node_mask, edge_mask, context, mask_indicator=mask_indicator, 
        #                                     expand_diff=args.expand_diff, property_label=property_label, bond_info=bond_info)

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

def compute_loss_and_nll_pure_x(args, generative_model, nodes_dist, x, h, 
                                node_mask, edge_mask, context, uni_diffusion=False, 
                                mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
    """
    负对数似然NLL和正则化项的计算
    """
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion' or args.probabilistic_model == 'diffusion_new' \
        or args.probabilistic_model == 'diffusion_another' or args.probabilistic_model == 'diffusion_pure_x'\
        or args.probabilistic_model == 'diffusion_transformer':
        
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        assert_correctly_masked(x, node_mask)
        
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
    

def compute_loss_and_nll_epoch(args, epoch, generative_model, nodes_dist, x, h, lengths, angles,
                         node_mask, edge_mask, context, uni_diffusion=False, mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
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
            nll, loss_dict = generative_model(x, h, lengths, angles, node_mask, edge_mask, context, mask_indicator=mask_indicator)
            # 默认的loss_dict是一个字典里面有很多个loss,此处调用了forward函数
        else:
            nll, loss_dict = generative_model(x, h, lengths, angles, node_mask, edge_mask, context, mask_indicator=mask_indicator, 
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


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def evaluate_properties(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', wandb=None): # node properties evaluation
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
            lengths = data['lengths'].to(device, dtype)
            angles = data['angles'].to(device, dtype)
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

            # Evaluate the model
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



import tqdm
from typing import Iterable, Literal
from joblib import Parallel, delayed, parallel_config


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar.

    Adapted from:
    - https://github.com/facebookresearch/flowmm
    - https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]
    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. " "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm."""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)

        
def joblib_map(
    func: callable,
    iterable: Iterable,
    n_jobs: int = 1,
    inner_max_num_threads: int | None = None,
    desc: str | None = None,
    total: int | None = None,
    backend: Literal["sequential", "loky", "threading", "multiprocessing"] = "loky",
) -> list:
    if backend != "loky" and inner_max_num_threads is not None:
        print(f"{backend=} does not support {inner_max_num_threads=}, setting to None.")
        inner_max_num_threads = None

    if backend != "sequential":
        with parallel_config(backend=backend, inner_max_num_threads=inner_max_num_threads):
            if backend == "loky":
                results = ParallelTqdm(n_jobs=n_jobs, total_tasks=total, desc=desc)(
                    delayed(func)(i) for i in iterable
                )
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in iterable)
    else:
        results = [func(i) for i in tqdm.tqdm(iterable, desc=desc, total=total)]
    return results


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)



def add_first_nan_detector(model):
    """
    注册 forward hooks，在模型 forward 中检测第一个产生 NaN/Inf 的层。
    一旦发现，立即打印详细信息并停止执行。
    """
    first_nan_found = {"flag": False}  # 用闭包保存检测状态

    def _hook_fn(module, input, output):
        # 如果已经找到第一个 NaN，就不再检测
        if first_nan_found["flag"]:
            return

        # 统一处理成列表
        outputs = output if isinstance(output, (tuple, list)) else [output]
        for o in outputs:
            if not isinstance(o, torch.Tensor):
                continue
            if torch.isnan(o).any() or torch.isinf(o).any():
                first_nan_found["flag"] = True
                print("\n🚨 Detected NaN/Inf in forward pass!")
                print(f"   ├─ Layer: {module._get_name()}")
                print(f"   ├─ Module path: {getattr(module, '_debug_name', '(unnamed)')}")
                print(f"   ├─ Output shape: {tuple(o.shape)}")
                print(f"   ├─ Output stats: min={torch.nan_to_num(o).min().item():.3e}, "
                      f"max={torch.nan_to_num(o).max().item():.3e}, mean={torch.nan_to_num(o).mean().item():.3e}")
                print("   └─ Stopping execution for debugging.\n")

                # 抛出异常，方便 traceback 定位源文件行号
                raise RuntimeError(f"NaN detected in layer: {module._debug_name}")
                break

    # 为每个子模块注册 hook
    for name, module in model.named_modules():
        module._debug_name = name
        module.register_forward_hook(_hook_fn)

    print("✅ NaN 追踪已开启：一旦某层输出出现 NaN，将立即打印该层信息并终止 forward。")
    return model


def save_nan_debug_info(module, input, output, layer_name=None):
    """
    Enhanced NaN/Inf debug hook.
    Detects invalid values in inputs/outputs/parameters,
    recursively scans all submodules, and saves detailed info.
    """
    # 确定层名
    layer_name = layer_name or module.__class__.__name__

    # 检查输入/输出是否含 NaN/Inf
    inputs_nan = any([isinstance(x, torch.Tensor) and not torch.isfinite(x).all() for x in input])
    output_nan = isinstance(output, torch.Tensor) and not torch.isfinite(output).all()

    if inputs_nan or output_nan:
        print(f"\n🚨 Detected NaN/Inf in layer: {layer_name}")
        print("--------------------------------------------------")

        # 记录哪些输入/输出坏了
        if inputs_nan:
            print("⚠️  Some inputs contain NaN/Inf!")
        if output_nan:
            print("⚠️  Output contains NaN/Inf!")

        # ✅ 递归遍历所有子层参数
        nan_params = []
        for name, param in module.named_parameters(recurse=True):
            if param is not None and not torch.isfinite(param).all():
                nan_count = (~torch.isfinite(param)).sum().item()
                total = param.numel()
                nan_params.append((name, nan_count, total))
                print(f"❌ Parameter [{name}] contains NaN/Inf "
                      f"({nan_count}/{total}, shape={tuple(param.shape)})")

        if not nan_params:
            print("✅ All parameters are finite (no NaN/Inf detected).")

        # 保存调试信息
        inputs_cpu = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in input]
        output_cpu = output.detach().cpu() if isinstance(output, torch.Tensor) else output

        params_cpu = {
            name: p.detach().cpu()
            for name, p in module.named_parameters(recurse=True)
            if p is not None
        }

        save_path = f"./nan_debug_{layer_name}.pt"
        try:
            torch.save({
                "layer_name": layer_name,
                "parameters": params_cpu,
                "input": inputs_cpu,
                "output": output_cpu,
                "nan_params": nan_params,
            }, save_path)
            print(f"📝 Saved debug info to: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"❌ Failed to save debug info: {e}")

        # 停止训练，强制中断
        raise RuntimeError(f"NaN/Inf detected in layer: {layer_name}")

