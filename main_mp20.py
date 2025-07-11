import os
import copy
import time
import utils
import argparse
import wandb
import numpy as np
import random
from os.path import join
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
import pickle
from mp20.shared_args import setup_shared_args
from mp20.mp20 import MP20
from mp20.get_model import get_model
from mp20.utils import *
from mp20.train_epoch import train_epoch



def get_dataset(args):
    if args.dataset == 'mp20':
        dataset = MP20(root=args.datadir)
    return dataset


def get_dataloaders(args, dataset):
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)              # CPU 和 GPU 的种子
    torch.cuda.manual_seed(seed)         # GPU 单卡种子
    torch.cuda.manual_seed_all(seed)     # 多GPU种子（如果用多卡）
    np.random.seed(seed)                 # numpy 随机数种子
    random.seed(seed)                    # python 内置 random 的种子
    # 打乱
    total_len = len(dataset)
    indices = np.arange(total_len)
    np.random.shuffle(indices)
    # 计算训练集、验证集和测试集的索引
    ntrain = args.num_train
    nval = args.num_val
    ntest = args.num_test
    train_indices = indices[:ntrain]
    val_indices = indices[ntrain : ntrain+nval]
    test_indices = indices[ntrain+nval:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    bs = args.batch_size
    n_workers = args.num_workers
    # 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=n_workers)

    # Initialize dataloader
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader

    return dataloaders
########################################################################考虑是否修改
    # 例子：
    # dataloaders = {split: DataLoader(dataset,
    #                                  batch_size=batch_size,
    #                                  shuffle=args.shuffle if (split == 'train') else False,
    #                                  num_workers=num_workers,
    #                                  collate_fn=preprocess.collate_fn, drop_last=True)
    #                                  # 预处理最终生成分子图的函数是preprocess.collate_fn！！！
    #                          for split, dataset in datasets.items()}
    # 后续再调整collate_fn,因为edge_index被转置了
    # return dataloaders, charge_scale

def get_dataset_info(args):
    if args.dataset == 'mp20':
        # 数据集统计信息
        dataset_info = {
            'name': 'mp20',
            'atom_encoder': {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 
'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 
'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 
'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 'Kr': 35, 'Rb': 36, 
'Sr': 37, 'Y': 38, 'Zr': 39, 'Nb': 40, 'Mo': 41, 'Tc': 42, 'Ru': 43, 'Rh': 44, 'Pd': 45, 'Ag': 46, 
'Cd': 47, 'In': 48, 'Sn': 49, 'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 'Ba': 55, 'La': 56, 
'Ce': 57, 'Pr': 58, 'Nd': 59, 'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 'Dy': 65, 'Ho': 66, 
'Er': 67, 'Tm': 68, 'Yb': 69, 'Lu': 70, 'Hf': 71, 'Ta': 72, 'W': 73, 'Re': 74, 'Os': 75, 'Ir': 76, 
'Pt': 77, 'Au': 78, 'Hg': 79, 'Tl': 80, 'Pb': 81, 'Bi': 82, 'Pu': 88},
            'atom_decoder': ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Pu'],
            'atomic_nb':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 
72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94],
            'max_n_nodes':  20,
            'n_nodes':{17: 418, 16: 3026, 8: 3563, 18: 2418, 2: 961, 5: 2130, 4: 6964, 20: 3949, 
12: 4396, 6: 3795, 10: 4395, 3: 924, 13: 979, 14: 2959, 9: 1527, 7: 991, 11: 590, 15: 666, 
19: 468, 1: 110}
        }

    return dataset_info


def construct_model(args, dataset_info, dataloader, uni_diffusion=False, use_basis=False, decoupling=False, finetune=False):
    # Create model
    device = args.device
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader, \
            uni_diffusion=args.uni_diffusion, use_basis=args.use_basis, decoupling=args.decoupling, \
            finetune=args.finetune)
    model = model.to(device)
    return model, nodes_dist, prop_dist


def get_optim(args, generative_model):
    # Optimizer
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)
    return optim

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
 

def test(args, loader, epoch, eval_model, partition, device, 
         dtype, nodes_dist, property_norms, uni_diffusion):
    print("test!!! We should complete this function")


def analyze_and_save(args, epoch, model_sample, nodes_dist, dataset_info, device, 
                     prop_dist, n_samples, evaluate_condition_generation):
    print("analyze_and_save!!! We should complete this function")


# def evaluate_properties(args, loader, epoch, eval_model, partition, device, 
#                         dtype, nodes_dist, property_norms, wandb):
#     print("evaluate_properties!!! We should complete this function")


def main(args):
    # 1 Load dataset
    dataset = get_dataset(args)
    dataset_info = get_dataset_info(args)
    dataloaders = get_dataloaders(args, dataset)
    data_dummy = next(iter(dataloaders['train']))
    # for batch in dataloaders['train']:
    #     print(batch.edge_index)
    #     break 
    # # 上面这个循环负责确认随机种子没问题
    print("Dataset loaded successfully.")


    # 2 Set conditioning and context in dataset
    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        print("property_norms", property_norms)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
    elif args.property_pred:
        context_node_nf = 0
        property_norms = compute_mean_mad(dataloaders, [args.target_property], args.dataset)    
    else:
        context_node_nf = 0
        property_norms = None
    args.context_node_nf = context_node_nf
    args.property_norms = property_norms


    # 3 Load model and create optimizer
    # 如果没有设置预训练模型，则直接使用当前模型的参数进行训练
    model, nodes_dist, prop_dist = construct_model(args, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)

    optim = get_optim(args, model)

    if args.resume is not None:
        assert args.start_epoch > 0
        # flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        flow_state_dict = torch.load(join(args.resume, f'generative_model_ema_{args.start_epoch}.npy'))
        # optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        optim_state_dict = torch.load(join(args.resume, f'optim_{args.start_epoch}.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        print("resume from: ", join(args.resume, f'generative_model_ema_{args.start_epoch}.npy'))

    if args.pretrained_model:
        print("device: ", args.device)
        state_dict = torch.load(args.pretrained_model, map_location=args.device)
            
        current_model_dict = model.state_dict()
        new_state_dict = {}
            
        for k,v in state_dict.items():
            if k in current_model_dict:
                if v.size() == current_model_dict[k].size():
                    new_state_dict[k] = v
                else:
                    print('warning size not match: ', k, v.size(), current_model_dict[k].size())
            else:
                print(f"unexpected key {k} not in current model")
            
        miss_key, unexcept_key  =  model.load_state_dict(new_state_dict, strict=False)
        print(f"load from {args.pretrained_model}, miss_key: {miss_key}")

    print("Model constructed successfully.")


    # 4 Initialize parallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()  
    else:
        model_dp = model


    # 5 Initialize EMA if enabled
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)
        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp


    # 6 Start training
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()

        train_epoch(args=args, dataloader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, property_norms=property_norms, nodes_dist=nodes_dist, 
                    dataset_info=dataset_info, gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)

        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        # 8 Eval and Save model
        if epoch % args.test_epochs == 0:   # 默认每10个epoch测试一次

            print('Evaluating model at epoch %d' % epoch)
            if args.uni_diffusion:
                # evaluate properties
                evaluate_properties(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp, 
                                    partition='Val', device=args.device, dtype=args.dtype, nodes_dist=nodes_dist, 
                                    property_norms=property_norms, wandb=wandb)
            
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            continue    # 分析与保存
            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=args.device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples, 
                                 evaluate_condition_generation=args.evaluate_condition_generation)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=args.device, dtype=args.dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, uni_diffusion=args.uni_diffusion)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=args.device, dtype=args.dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, uni_diffusion=args.uni_diffusion)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='E3Diffusion')
    parser.add_argument('--exp_name', type=str, default='debug_mp20')
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | '
                            'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                        help='diffusion')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        )
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--brute_force', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--actnorm', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--break_train_epoch', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--condition_time', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--clip_grad', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='hutch | exact')
    # EGNN args -->
    parser.add_argument('--n_layers', type=int, default=9,
                        help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nf', type=int, default=256,
                        help='number of layers')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--norm_constant', type=float, default=1,
                        help='diff/(|diff| + norm_constant)')
    parser.add_argument('--sin_embedding', type=eval, default=False,
                        help='whether using or not the sin embedding')
    # <-- EGNN args
    parser.add_argument('--ode_regularization', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='mp20',
                        help='mp20 | mp20_second_half (train only on the last 50K samples of the training dataset)')
    parser.add_argument('--datadir', type=str, default='./mp20',
                        help='mp20 directory')
    parser.add_argument('--dequantization', type=str, default='argmax_variational',
                        help='uniform | variational | argmax_variational | deterministic')
    parser.add_argument('--n_report_steps', type=int, default=1)
    parser.add_argument('--wandb_usr', default='maochenwei-ustc', type=str)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', type=eval, default=True,
                        help='save model')
    parser.add_argument('--generate_epochs', type=int, default=1,
                        help='save model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker for the dataloader')
    parser.add_argument('--test_epochs', type=int, default=10)
    parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    parser.add_argument("--conditioning", nargs='+', default=[],
                        help='arguments : formation_energy_per_atom| band_gap | e_above_hull |' )
    parser.add_argument('--resume', type=str, default=None,
                        help='')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='Amount of EMA decay, 0 means off. A reasonable value'
                            ' is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10, 1, 1],
                        help='normalize factors for [x, categorical, integer, lengths, angles]')
    parser.add_argument('--include_charges', type=eval, default=True,
                        help='include atom charge or not')
    parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                        help="Can be used to visualize multiple times per epoch")
    parser.add_argument('--normalization_factor', type=float, default=1,
                        help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='"sum" or "mean"')
    parser.add_argument('--condition_decoupling', type=eval, default=False,
                        help='decouple the conditioning from the model')
    parser.add_argument('--uni_diffusion', type=int, default=0,
                        help='whether use uni diffusion of the diffusion steps')
    parser.add_argument('--use_basis', type=int, default=0,
                        help='whether use basis of the model')
    parser.add_argument('--evaluate_condition_generation', type=int, default=0,)
    parser.add_argument('--decoupling', type=int, default=0,)
    parser.add_argument('--finetune', type=int, default=0,)
    parser.add_argument('--expand_diff', type=int, default=0,
                        help='whether expand the diffusion steps')
    parser.add_argument('--pretrained_model', type=str, default='',)
    parser.add_argument('--denoise_pretrain', type=int, default=0, 
                        help='pretrain the model only using the denoise')
    parser.add_argument("--property_pred", type=int, default=0, help='whether predict properties')
    parser.add_argument("--prediction_threshold_t", type=int, default=10, 
                        help='threshold for adding the loss of  property prediction')
    parser.add_argument("--target_property", type=str, 
                        help='arguments : formation_energy_per_atom| band_gap | e_above_hull')
    parser.add_argument("--use_prop_pred", type=int, default=1, help='whether use property prediction')
    parser.add_argument("--freeze_gradient", type=int, default=0, 
                        help='freeze gradient for the property prediction, if set true, the gradient of molecular prediction do not inflence the generation backbone.')
    parser.add_argument("--basic_prob", type=int, default=0, help='whether use basic property')
    parser.add_argument("--unnormal_time_step", type=int, default=0, help='using abnormal time step')
    parser.add_argument("--only_noisy_node", type=int, default=0, help='only noisy node')
    parser.add_argument("--half_noisy_node", type=int, default=0, help='half 0-10, half 0-1000')
    parser.add_argument("--sep_noisy_node", type=int, default=1, help='half 0-10, half 10-1000')
    parser.add_argument("--atom_type_pred", type=int, default=0, help='atom type prediction under the DGAP setting')
    parser.add_argument("--branch_layers_num", type=int, default=0, help="branch layer number of the second branch")
    parser.add_argument("--bfn_schedule", type=int, default=0, help="whether use the bfn schedule")
    parser.add_argument("--sample_steps", type=int, default=1000, help="bfn sample steps")
    parser.add_argument("--use_get", type=int, default=0, help="use GET network for bond prediction")
    parser.add_argument("--bond_pred", type=int, default=0, help="bond prediction")
    # bfn str
    parser.add_argument("--bfn_str", type=int, default=0, help="using str schedule in the bfn")
    parser.add_argument("--str_loss_type", type=str, default="denoise_loss", help="loss type for the str schedule")
    parser.add_argument("--str_sigma_x", type=float, default=0.01, help="sigma for the str schedule for coordinate")
    parser.add_argument("--str_sigma_h", type=float, default=0.01, help="sigma for the str schedule for charge")
    parser.add_argument("--str_schedule_norm", type=float, default=0, help="whether use input nprmal scale for the str schedule")
    parser.add_argument("--temp_index", type=int, default=0, help="temp index for the str schedule")

    parser = setup_shared_args(parser)
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device(args.device)
    args.dtype = torch.float32

    print(args)

    utils.create_folders(args)  # Create folders for saving models and logs

    # Wandb config
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    os.environ["WANDB_SERVICE_WAIT"] = "60" 
    wandb.init(**kwargs)
    wandb.save('*.txt')

    """******  main function  ******"""

    main(args)

    """******  train & test  ******"""

"""
debug 命令：
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2 
--test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1

python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2 
--test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 
--property_pred 1 --target_property band_gap --visualize_every_batch 100 --num_train 1000
--conditioning band_gap

run 命令：
python main_mp20.py --exp_name mp20_egnn_dynamics --n_epochs 200 --model DGAP --atom_type_pred 1 --test_epochs 10 --batch_size 64
"""
