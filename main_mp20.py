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
from mp20.train_epoch_pure_x import train_epoch_pure_x
from mp20.analyze_test import analyze_and_save, test, analyze_and_save_pure_x, test_pure_x
from train_lattice_egnn import construct_lattice_model
from mp20.gradient_watcher import GradientWatcher


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
    if ntest == -1: # include the whole dataset, for training
        test_indices = indices[ntrain+nval:]
    else: # for debug
        test_indices = indices[ntrain+nval:ntrain+nval+ntest]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    bs = args.batch_size
    n_workers = args.num_workers
    # 构造 DataLoader, 丢弃最后一个batch
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=n_workers, drop_last=True)  
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=n_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=n_workers, drop_last=True)

    # Initialize dataloader
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader

    return dataloaders


def get_dataset_info(args):
    if args.dataset == 'mp20':
        # 数据集统计信息
        dataset_info = {
            'name': 'mp20',
            'atom_encoder': {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 
                'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 
                'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 
                'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 
                'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 
                'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 
                'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 
                'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 
                'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Pu': 89},
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


def construct_model(args, dataset_info, dataloader):
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
    model = add_first_nan_detector(model)

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)

    optim = get_optim(args, model)
    # watcher = GradientWatcher(model, threshold=100.0, log_path="logs/grad_log.txt", verbose=False)

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
    if args.dp and torch.cuda.device_count() > 1 and str(args.device) == "cuda":
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()  
    else:   
        model_dp = model


    # 5 Initialize EMA if enabled
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        model_ema = model_ema.to(args.device)
        ema = flow_utils.EMA(args.ema_decay)
        if args.dp and torch.cuda.device_count() > 1 and str(args.device) == "cuda":
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
        if args.probabilistic_model == 'diffusion_pure_x':
            train_epoch_pure_x(args=args, dataloader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                        model_ema=model_ema, ema=ema, property_norms=property_norms, nodes_dist=nodes_dist, 
                        dataset_info=dataset_info, gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        else:
            train_epoch(args=args, dataloader=dataloaders['train'], epoch=epoch, model_dp=model_dp,
                        model_ema=model_ema, ema=ema, property_norms=property_norms, nodes_dist=nodes_dist, 
                        dataset_info=dataset_info, gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)

        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        # 8 Eval and Save model
        if epoch % args.test_epochs == 0 and epoch != 0 and epoch >= args.visulaize_epoch:   # 默认每10个epoch测试一次

            print('Evaluating model at epoch %d' % epoch)
            
            # if isinstance(model, en_diffusion.EnVariationalDiffusion):
            #     wandb.log(model.log_info(), commit=True)

            # 分析与保存

            if args.probabilistic_model == 'diffusion_pure_x':
                lattice_pred_model, _, _ = construct_lattice_model(args, dataset_info)
                filename = f"lattice_checkpoints/epoch_210_val_loss_35.3340.pth"
                print("load model for test: ", filename)
                lattice_pred_model.load_state_dict(torch.load(filename))
                analyze_and_save_pure_x(args, epoch, model_ema, nodes_dist,
                            dataset_info, prop_dist, args.evaluate_condition_generation, lattice_pred_model=lattice_pred_model)
                nll_val = test_pure_x(args, dataloaders['val'], dataset_info, epoch, model_ema_dp, 
                               property_norms, nodes_dist, partition='Val')
                nll_test = test_pure_x(args, dataloaders['test'], dataset_info, epoch, model_ema_dp, 
                                property_norms, nodes_dist, partition='Test')
            else:
                analyze_and_save(args, epoch, model_ema, nodes_dist,
                                dataset_info, prop_dist, args.evaluate_condition_generation)
                nll_val = test(args, dataloaders['val'], dataset_info, epoch, model_ema_dp, 
                            property_norms, nodes_dist, partition='Val')
                nll_test = test(args, dataloaders['test'], dataset_info, epoch, model_ema_dp, 
                                property_norms, nodes_dist, partition='Test')
                        
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test

                if args.save_model and epoch >= args.save_epoch:  
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            # if args.save_model:
            #     utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
            #     utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
            #     if args.ema_decay > 0:
            #         utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
            #     with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
            #         pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='E3Diffusion')
    parser.add_argument('--dataset_folder_path', type=str, default='./mp20/raw',)
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
    parser.add_argument('--sample_batch_size', type=int, default=10)
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
    parser.add_argument('--online', type=int, default=1, help='1 = wandb online -- 0 = wandb offline')
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
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 16, 3, 18],
                        help='normalize factors for [x, categorical, integer, lengths, angles]')
    parser.add_argument('--normalize_biases', type=eval, default=[0, 0, 0, 0, 0],
                        help='normalize biases for [x, categorical, integer, lengths, angles]')
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

    parser.add_argument("--lambda_l", type=float, default=1.0, help="the loss weight of lattice lengths")
    parser.add_argument("--lambda_a", type=float, default=1.0, help="the loss weight of lattice angles")

    parser.add_argument("--visulaize_epoch", type=int, default=60, help="visualize after this epoch")   
    # visulaize_epoch之后打印所需的实验信息
    # parser.add_argument("--record_more_info", type=int, default=1, help="record more information about loss in visulaize_epoch")
    parser.add_argument("--n_samples", type=int, default=10, help="number of samples for visualization")
    parser.add_argument("--frac_coords_mode", type=int, default=0, help="whether use frac_coords")
    parser.add_argument("--save_epoch", type=int, default=150, help="begin to save model")

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

    # 设置环境变量
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    os.environ["WANDB_SERVICE_WAIT"] = "120" 
    os.environ["WANDB_MODE"] = mode

    kwargs = {
        'entity': args.wandb_usr, 
        'name': args.exp_name, 
        'project': 'e3_diffusion', 
        'config': args,
        'settings': wandb.Settings(_disable_stats=True),
        'mode': mode
    }

    # 直接初始化，如果失败则切换到离线模式
    try:
        wandb.init(**kwargs)
    except wandb.errors.CommError:
        print("Online mode failed, switching to offline...")
        os.environ["WANDB_MODE"] = "offline"
        kwargs['mode'] = 'offline'
        wandb.init(**kwargs)

    wandb.save('*.txt')

    """******  main function  ******"""

    main(args)

    """******  train & test  ******"""

