import numpy as np
import getpass
import os
import torch

# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


# def gradient_clipping(flow, gradnorm_queue):
        
#     # Allow gradient norm to be 150% + 2 * stdev of the recent history.
#     min_clip = 1.0  # 或者 100，根据你梯度量级
#     max_grad_norm = max(1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std(), min_clip)

#     # Clips gradient and returns the norm
#     grad_norm = torch.nn.utils.clip_grad_norm_(
#         flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

#     if float(grad_norm) > max_grad_norm:
#         gradnorm_queue.add(float(max_grad_norm))
#     else:
#         gradnorm_queue.add(float(grad_norm))

#     if float(grad_norm) > max_grad_norm:
#         print(f'Clipped gradient with value {grad_norm:.1f} '
#               f'while allowed {max_grad_norm:.1f}')
#     return grad_norm


# def gradient_clipping(flow, gradnorm_queue, min_clip=100.0, safety_factor=1.5):
#     """
#     自适应梯度剪裁函数，兼容你的 Queue 类。
#     参数:
#         flow: 模型
#         gradnorm_queue: Queue 对象，记录最近梯度范数
#         min_clip: 队列未积累够数据时使用的最小梯度阈值
#         safety_factor: 队列均值的放大系数
#     """
#     # 队列均值与标准差
#     queue_mean = gradnorm_queue.mean() if len(gradnorm_queue) > 0 else 0.0
#     queue_std = gradnorm_queue.std() if len(gradnorm_queue) > 0 else 0.0
#     # 最大梯度阈值
#     max_grad_norm = max(safety_factor * queue_mean + 2 * queue_std, min_clip)
#     # 梯度剪裁
#     grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)
#     # 更新队列
#     gradnorm_queue.add(min(float(grad_norm), float(max_grad_norm)))
#     # 打印剪裁信息
#     if grad_norm > max_grad_norm:
#         print(f'Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')
#     return grad_norm

def gradient_clipping(flow, gradnorm_queue, min_clip=100.0, safety_factor=1.5):
    """
    自适应梯度剪裁函数（修正版）
    - 支持 torch.nn.DataParallel
    - 支持 Queue 动态阈值
    """
    # 兼容 DataParallel
    if isinstance(flow, torch.nn.DataParallel):
        params = flow.module.parameters()
    else:
        params = flow.parameters()
    # 队列均值与标准差
    queue_mean = gradnorm_queue.mean() if len(gradnorm_queue) > 0 else 0.0
    queue_std = gradnorm_queue.std() if len(gradnorm_queue) > 0 else 0.0
    # 动态计算最大梯度阈值
    max_grad_norm = max(safety_factor * queue_mean + 2 * queue_std, min_clip)
    # 实际梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm, norm_type=2.0)
    # 更新队列
    gradnorm_queue.add(min(float(grad_norm), float(max_grad_norm)))
    # 打印日志
    if grad_norm > max_grad_norm:
        print(f'⚠️ Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')

    return grad_norm


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


# Other utilities
def get_wandb_username(username):
    if username == 'cvignac':
        return 'cvignac'
    current_user = getpass.getuser()
    if current_user == 'victor' or current_user == 'garciasa':
        return 'vgsatorras'
    else:
        return username


if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)
