# import pandas as pd
# import os
# dataset_folder_path = "./mp20/raw"
# dataset_cif_list=pd.read_csv(
#                 os.path.join(dataset_folder_path, f"all.csv")
#             )["cif"].tolist()

import torch
a = []
b = [1,2,3,4]
c = torch.tensor(a+b, dtype=torch.float32)
print(c)
print(c.sum())
print(c.mean())
print(c.std())

d = torch.tensor([[1,2,3,4], 
                  [1,2,3,4]], 
                  dtype=torch.float32)
print(d.sum(dim=1))

e = []
f = [1,2,3,4]
e.append(f)
print(e)

a = torch.ones((2,3))
print(a[:, :2])

#######################################################
import torch

# 模型 softmax 输出 (batch_size=3, num_classes=4)
p = torch.tensor([
    [0.7, 0.1, 0.1, 0.1],
    [0.05, 0.05, 0.05, 0.85],
    [0.1, 0.8, 0.05, 0.05]
], dtype=torch.float32)

# one-hot 标签
y = torch.tensor([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
], dtype=torch.float32)

import torch.nn.functional as F

# 模型输出 logits (可选) 或 softmax 概率
log_p = torch.log(p + 1e-9)
loss = F.kl_div(log_p, y, reduction='batchmean')
print(loss)

#######################################################
import torch
import torch.nn as nn

# 模拟网络输出（未经过 softmax 的 logits）
# 假设 batch=3，每个样本有 4 个类别
logits = torch.tensor([
    [2.0, 1.0, 0.1, -1.0],
    [0.5, 2.2, -0.3, 0.0],
    [-1.0, 0.3, 2.0, 0.7]
])  # [N, num_classes]

# 模拟标签（类别索引，不是 one-hot！）
# 正确类别分别是 0, 1, 2
labels = torch.tensor([0, 1, 2])    # [N]

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

print("CrossEntropyLoss:", loss.item())
