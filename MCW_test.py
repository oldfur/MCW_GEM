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