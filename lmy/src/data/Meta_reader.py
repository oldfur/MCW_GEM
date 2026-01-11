import torch

SAVE_PATH = "processed_dataset.pt"

data = torch.load(SAVE_PATH, map_location='cpu', weights_only=False)

train_set = data['train_set']
test_set  = data['test_set']
e0_dict   = data['e0_dict']
config    = data['config']

print(len(train_set), len(test_set))
print(e0_dict)
print(config)
