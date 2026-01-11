import os
import torch
from compute_average_e0 import compute_average_e0
from extxyz_to_pyg_custom import extxyz_to_pyg_custom
from tqdm.auto import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# ==========================================
# 0. 预设：请确保你的辅助函数已定义
# ==========================================
# 假设 extxyz_to_pyg_custom(file_path, cutoff, topo_abalation) 已经定义
# 假设 compute_average_e0(dataset) 已经定义

# --- 辅助函数：批量读取 ---
def load_dataset_from_files(file_paths, cutoff=6.0):
    dataset = []
    print(f"Loading {len(file_paths)} files...")
    for file_path in tqdm(file_paths):
        # 跳过空文件
        if os.path.getsize(file_path) == 0:
            print(f"⚠️ [跳过] 空文件: {os.path.basename(file_path)}")
            continue
        
        # 调用你的转换函数
        data_list = extxyz_to_pyg_custom(file_path, cutoff=cutoff)
        dataset.extend(data_list)
    return dataset

# ==========================================
# 1. 用户配置区
# ==========================================
# for linux
file_dir_1 = r"005" # C:\Users\1\Desktop\traIning set\zip_files\005_part1
file_dir_2 = r"outcar_selected_xyz"
file_dir_3 = r"xyz_grouped"
all_files = [os.path.join(file_dir_1, f) for f in os.listdir(file_dir_1) if f.endswith('.xyz')] + \
            [os.path.join(file_dir_2, f) for f in os.listdir(file_dir_2) if f.endswith('.xyz')] + \
            [os.path.join(file_dir_3, f) for f in os.listdir(file_dir_3) if f.endswith('.xyz')]
# for windows
import random
random.shuffle(all_files)


# file_dir_1 = r"C:\Users\1\Desktop\traIning set\AIMD_selected_xyz\outcar_selected_xyz" # C:\Users\1\Desktop\traIning set\zip_files\005_part1
# file_dir_2 = r"C:\Users\1\Desktop\traIning set\zip_files\005_part1"

# all_files = [os.path.join(file_dir_1, f) for f in os.listdir(file_dir_1) if f.endswith('.xyz')] + \
#             [os.path.join(file_dir_2, f) for f in os.listdir(file_dir_2)[0:8000] if f.endswith('.xyz')]
#打乱
import random
random.shuffle(all_files)

# 模式选择
SPLIT_MODE = 'manual' 
# SPLIT_MODE = 'random' 
BATCH_SIZE = 8

# --- 💾 保存控制开关 ---
IS_SAVE = True               # 设置为 True 则保存，False 则不保存
SAVE_PATH = "processed_dataset.pt"  # 保存的文件名

# ==========================================
# 2. 数据加载与划分
# ==========================================
if SPLIT_MODE == 'manual':
    print(">>> 使用手动指定文件划分数据集")
    train_files = all_files[:20000]  
    test_files = all_files[20000:21000]
    train_set = load_dataset_from_files(train_files)
    test_set = load_dataset_from_files(test_files)

elif SPLIT_MODE == 'random':
    print(">>> 使用随机比例划分数据集 (Train: 90%, Test: 10%)")
    full_dataset = load_dataset_from_files(all_files)
    total_len = len(full_dataset)
    train_len = int(0.9 * total_len)
    test_len = total_len - train_len
    
    train_set, test_set = random_split(
        full_dataset, 
        [train_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )

print(f"✅ 数据集准备完成: Train={len(train_set)}, Test={len(test_set)}")

# ==========================================
# 3. 计算原子平均能量 (E0)
# ==========================================
print("计算原子平均能量 (E0)...")
e0_dict = compute_average_e0(train_set)
print(f"E0 计算完成: {e0_dict}")

# ==========================================
# 4. 保存处理后的数据 (根据 IS_SAVE 判断)
# ==========================================
if IS_SAVE:
    print(f"💾 开关已打开，正在保存数据到 {SAVE_PATH} ...")
    
    data_to_save = {
        'train_set': train_set,
        'test_set': test_set,
        'e0_dict': e0_dict,
        'config': {'split_mode': SPLIT_MODE, 'train_source': train_files, 'test_source': test_files}
    }
    
    try:
        torch.save(data_to_save, SAVE_PATH)
        print("🎉 保存成功！")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
else:
    print("⏩ IS_SAVE 为 False，跳过保存步骤。")

# ==========================================
# 5. 构建 DataLoader
# ==========================================
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print("🚀 Loader 构建完成，准备开始训练！")
