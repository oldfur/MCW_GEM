# ==========================================
# 🔥 必须放在文件最最最开头！🔥
# 在导入任何 torch/numpy 之前就限制线程
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ==========================================
# 现在才开始导入其他库
# ==========================================
import torch
import random
import multiprocessing
import gc
import numpy as np
from tqdm import tqdm

# 假设你的转换函数在这里 (请确保该文件在同一目录下)
from extxyz_to_pyg_custom_new import extxyz_to_pyg_custom 

def worker_task(args):
    """
    Worker 进程任务：
    1. 读取 XYZ 文件
    2. 转换为 PyG 图数据 (计算边/邻居)
    3. 保存数据块 (.pt)
    4. 返回元数据 (Metadata) 给主进程
    """
    # 双重保险
    torch.set_num_threads(1)
    
    worker_id, file_paths, save_dir, prefix, cutoff, chunk_size = args
    
    buffer = []
    local_metadata = [] # 🔥 重点：只存索引信息
    save_counter = 0
    
    try:
        for fpath in file_paths:
            if os.path.getsize(fpath) == 0: continue

            # 1. 这里进行最耗时的图计算 (计算边、邻居)
            # extxyz_to_pyg_custom 内部应调用 neighbor_list 或 radius_graph
            try:
                data_list = extxyz_to_pyg_custom(fpath, cutoff=cutoff)
            except Exception as e:
                print(f"Skipping bad file {fpath}: {e}")
                continue
            
            if not data_list: continue

            for data in data_list:
                buffer.append(data)
                
                # 🔥 2. 记录元数据 (为多目标优化做准备)
                # 记录：这个图在哪个文件的哪个位置，有多大
                # 这个 dict 非常小，几百万个样本也就几百 MB 内存
                local_metadata.append({
                    'file_path': f"{prefix}_w{worker_id}_p{save_counter}.pt", # 数据存在哪个文件
                    'index_in_file': len(buffer) - 1,                         # 文件里的第几个
                    'num_atoms': data.num_nodes,                              # 显存瓶颈
                    'num_edges': data.edge_index.size(1)                      # 计算瓶颈
                })

                # 3. 存盘逻辑
                if len(buffer) >= chunk_size:
                    save_name = f"{prefix}_w{worker_id}_p{save_counter}.pt"
                    full_path = os.path.join(save_dir, save_name)
                    
                    # 使用 torch.save 保存 buffer
                    torch.save(buffer, full_path)
                    
                    buffer = []
                    save_counter += 1
                    gc.collect() # 显式 GC 防止内存泄漏

        # 处理剩余数据 (Last Chunk)
        if buffer:
            save_name = f"{prefix}_w{worker_id}_p{save_counter}.pt"
            torch.save(buffer, os.path.join(save_dir, save_name))
            gc.collect()
        
        return local_metadata # 返回元数据给主进程合并

    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        # 出错时尽量返回已收集的元数据，避免全部丢失
        return local_metadata

def main():
    # 1. 设置共享策略 (防止 Too many open files 错误)
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass

    # 2. 准备文件路径
    # 请根据你的实际目录修改这里
    file_dirs = [r"../005_all", r"../100_all", r"../outcar_selected_xyz", r"../xyz_grouped"]
    all_files = []
    unique_names = set()

    print("🔍 Scanning files...")
    for d in file_dirs:
        if os.path.exists(d):
            # 获取完整路径
            files_in_dir = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xyz')]
            all_files.extend(files_in_dir)

            # 获取唯一标识名 (防止数据泄漏)
            for f in os.listdir(d):
                if f.endswith('.xyz'):
                    unique_names.add(f.split('.')[0])
    
    print(f"📂 Found {len(all_files)} files with {len(unique_names)} unique names.")

    # 3. 配置参数
    NUM_WORKERS = 120    # 建议 8-16，太高会卡 IO
    CHUNK_SIZE = 100   # 每个 .pt 文件存多少个图 (越大读取越快，但随机性越差)
    SAVE_DIR = "../processed_data"
    CUTOFF = 6.0
    TEST_RATIO = 0.05
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 4. 划分数据集 (按 unique name)
    unique_names_list = sorted(list(unique_names))  # 排序保证可复现
    random.seed(42)
    random.shuffle(unique_names_list)
    
    num_test = max(1, int(len(unique_names_list) * TEST_RATIO))
    test_names_set = set(unique_names_list[:num_test])
    train_names_set = set(unique_names_list[num_test:])

    # 重新过滤文件
    train_files = []
    test_files = []
    
    for f in all_files:
        # 假设文件名格式为 "MoleculeName.xyz" 或 "MoleculeName.config1.xyz"
        # 这里取第一个点前的部分作为唯一标识
        fname = f.split(os.sep)[-1].split('.')[0] 

        if fname in train_names_set:
            train_files.append(f)
        elif fname in test_names_set:
            test_files.append(f)

    # 简单打乱文件顺序 (预Shuffle)
    random.shuffle(train_files)
    random.shuffle(test_files)
    print(f"🚂 Training files: {len(train_files)}, Testing files: {len(test_files)}")


    # ==========================================
    # 处理测试集 (Test)
    # ==========================================
    if test_files:
        print(f"\n🚀 Processing Test Set ({len(test_files)} files)...")
        
        real_workers = min(NUM_WORKERS, len(test_files))
        if real_workers > 0:
            file_chunks = np.array_split(test_files, real_workers)
            tasks = []
            for i in range(real_workers):
                tasks.append((i, file_chunks[i].tolist(), SAVE_DIR, "test", CUTOFF, CHUNK_SIZE))
            
            all_test_metadata = []
            with multiprocessing.Pool(real_workers) as pool:
                for meta in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers):
                    all_test_metadata.extend(meta)

            # 🔥 保存测试集总索引文件
            torch.save(all_test_metadata, os.path.join(SAVE_DIR, "test_metadata.pt"))
            print(f"✅ Test Done! Metadata saved: {len(all_test_metadata)} samples.")


    # ==========================================
    # 处理训练集 (Train)
    # ==========================================
    if train_files:
        print(f"\n🚀 Processing Train Set ({len(train_files)} files)...")
        
        # 动态分配 Worker
        real_workers = min(NUM_WORKERS, len(train_files))
        file_chunks = np.array_split(train_files, real_workers)
        
        tasks = []
        for i in range(real_workers):
            # args: worker_id, file_paths, save_dir, prefix, cutoff, chunk_size
            tasks.append((i, file_chunks[i].tolist(), SAVE_DIR, "train", CUTOFF, CHUNK_SIZE))
        
        all_train_metadata = []
        
        with multiprocessing.Pool(real_workers) as pool:
            # imap 保证有序返回结果，或者用 imap_unordered 更快但乱序
            # 这里用 tqdm 包装进度条
            for meta in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers):
                all_train_metadata.extend(meta)

        # 🔥 保存训练集总索引文件
        torch.save(all_train_metadata, os.path.join(SAVE_DIR, "train_metadata.pt"))
        print(f"✅ Train Done! Metadata saved: {len(all_train_metadata)} samples.")

    print("\n🎉 All processing finished successfully.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()