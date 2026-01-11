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

from tqdm.auto import tqdm

# 移除 E0 相关的 import

from extxyz_to_pyg_custom import extxyz_to_pyg_custom


# ==========================================

# Worker 任务

# ==========================================

def worker_task(args):
    # 双重保险：在进程内再次强制设置 PyTorch 线程

    torch.set_num_threads(1)

    # 移除 need_e0_sample 参数

    (worker_id, file_paths, save_dir, prefix, cutoff, chunk_size) = args

    buffer = []

    save_counter = 0

    # 移除 e0_samples 列表

    try:

        for fpath in file_paths:

            if os.path.getsize(fpath) == 0: continue

            try:

                data_list = extxyz_to_pyg_custom(fpath, cutoff=cutoff)

            except Exception:

                continue

            if not data_list: continue

            for data in data_list:

                buffer.append(data)

                # [已移除] E0 采样逻辑

                # 存盘逻辑

                if len(buffer) >= chunk_size:
                    save_name = f"{prefix}_w{worker_id}_part_{save_counter}.pt"

                    torch.save(buffer, os.path.join(save_dir, save_name))

                    buffer = []

                    save_counter += 1

                    gc.collect()  # 释放内存

        # 处理剩余数据

        if len(buffer) > 0:
            save_name = f"{prefix}_w{worker_id}_part_{save_counter}.pt"

            torch.save(buffer, os.path.join(save_dir, save_name))

            buffer = []

            gc.collect()

        return True  # 不再返回 E0 样本，仅返回完成标志



    except Exception as e:

        print(f"❌ Worker-{worker_id} Error: {e}")

        return False


# ==========================================

# 管理器

# ==========================================

def process_manager(file_files, save_dir, prefix, num_workers, chunk_size, cutoff):
    # 移除 calc_e0 参数

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 动态调整 worker 数量

    real_workers = min(num_workers, len(file_files))

    if real_workers == 0: return

    chunked_files = np.array_split(file_files, real_workers)

    tasks = []

    for i in range(real_workers):
        # 移除任务参数中的 calc_e0

        tasks.append((i, chunked_files[i].tolist(), save_dir, prefix, cutoff, chunk_size))

    print(f"🚀 [Start] {prefix}: {len(file_files)} files -> {real_workers} Workers")

    # 不再收集 E0 结果，只是单纯跑完进度条

    with multiprocessing.Pool(processes=real_workers) as pool:

        for _ in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers):
            pass

    return


# ==========================================

# 主程序

# ==========================================

def main():
    # 1. 尝试修改共享策略

    try:

        torch.multiprocessing.set_sharing_strategy('file_system')

    except:
        pass

    # 2. 准备文件

    file_dirs = [r"005_all", r"100_all", r"outcar_selected_xyz", r"xyz_grouped"]

    all_files = []

    # 修正了你原代码中 unqie_names 集合生成的逻辑错误

    unique_names = set()

    for d in file_dirs:

        if os.path.exists(d):

            # 获取完整路径

            files_in_dir = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xyz')]
            all_files.extend(files_in_dir)

            # 获取唯一标识名

            for f in os.listdir(d):

                if f.endswith('.xyz'):
                    unique_names.add(f.split('.')[0])
    print(f"📂 Found {len(all_files)} files with {len(unique_names)} unique names.")

    # 🔥🔥🔥 核心参数建议 🔥🔥🔥

    NUM_WORKERS = 96  # 建议 8-16，IO瓶颈下 64 可能会卡顿

    TRAIN_CHUNK_SIZE = 5120

    test_ratio = 0.05

    SAVE_DIR = "processed_dataset"

    CUTOFF = 6.0

    # 划分数据集 (基于 unique name)

    unique_names_list = sorted(list(unique_names))  # 转为列表并排序

    random.seed(42)

    random.shuffle(unique_names_list)

    num_test = max(1, int(len(unique_names_list) * test_ratio))

    test_names_set = set(unique_names_list[:num_test])

    train_names_set = set(unique_names_list[num_test:])

    # 重新过滤文件

    train_files = []

    test_files = []

    for f in all_files:

        fname = f.split(os.sep)[-1].split('.')[0]

        if fname in train_names_set:

            train_files.append(f)

        elif fname in test_names_set:

            test_files.append(f)

    # 简单打乱文件顺序

    random.shuffle(train_files)

    random.shuffle(test_files)

    print(f"📝 Tasks: Train={len(train_files)}, Test={len(test_files)}")

    print(f"⚙️ Config: Workers={NUM_WORKERS}, Chunk={TRAIN_CHUNK_SIZE}")

    # 执行处理 (移除了 calc_e0 参数和返回值接收)

    print("\n--- Processing Test Set ---")

    process_manager(test_files, SAVE_DIR, "test", NUM_WORKERS, 5120, CUTOFF)

    print("\n--- Processing Train Set ---")

    process_manager(train_files, SAVE_DIR, "train", NUM_WORKERS, TRAIN_CHUNK_SIZE, CUTOFF)

    # [已移除] E0 计算和 meta_data.pt 保存部分

    print("\n✅ All processing finished.")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    main()
