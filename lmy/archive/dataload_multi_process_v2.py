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

# 确保你的辅助函数在同级目录下
from compute_average_e0 import compute_average_e0
from extxyz_to_pyg_custom import extxyz_to_pyg_custom

# ==========================================
# Worker 任务
# ==========================================
def worker_task(args):
    # 双重保险：在进程内再次强制设置 PyTorch 线程
    torch.set_num_threads(1)
    
    (worker_id, file_paths, save_dir, prefix, cutoff, chunk_size, need_e0_sample) = args
    
    buffer = []
    save_counter = 0
    e0_samples = []
    
    try:
        # 调试打印：确认该 Worker 启动
        # print(f"🔧 Worker-{worker_id} 启动，处理 {len(file_paths)} 个文件")
        
        for fpath in file_paths:
            if os.path.getsize(fpath) == 0: continue
            
            try:
                data_list = extxyz_to_pyg_custom(fpath, cutoff=cutoff)
            except Exception:
                continue
            
            if not data_list: continue
            
            for data in data_list:
                buffer.append(data)
                
                # 收集少量 E0 样本
                if need_e0_sample and len(e0_samples) < 3000:
                    e0_samples.append(data)

                # 存盘逻辑
                if len(buffer) >= chunk_size:
                    save_name = f"{prefix}_w{worker_id}_part_{save_counter}.pt"
                    torch.save(buffer, os.path.join(save_dir, save_name))
                    buffer = [] 
                    save_counter += 1
                    gc.collect() # 释放内存
        
        # 处理剩余数据
        if len(buffer) > 0:
            save_name = f"{prefix}_w{worker_id}_part_{save_counter}.pt"
            torch.save(buffer, os.path.join(save_dir, save_name))
            buffer = []
            gc.collect()
            
        return e0_samples
        
    except Exception as e:
        print(f"❌ Worker-{worker_id} Error: {e}")
        return []

# ==========================================
# 管理器
# ==========================================
def process_manager(file_files, save_dir, prefix, num_workers, chunk_size, cutoff, calc_e0):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # 动态调整 worker 数量
    real_workers = min(num_workers, len(file_files))
    if real_workers == 0: return []
    
    chunked_files = np.array_split(file_files, real_workers)
    
    tasks = []
    for i in range(real_workers):
        tasks.append((i, chunked_files[i].tolist(), save_dir, prefix, cutoff, chunk_size, calc_e0))
    
    print(f"🚀 [Start] {prefix}: {len(file_files)} files -> {real_workers} Workers")
    
    collected_e0 = []
    
    # 使用 spawn 启动方式可以更彻底地隔离环境（可选，但通常 fork 就够了如果 env 设置得早）
    # ctx = multiprocessing.get_context('spawn')
    # with ctx.Pool(processes=real_workers) as pool:
    
    with multiprocessing.Pool(processes=real_workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_task, tasks), total=real_workers):
            collected_e0.extend(res)
            
    return collected_e0

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 尝试修改共享策略
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except: pass

    # 2. 准备文件
    file_dirs = [r"005", r"100", r"outcar_selected_xyz", r"xyz_grouped"]
    all_files = []
    unqie_names = set()
    for d in file_dirs:
        if os.path.exists(d):
            all_files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xyz')]) # 完整路径
            unqie_names.add(f.split('.')[0] for f in os.listdir(d) if f.endswith('.xyz'))
    
    # 🔥🔥🔥 核心参数建议 🔥🔥🔥
    # 即使你有 60 核，也不要超过 16。IO 瓶颈下，核多反而慢。
    NUM_WORKERS = 64      # 建议 8-12，绝对不要 60
    TRAIN_CHUNK_SIZE = 5120
    test_ratio = 0.05
    SAVE_DIR = "processed_dataset"
    CUTOFF = 6.0
    
    # 划分
    # 按照unique name划分，确保同一材料不在train和test里
    random.seed(42)
    all_files_sorted = sorted(all_files, key=lambda x: x.split(os.sep)[-1].split('.')[0]) # 按名称排序，确保同一材料文件在一起
    unique_names = sorted(list(unqie_names)) # 排序以确保可复现
    random.shuffle(unique_names)
    num_test = max(1, int(len(unique_names) * test_ratio)) # 至少1个
    test_names = set(unique_names[:num_test]) # 测试集名称
    train_names = set(unique_names[num_test:]) # 训练集名称
    # 完整路径分配
    train_files = [f for f in all_files_sorted if f.split(os.sep)[-1].split('.')[0] in train_names]
    test_files = [f for f in all_files_sorted if f.split(os.sep)[-1].split('.')[0] in test_names]
    manual = False
    if manual:
        train_files = train_files[:train_end]
        test_files = test_files[train_end:test_end]
    else:
        train_files = train_files
        test_files = test_files
        
    print(f"📝 Tasks: Train={len(train_files)}, Test={len(test_files)}")
    print(f"⚙️ Config: Workers={NUM_WORKERS}, Chunk={TRAIN_CHUNK_SIZE}")

    # 执行
    process_manager(test_files, SAVE_DIR, "test", NUM_WORKERS, 5120, CUTOFF, False)
    train_e0 = process_manager(train_files, SAVE_DIR, "train", NUM_WORKERS, TRAIN_CHUNK_SIZE, CUTOFF, True)
#    process_manager(test_files, SAVE_DIR, "test", NUM_WORKERS, 2000, CUTOFF, False)

    # 保存 Meta
    if train_e0:
        print("Calculating E0...")
        e0_dict = compute_average_e0(train_e0[:3000])
        torch.save({'e0_dict': e0_dict}, os.path.join(SAVE_DIR, "meta_data.pt"))
        print("✅ Done.")

if __name__ == '__main__':
    # 只有在 main 里才 freeze
    multiprocessing.freeze_support()
    main()
