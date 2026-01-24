ulimit -n 65535

# 1. 握手网卡 (必须有)
export NCCL_SOCKET_IFNAME=ens22f0np0

export NCCL_IB_CUDA_SUPPORT=0
# 2. 🔥【补丁A】防止节点内 PCIe 死锁 (保留 IB，但禁用 P2P)
export NCCL_P2P_DISABLE=1

# 3. 🔥【补丁B】强制过滤，只允许 IB 网卡做 RDMA (排除 ens 和 docker)
export NCCL_IB_HCA=^ens,eth,docker

# 4. 🔥【补丁C】打开日志，看看到底卡在哪
export NCCL_DEBUG=INFO

# 5. 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_IB_DISABLE

# 6. 启动
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=66666 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.0.5:29500 \
    Train_dist_restart.py
