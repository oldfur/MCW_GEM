ulimit -n 65535
export NCCL_SOCKET_IFNAME=ens22f0np0

export NCCL_IB_CUDA_SUPPORT=0
# 🔥 加上这三个补丁
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=^ens,eth,docker
export NCCL_DEBUG=INFO

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_IB_DISABLE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
torchrun \
    --nproc_per_node=5 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=66666 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.0.5:29500 \
    Train_dist_restart.py
