ulimit -n 65535
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=5 Train_dist_restart.py
#python Train.py
