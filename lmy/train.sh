ulimit -n 65535
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=4 --master_port=29601 ./lmy/Train_dist.py
#python Train.py

# 运行命令：nohup ./lmy/train.sh > training_log.txt 2>&1 &