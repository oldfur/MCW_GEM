# 晶体生成

## 环境配置
```
# python版本: 3.10.16
conda create -n mpgem python=3.10.16

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.7.1+cu126.html

pip install -r requirements.txt

pip install lightning_utilities svgwrite importlib_resources
```

- 条件生成
- 可视化

## debug (非条件生成)

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2 --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1
```

测试diffusion_another:

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP  --atom_type_pred 1 --visualize_every_batch 10000 --num_train 100  --num_val 100 --num_test 100 --lambda_l 1.0 --lambda_a 1.0 --visulaize_epoch 1 --probabilistic_model diffusion_another
```

## debug (条件生成, 测试train_epoch中所有过程)

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --visualize_every_batch 100 --num_train 1000 --conditioning band_gap 
```

## debug (测试test， visualize_every_batch设为10000即可)

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --visualize_every_batch 10000 --num_train 1000 --conditioning band_gap --lambda_l 1.0 --lambda_a 1.0 --visulaize_epoch 0
```

## debug (frac_coords, 有test)

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --visualize_every_batch 10000 --num_train 1000 --conditioning band_gap --lambda_l 1.0 --lambda_a 1.0 --visulaize_epoch 0 --frac_coords_mode 1
```

## train

```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name mp20_egnn_dynamics --n_epochs 200 --model DGAP --atom_type_pred 1 --test_epochs 10 --batch_size 64
```

## train diffusion_new

```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_mp20 --n_epochs 400 --batch_size 64  --test_epochs 20 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1  --n_report_steps 8  --visulaize_epoch 200 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 200 --lambda_l 0.1 --lambda_a 1.0 --probabilistic_model diffusion_new
```

## train diffusion_another
```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_mp20 --n_epochs 400 --batch_size 64  --test_epochs 20 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1  --n_report_steps 8  --visulaize_epoch 200 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 200 --lambda_l 1.0 --lambda_a 1.0 --probabilistic_model diffusion_another
```

服务器(wandb离线)
```
CUDA_VISIBLE_DEVICES=3 python main_mp20.py --exp_name train_mp20 --n_epochs 600 --batch_size 128 --test_epochs 10 --wandb_usr maochenwei-ustc --model DGAP --atom_type_pred 1 --n_report_steps 16 --visulaize_epoch 100 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 400 --lambda_l 1 --lambda_a 1 --online 0 --num_workers 0 --probabilistic_model diffusion_another
```

## train lattice predictor

```
CUDA_VISIBLE_DEVICES=0 python train_lattice_egnn.py --exp_name train_lattice --n_epochs 600 --batch_size 128 --test_epochs 10 --wandb_usr maochenwei-ustc --model DGAP --atom_type_pred 1 --n_report_steps 16 --visulaize_epoch 20 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 400 --lambda_l 1 --lambda_a 1 --online 0 --num_workers 0 --probabilistic_model diffusion_another
```

## train diffusion_pure_x

服务器(wandb离线)
```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_lattice --n_epochs 600 --batch_size 128 --test_epochs 10 --wandb_usr maochenwei-ustc --model DGAP --atom_type_pred 1 --n_report_steps 16 --visulaize_epoch 20 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 400 --lambda_l 1 --lambda_a 1 --online 0 --num_workers 0 --probabilistic_model diffusion_pure_x

```

## train diffusion_concat

服务器(wandb离线)
```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_lattice --n_epochs 600 --batch_size 128 --test_epochs 10 --wandb_usr maochenwei-ustc --model DGAP --atom_type_pred 1 --n_report_steps 16 --visulaize_epoch 20 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 400 --lambda_l 0.1 --lambda_a 0.1 --online 0 --num_workers 0 --probabilistic_model diffusion_concat
```


## 只test不可视化(unconditional)
```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_mp20 --n_epochs 400 --batch_size 64  --test_epochs 20 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1  --n_report_steps 8  --visulaize_epoch 200 --visualize_every_batch 20000 --n_samples 20 --sample_batch_size 200 --lambda_l 0.1 --lambda_a 1.0
```

### 实验室服务器上还需设置--online 0 --num_workers 0

## conditional train(no wandb)

```
CUDA_VISIBLE_DEVICES=0 python main_mp20.py --exp_name train_mp20 --n_epochs 300 --batch_size 128  --test_epochs 40 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --conditioning band_gap --n_report_steps 8  --visulaize_epoch 100 --visualize_every_batch 200 --n_samples 10 --sample_batch_size 1000 --lambda_l 1.0 --lambda_a 1.0
```

## CrystalGRW 运行

![CrystalGRM-GPT](CrystalGRM架构.png){: style="display:block; margin:0 auto;" width=50%}

```
export PYTHONPATH=./src
python scripts/train.py --config_path conf/mp20_condition.yaml --output_path output_dir --ddp False
```


## debug diffusion_transformer

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --visulaize_epoch 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP  --atom_type_pred 1 --visualize_every_batch 10000 --num_train 100  --num_val 100 --num_test 100 --lambda_l 1.0 --lambda_a 1.0 --frac_coords_mode 1 --probabilistic_model diffusion_transformer --include_charges False
```
