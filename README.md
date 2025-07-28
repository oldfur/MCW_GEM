# 晶体生成

- 条件生成
- 可视化

## debug (非条件生成)

```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2 --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1
```

## debug (条件生成)
```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --visualize_every_batch 100 --num_train 1000 --conditioning band_gap
```

## debug (测试test， visualize_every_batch设为10000即可)
```
python main_mp20.py --device cpu --no-cuda --exp_name debug_mp20 --n_epochs 2 --batch_size 2  --test_epochs 1 --wandb_usr maochenwei-ustc --no_wandb --model DGAP --atom_type_pred 1 --property_pred 1 --target_property band_gap --visualize_every_batch 10000 --num_train 1000 --conditioning band_gap
```

## train

```
python main_mp20.py --exp_name mp20_egnn_dynamics --n_epochs 200 --model DGAP --atom_type_pred 1 --test_epochs 10 --batch_size 64