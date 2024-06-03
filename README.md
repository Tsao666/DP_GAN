# AI CUP 2024 春季賽：以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 I － 影像資料生成競賽

TEAM_5101，feat. [@Tianming8585](https://github.com/Tianming8585)'s [PITI](https://github.com/Tianming8585/PITI)
[報告](report.md) [Ensemble](ensemble.md)

## 資料前處理

資料存放路徑與前處理參考 [UAV.py](./dataloaders/UAV.py)

## 訓練

```bash
python train.py --name UAV_instance_512_z128_400 --dataset_mode UAV_34_train --param_free_norm instance --z_dim 128 --num_epochs 400 --gpu_ids 0 --batch_size 8
```

## 生成影像

```bash
python test.py --name UAV_instance_512_z128_400 --dataset_mode UAV_34_public --param_free_norm instance --z_dim 128 --ckpt_iter best --results_dir private --gpu_ids 0 --batch_size 8
```

## 生成 D 網路輸出

```
python eval_pri.py --name UAV_instance_512_z128_400 --dataset_mode UAV_34_public --param_free_norm instance --z_dim 128 --gpu_ids 0 --ckpt_iter best --results_dir private
```

