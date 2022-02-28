# Training Case Level Classification
```
python3 train_3dcnn.py --dataset_path ../skull --z_dim 48 --model_name resnet18
python3 train_3dcnn.py --dataset_path ../skull --z_dim 48 --model_name resnet34
python3 train_3dcnn.py --dataset_path ../skull --z_dim 48 --model_name resnet50
python3 train_3dcnn.py --dataset_path ../skull --z_dim 36 --model_name resnet101
```
# Training AE for Skull Point Detection
```
python3 train_ae.py -p ../dlcv_final/skull/
```
# Evaluate AE
```
python3 eval_ae.py -p [model_path] -o [output_path]
```