## Data Preparation
```
cd data/MovieLens1M
python preprocess.py
```

## Running Scripts
### MovieLens1M + DLRM
Standalone
```
python standalone_main.py --dataset_name MovieLens1M --model_name dlrm --criterion_name mse --evaluation_name rmse --task_num 1 --epoch 5 --device cpu
```

Federated Learning
```
python federated_main.py --dataset_name MovieLens1M --model_name dlrm --criterion_name mse --evaluation_name rmse --task_num 1 --epoch 50 --device cpu
```