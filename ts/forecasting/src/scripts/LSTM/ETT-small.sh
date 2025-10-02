#!/bin/bash

window_size=96
data_path=/workspace/DSBAPretraining/data/datasets/all_datasets/ETT-small
data_names=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
model_name=LSTM
batch_size=64

for data_name in "${data_names[@]}"; do
  echo "Running dataset: $data_name"

  accelerate launch main.py \
    --model_name $model_name \
    --default_cfg /workspace/DSBAPretraining/Time-series/Time-series-forecasting/src/configs/default_setting.yaml \
    --model_cfg /workspace/DSBAPretraining/Time-series/Time-series-forecasting/src/configs/model_setting.yaml \
    DATASET.window_size=$window_size \
    DATASET.datadir=$data_path \
    DATASET.dataname=$data_name \
    DATASET.pred_len=96 \
    DEFAULT.exp_name=forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size=$batch_size \
    MODELSETTING.d_model=16

done
