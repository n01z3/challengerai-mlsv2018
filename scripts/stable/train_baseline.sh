#!/usr/bin/env bash
# Usage: 
# ./scripts/stable/train_baseline.sh GPU NET
# 
#  Example:
# Use 0, 2, and 3 GPU to train network
# ./scripts/stable/train_baseline.sh 0,2,3 se_resnet50 0.0001
# Use 2 GPUs (0, 1) to train network0.0001 128
# ./scripts/stable/train_baseline.sh 2 se_resnet50

echo "This script is used to train baseline"

echo "Current directory is " $PWD

: ${1? "Usage: $0 GPU NETWORK LR"}


if [ ${#1} -eq 1 ]
then
  echo "arrange gpus"
  N_GPU="$(($1 - 1))"
  N_GPU="$(seq -s ',' 0 $N_GPU)"
else
  echo "use user provided gpus"
  N_GPU=$1
fi

NET=$2

TRAIN_DATA_DIR="/mnt/ssd1/dataset/train_data/"
VAL_DATA_DIR="/mnt/ssd1/dataset/val_data/"

echo "use following GPUs" $N_GPU

#baseline parameters
height=224
width=224
learning_rate=$3
bath_size=64
network=se_resnet50
epochs=150

LOG="logs/baseline_`date +'%Y-%m-%d_%H-%M-%S'`/"

echo Logging output to "$LOG"

echo Training data dir "$TRAIN_DATA_DIR"
echo Validation data dir "$VAL_DATA_DIR"


CUDA_VISIBLE_DEVICES=$N_GPU python train_net.py \
     -b $bath_size \
     -a $network \
     --epochs $epochs  \
     --lr $learning_rate \
     --width $width \
     --height $height \
     --logs-dir $LOG  \
     --train_data_dir $TRAIN_DATA_DIR \
     --val_data_dir $VAL_DATA_DIR