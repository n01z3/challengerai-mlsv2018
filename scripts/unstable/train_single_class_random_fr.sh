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

TRAIN_DATA_DIR="/mnt/ssd1/dataset/train/"
TRAIN_ANN_FILE="/mnt/ssd1/dataset/short_video_trainingset_annotations.txt"
VAL_DATA_DIR="/mnt/ssd1/dataset/val/"
VAL_ANN_FILE="/mnt/ssd1/dataset/short_video_validationset_annotations.txt"
echo "use following GPUs" $N_GPU

#baseline parameters
height=224
width=224
learning_rate=$3
bath_size=$4
network=se_resnet50
epochs=150

LOG="logs/un_baseline_single_label_random_`date +'%Y-%m-%d_%H-%M-%S'`/"

echo Logging output to "$LOG"

echo Training data dir "$TRAIN_DATA_DIR"
echo Training annotation file "$TRAIN_ANN_FILE"

echo Validation data dir "$VAL_DATA_DIR"
echo Validation annotation file "$VAL_ANN_FILE"



CUDA_VISIBLE_DEVICES=$N_GPU python train_net_single_label.py \
     -b $bath_size \
     -a $network \
     --epochs $epochs  \
     --lr $learning_rate \
     --width $width \
     --height $height \
     --logs-dir $LOG  \
     --train_data_dir $TRAIN_DATA_DIR \
     --train_ann_file $TRAIN_ANN_FILE \
     --val_data_dir $VAL_DATA_DIR \
     --val_ann_file $VAL_ANN_FILE \
     --gpu 0 \
     --n_frames 2