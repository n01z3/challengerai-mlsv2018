# challengerai-mlsv2018


Simple baseline is composed of the following steps:
1. Extract the first frame from each video.

```
python scripts/video_to_frames.py --ann_file /mnt/ssd1/dataset/short_video_trainingset_annotations.txt.082902 --data_dir /mnt/ssd1/dataset/train/ --out_dir /mnt/ssd1/dataset/train_jpg
```

```
python scripts/video_to_frames.py --ann_file /mnt/ssd1/dataset/short_video_validationset_annotations.txt.0829 --data_dir /mnt/ssd1/dataset/val/ --out_dir /mnt/ssd1/dataset/val_jpg
```

2. Extract features

```
CUDA_VISIBLE_DEVICES=0 screen python extract_features.py --ann_file /mnt/ssd1/dataset/new_short_video_validationset_annotations.txt.txt --data_dir /mnt/ssd1/dataset/val_jpg/ --out_dir /mnt/ssd1/dataset/val_features
```

```
 CUDA_VISIBLE_DEVICES=0 screen python extract_features.py --ann_file /mnt/ssd1/dataset/new_short_video_trainingset_annotations.txt.txt --data_dir /mnt/ssd1/dataset/train_jpg/ --out_dir /mnt/ssd1/dataset/train_features
```

3. Employ lightgbm to classify extracted features
```
python lightgbm_example.py --train_dir /mnt/ssd1/dataset/train_features/ --val_dir /mnt/ssd1/dataset/val_features/
```

Network training example
```
python train_net.py --train_ann_file /mnt/ssd1/dataset/new_short_video_trainingset_annotations.txt.txt --train_data_dir /mnt/ssd1/dataset/train_jpg/ --val_ann_fil
e /mnt/ssd1/dataset/new_short_video_validationset_annotations.txt.txt  --val_data_dir /mnt/ssd1/dataset/val_jpg/ -a se_resnet50
```