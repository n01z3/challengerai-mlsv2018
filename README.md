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

## Network training example
```
python train_net.py --train_ann_file /mnt/ssd1/dataset/new_short_video_trainingset_annotations.txt.txt --train_data_dir /mnt/ssd1/dataset/train_jpg/ --val_ann_fil
e /mnt/ssd1/dataset/new_short_video_validationset_annotations.txt.txt  --val_data_dir /mnt/ssd1/dataset/val_jpg/ -a se_resnet50
```

The better way
```
screen ./scipts/stable/train_baseline.sh 4 se_resnet50 0.0001
```

# NEW basseline
```
screen ./scripts/unstable/train_baseline.sh 4 se_resnet50 0.0001 128
```

## Network evaluation example
```
CUDA_VISIBLE_DEVICES=2 python eval_net.py -b 32 -a se_resnet50 -w /mnt/ssd1/easygold/challengerai-mlsv2018/logs/un_baseline_2018-10-10_16-57-05/checkpoint.pth.tar --ann_fi
le /mnt/ssd1/dataset/short_video_validationset_annotations.txt --data_dir /mnt/ssd1/dataset/val/ -t 1 --gpu
```

## Tags analysis
Look for the frequence of tags to determine what tags often meet together. May be useful for multilabel analysis.
```
python tags_analysis.py --ann_file short_video_trainingnset_annotations.txt --out_dir mnt/ssd1/dataser/train_tags_analysis
```
## Meta classifier
Run meta classifier. It is based on majority voting. You can include uo to 5 single model predictions(lightgbm, catboost, log regression, neural network from pickles. Time for one prediction is printed for every model.
```
CUDA_VISIBLE_DEVICES=2 python infer_meta_classifier.py --train_dir /mnt/ssd1/dataset/multi_class/6_frames_train_se_resnet_merged/ --val_dir /mnt/ssd1/dataset/multi_class/6_frames_val_se_resnet_merged/ -multi_label -val_1_frame -net_pred -lgb -logreg -svm -catboost -voting 
```
