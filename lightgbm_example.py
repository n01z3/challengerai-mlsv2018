from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import glob
import numpy as np
import sys
import torch
from torch import nn

from sklearn.decomposition import PCA
import lightgbm as lgb

from sklearn.metrics import accuracy_score
#sorted(glob.glob('5_frames_val/torch_tags_*.th'), key = os.path.getmtime)
def main(args):
    #load train features
    train_features = [torch.load(c) for c in sorted(glob.glob(osp.join(args.train_dir, "torch_features_*.th")), key = os.path.getmtime)]
    train_features = torch.cat((train_features), dim = 0)
    train_tags = [torch.load(c) for c in sorted(glob.glob(osp.join(args.train_dir, "torch_tags_*.th")), key = os.path.getmtime)]
    train_tags = torch.cat((train_tags), dim = 0)

    #load validation features
    val_features = [torch.load(c) for c in sorted(glob.glob(osp.join(args.val_dir, "torch_features_*.th")), key = os.path.getmtime)]
    val_features = torch.cat((val_features), dim = 0)
    val_tags = [torch.load(c) for c in sorted(glob.glob(osp.join(args.val_dir, "torch_tags_*.th")), key = os.path.getmtime)]
    val_tags = torch.cat((val_tags), dim = 0)

    train_features = torch.mean(train_features, 1)
    val_features = torch.mean(val_features, 1)

    print ('train features: {}'.format(train_features.size()))
    print ('val features: {}'.format(val_features.size()))

    print(train_features.size())

    X_train = train_features.numpy()
    y_train = train_tags.numpy().astype(int)
    #print(train_features)

    X_val = val_features.numpy()
    y_val = val_tags.numpy().astype(int)

    n_classes = len(np.unique(y_train))

    print ('X_train {} | y_train {} | X_val {} | y_val {}'.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    #PCA
    pca = PCA(n_components=64)

    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    #lightgbm
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, free_raw_data=False)

    params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multiclass',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_class': n_classes, 
    }

    print('Start training...')
    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_train)  # eval training data

    
    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    print('accuracy score: {}'.format(accuracy_score(y_val, np.argmax(y_pred, axis = 1))))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lightgbm example")
    # model
    parser.add_argument('--train_dir', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--val_dir', type=str, metavar='PATH', help = "path to the data folder")

    main(parser.parse_args())