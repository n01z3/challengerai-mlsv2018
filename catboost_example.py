import argparse
import os.path as osp
import os
import glob
import numpy as np
import sys
import torch
from torch import nn

from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool, cv

from sklearn.metrics import accuracy_score

from copy import deepcopy

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


    print(y_train, y_val)

    #PCA
    pca = PCA(n_components=64)

    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)


    model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    loss_function='MultiClass',
    random_seed=42,
    logging_level='Silent',
    learning_rate=0.02,
    iterations=2000,
    task_type="GPU"
    )


    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        logging_level='Verbose'  # you can uncomment this for text output
        #plot=True
    )

    predictions = model.predict(X_val)
    print(predictions, y_val)
    print('accuracy score: {}'.format(accuracy_score(y_val, predictions)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="catboost example")
    # model
    parser.add_argument('--train_dir', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--val_dir', type=str, metavar='PATH', help = "path to the data folder")

    main(parser.parse_args())