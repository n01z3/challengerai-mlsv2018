from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import glob
import numpy as np
import time
import sys
import torch
from torch import nn
import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from catboost import CatBoostClassifier, Pool, cv

def main(args):
    train_features = [torch.load(c) for c in sorted(glob.glob(osp.join(args.train_dir, "torch_features_*.th")), key = os.path.getmtime)]
    train_features = torch.cat((train_features), dim = 0)
    train_tags = [torch.load(c) for c in sorted(glob.glob(osp.join(args.train_dir, "torch_tags_*.th")), key = os.path.getmtime)]
    train_tags = torch.cat((train_tags), dim = 0)

     #load validation features
    val_features = [torch.load(c) for c in sorted(glob.glob(osp.join(args.val_dir, "torch_features_*.th")), key = os.path.getmtime)]
    val_features = torch.cat((val_features), dim = 0)
    val_tags = [torch.load(c) for c in sorted(glob.glob(osp.join(args.val_dir, "torch_tags_*.th")), key = os.path.getmtime)]
    val_tags = torch.cat((val_tags), dim = 0)

    train_features = torch.max(train_features, 1)[0]
    val_features = torch.max(val_features, 1)[0]

    print ('train features: {}'.format(train_features.size()))
    print ('val features: {}'.format(val_features.size()))

    print(train_features.size())

    X_train = train_features.numpy().clip(min =0)
    y_train = train_tags.numpy().astype(int)

    X_val = val_features.numpy().clip(min = 0)
    y_val = val_tags.numpy().astype(int)
    n_classes = len(np.unique(y_train))

    if args.multi_label:
       y_train = y_train[:,0]
       y_val = y_val[:,0]
    print ('X_train {} | y_train {} | X_val {} | y_val {}'.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    #PCA
    with open("/mnt/ssd1/dataset/pickles/pca_model.pkl", "rb") as handle:
        pca = pickle.load(handle)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    start = time.time()
    if args.debug:
        X_train = X_train[:1]
        X_val=X_val[:1]
        y_train = y_train[:1]
        y_val = y_val[:1]
    # knn
    if args.knn:
        print("knn starting")
        def make_unsupervised_knn(data, N_NEIGHBORS):
            model = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm = 'ball_tree',n_jobs = -1) # ball tree works faster, for better results use 'auto'
            model.fit(data)
            k_distances, indices = model.kneighbors(data)

            res = pd.DataFrame()
            for i in range(1, N_NEIGHBORS):
                res["dist_{}_neigh".format(i)] = k_distances[:, i]
            return res

        def prepare_feats(data, feat):
            feat = feat.values
            feats = np.concatenate((data, feat), axis=1)
            return feats
        feat_train = make_unsupervised_knn(X_train,4)
        feat_val = make_unsupervised_knn(X_val,4)
        X_train = prepare_feats(X_train,feat_train)
        X_val = prepare_feats(X_val, feat_val)
    start = time.time()
    #lightgbm
    if args.lgb:
        print('Start training lgb...')
        with open("/mnt/ssd1/dataset/pickles/gbm_model.pkl", "rb") as handle:
            gbm = pickle.load(handle)
        start_lgb = time.time()
        gbm_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        print("time for 1 lgb prediction:",time.time() - start_lgb)
        gbm_1d_pred = np.argmax(gbm_pred, axis = 1)
        if args.save_pred:
            np.save(args.pred_dir, gbm_1d_pred)
        print('lightgbm_accuracy score: {}'.format(accuracy_score(y_val, np.argmax(gbm_pred, axis = 1))))

    if args.logreg:
        print("start loading logreg...")
        with open("/mnt/ssd1/dataset/pickles/logreg_model.pkl", 'rb') as f:
            log_clf = pickle.load(f)
        start_logreg = time.time()
        log_pred = log_clf.predict_proba(X_val)
        print("time for 1 logreg prediction:",time.time() - start_logreg)
        print('logreg pred', log_pred)
        log_not_scaled_pred = np.argmax(log_pred, axis = 1)
        print("time for 1 logreg prediction after argmax:",time.time() - start_logreg)
        print('log_not_scalerd',  log_not_scaled_pred)
        if args.save_pred:
            np.save(args.pred_dir, log_not_scaled_pred)
        print('logreg_accuracy score: {}'.format(accuracy_score(y_val, np.argmax(log_pred, axis = 1))))

    if args.catboost:
        print("start loading catboost")
        with open("/mnt/ssd1/dataset/pickles/catboost_model.pkl", 'rb') as f:
            model = pickle.load(f)
        start_cat = time.time()
        catboost_pred = model.predict(X_val)
        print("time for 1 catboost prediction:",time.time() - start_cat)
        if args.save_pred:
            np.save(args.pred_dir, catboost_pred)
        print('catboost_accuracy score: {}'.format(accuracy_score(y_val, catboost_pred)))

    if args.svm:
        print("start loading SVM")
        # SVM
        start_svm = time.time()
        with open("/mnt/ssd1/dataset/pickles/svm_scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        with open("/mnt/ssd1/dataset/pickles/svm_model.pkl", 'rb') as f:
            Lin_clf = pickle.load(f)
        start = time.time()
        svm_scaled_pred = Lin_clf.predict(X_val)
        print("time for 1 svm prediction:",time.time() - start_svm)
        if args.save_pred:
            np.save(args.pred_dir, svm_scaled_pred)
        print('SVM_accuracy score: {}'.format(accuracy_score(y_val, svm_scaled_pred)))

    # Voting
    if args.voting:
        def make_lst(pred):
           lst = []
           for i in pred:
               lst.append(int(i))
           return lst
        start_vote = time.time()
        catboost_pred = make_lst(catboost_pred)
      #  gbm_1d_pred = make_lst(gbm_1d_pred)
        log_not_scaled_pred = make_lst(log_not_scaled_pred)
        svm_scaled_pred = make_lst(svm_scaled_pred)

        def voting(pred_1, pred_2, pred_3):
            def most_common(val1, val2, val3):
                lst = [val1, val2, val3]
                return max(set(lst), key=lst.count)
            vote = []
            for i,j,k in zip(pred_1,pred_2,pred_3):
                vote.append(most_common(i,j,k))
            return vote
        start_last = time.time()
        vote = voting(svm_scaled_pred, log_not_scaled_pred, catboost_pred)
        print("time for only vote_last prediction:",time.time() - start_last)
        print("time for only vote prediction:",time.time() - start_vote)
        print("time for WHOLE VOTE prediction:",time.time() - start)
        if args.save_pred:
            np.save(args.pred_dir, vote)

        print("__________ALL_PREDICTIONS________")
        print('SVM_accuracy score: {}'.format(accuracy_score(y_val, svm_scaled_pred))) 
        print('logreg_accuracy score: {}'.format(accuracy_score(y_val, np.argmax(log_pred, axis = 1))))
#        print('lightgbm_accuracy score: {}'.format(accuracy_score(y_val, np.argmax(gbm_pred, axis = 1)))) didnt use in finel ensembling
        print('catboost_accuracy score: {}'.format(accuracy_score(y_val, catboost_pred)))
        print('voting_accuracy score: {}'.format(accuracy_score(y_val, vote)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="meta_classifier")
    # model
    parser.add_argument('--train_dir', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--val_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('-knn', action='store_true')
    parser.add_argument('-lgb', action='store_true')
    parser.add_argument('-logreg', action='store_true')
    parser.add_argument('-svm', action='store_true')
    parser.add_argument('-catboost', action='store_true')
    parser.add_argument('-voting', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_pred', action='store_true')
    parser.add_argument('--pred_dir', type=str, metavar='PATH', help = "path to the pred file")
    parser.add_argument('-multi_label', action='store_true')
    main(parser.parse_args())
