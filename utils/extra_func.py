
import math
from collections import Counter
import numpy as np
import torch
import errno
import os 

import os.path as osp

def create_class_weight(tags, mu = 0.15):
    labels_dict = dict(Counter(tags).items())


    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()

    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    weights = np.zeros(len(class_weight.keys()))
    #print(weights)
    for i in range(len(weights)):
        weights[i] = class_weight[i]
    
    print(weights)

    return weights


def read_tags_and_create_weights(ann_file, mu = 0.15):
    labels = None
    with open(ann_file) as infile:
        labels = infile.readlines()
    infile.close()

    tags = []
    for line in labels:
        sp_line = line.split(",")
        tags.append(int(sp_line[1]))
    tags = np.asarray(tags)

    weights = create_class_weight(tags)
    weights = torch.from_numpy(weights).float()
    return weights

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location='cpu')
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))