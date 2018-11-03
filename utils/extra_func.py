
import math
from collections import Counter
import numpy as np
import torch

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
    weights = torch.from_numpy(weights).float()
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

    return create_class_weight(tags)