from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys
from torch.nn import functional as F
from torch.nn import init

from .inceptionv4_blocks import inceptionv4_base

class InceptionV4(nn.Module):

    def __init__(self, n_classes=1001, pretrained = 'imagenet', cut_at_pooling=False, features = False, dropout = 0.5):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.base = inceptionv4_base(pretrained = pretrained, num_classes=1000)
        self.cut_at_pooling = cut_at_pooling
        self.features = features
        #self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        #self.last_linear = nn.Linear(1536, num_classes)
        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 1024)
            self.feat_bn = nn.BatchNorm1d(1024)
            print('init by xavier')
            init.xavier_uniform_(self.feat.weight)
            #init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(1024, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

    def forward(self, input):
        x = self.base(input)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
       
        if not self.training and self.features:
            return x
        else:
            x = self.feat(x)
            x = self.feat_bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = self.drop(x)
            if self.num_classes > 0:
                x = self.classifier(x)
        return x

def inceptionv4(n_classes=1000, pretrained='imagenet', last_stride = 2):
    
    model = InceptionV4(n_classes=n_classes, pretrained = pretrained)
    return model