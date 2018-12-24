from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import pretrainedmodels
from torch import load
from .seresnet_blocks import se_resnet50_base, se_resnext50_32x4d_base
import torch

__all__ = ['SE_ResNet', 'se_resnet50', 'se_resnet101', 'se_resnet50_trained']


class SE_ResNet(nn.Module):
    __factory = {
        50: pretrainedmodels.models.senet.se_resnet50,
        101: pretrainedmodels.models.senet.se_resnet101,
    }

    def __init__(self, depth, pretrained=True, dropout = 0.5, n_classes = 1000, cut_at_pooling=False, features = False, last_stride = 2):
        super(SE_ResNet, self).__init__()

        #self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.base = se_resnet50_base(pretrained=None, last_stride=last_stride)
        self.stop_layer = SE_ResNet
        self.cut_at_pooling = cut_at_pooling
        self.features = features

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 2048)
            self.feat_bn = nn.BatchNorm1d(2048)
            #print('init by xavier')
            init.xavier_uniform_(self.feat.weight)
            #init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.out_planes, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)


    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'last_linear':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        x = torch.max(x, 0, keepdim = True)[0]
                
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

class SE_ResNetx4d(nn.Module):
    __factory = {
        50: pretrainedmodels.models.senet.se_resnext50_32x4d
        }

    def __init__(self, depth, pretrained=True, dropout = 0.5, n_classes = 1000, cut_at_pooling=False, features = False, last_stride = 2,
    aggr = None):
        super(SE_ResNetx4d, self).__init__()

        #self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.base = se_resnext50_32x4d_base(pretrained=None)
        self.stop_layer = SE_ResNetx4d
        self.cut_at_pooling = cut_at_pooling
        self.features = features
        self.aggr =  aggr

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 1024)
            self.feat_bn = nn.BatchNorm1d(1024)
            #print('init by xavier')
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


    def forward(self, x, bs = None, n_frames = None):
        for name, module in self.base._modules.items():
            if name == 'last_linear':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        features = x
        if self.aggr == 'max':
            x = torch.max(x, 0, keepdim = True)[0]
            features = x
            
        x = self.feat(x)
        x = self.feat_bn(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        if not self.training and self.features:
            return x, features
        return x

    

def se_resnet50(weights = None,  **kwargs):
    model = SE_ResNet(50, **kwargs)
    if weights is not None:
        if torch.cuda.is_available():
            state_dict = load(weights)['state_dict']
        else:
            state_dict = load(weights, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model


def se_resnext50_32x4d(weights = None, **kwargs):
    model = SE_ResNetx4d(50, **kwargs)
    if weights is not None:
        if torch.cuda.is_available():
            state_dict = load(weights)['state_dict']
        else:
            state_dict = load(weights, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model