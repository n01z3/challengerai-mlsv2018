from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import pretrainedmodels
from torch import load
from .seresnet_blocks import se_resnet50_base, se_resnext50_32x4d_base

__all__ = ['SE_ResNet', 'se_resnet50', 'se_resnet101', 'se_resnet50_trained', 'se_resnet_cls_50', 'se_resnext50_32x4d']


class SE_ResNet(nn.Module):
    __factory = {
        50: pretrainedmodels.models.senet.se_resnet50,
        101: pretrainedmodels.models.senet.se_resnet101,
    }

    def __init__(self, depth, pretrained=True, dropout = 0.5, n_classes = 1000, cut_at_pooling=False, features = False, last_stride = 2):
        super(SE_ResNet, self).__init__()

        #self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.base = se_resnet50_base(pretrained='imagenet', last_stride=last_stride)
        self.stop_layer = SE_ResNet
        self.cut_at_pooling = cut_at_pooling
        self.features = features

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 2048)
            self.feat_bn = nn.BatchNorm1d(2048)
            print('init by xavier')
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

    def __init__(self, depth, pretrained=True, dropout = 0.5, n_classes = 1000, cut_at_pooling=False, features = False, last_stride = 2):
        super(SE_ResNetx4d, self).__init__()

        #self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.base = se_resnext50_32x4d_base(pretrained='imagenet')
        self.stop_layer = SE_ResNetx4d
        self.cut_at_pooling = cut_at_pooling
        self.features = features

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 2048)
            self.feat_bn = nn.BatchNorm1d(2048)
            print('init by xavier')
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
class SE_ResNet_cls(nn.Module):
    __factory = {
        50: pretrainedmodels.models.senet.se_resnet50,
        101: pretrainedmodels.models.senet.se_resnet101,
    }

    def __init__(self, depth, pretrained=True, dropout = 0.5, n_classes = 1000, cut_at_pooling=False, features = False, last_stride = 2, 
                n_class_pred = True):
        super(SE_ResNet_cls, self).__init__()

        #self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.base = se_resnet50_base(pretrained='imagenet', last_stride=last_stride)
        self.stop_layer = SE_ResNet_cls
        self.cut_at_pooling = cut_at_pooling
        self.features = features
        self.n_class_pred = n_class_pred

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features
            self.feat = nn.Linear(self.out_planes, 2048)
            self.feat_bn = nn.BatchNorm1d(2048)
            print('init by xavier')
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
        if n_class_pred:
            self.cls_feat_1 = nn.Linear(self.out_planes, 512)
            self.cls_bn_1 = nn.BatchNorm1d(512)
            
            self.cls_feat_2 = nn.Linear(512, 256)
            self.cls_bn_2 = nn.BatchNorm1d(256)

            self.cls_feat_3 = nn.Linear(256, 1)
            if self.dropout > 0:
                self.drop_cls_1 = nn.Dropout(self.dropout)
                self.drop_cls_2 = nn.Dropout(self.dropout)

            init.xavier_uniform_(self.cls_feat_1.weight)
            init.xavier_uniform_(self.cls_feat_2.weight)
            init.constant_(self.cls_feat_1.bias, 0)
            init.constant_(self.cls_feat_2.bias, 0)
            init.normal_(self.cls_feat_3.weight, std = 0.001)
            init.constant_(self.cls_feat_3.bias, 0)
            init.constant_(self.cls_bn_1.weight, 1)
            init.constant_(self.cls_bn_1.bias, 0)

            init.constant_(self.cls_bn_2.weight, 1)
            init.constant_(self.cls_bn_2.bias, 0)


    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'last_linear':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
                
        if not self.training and self.features and not self.n_class_pred:
            return x
        elif not self.n_class_pred:
            x = self.main_cls_path(x)
        else:
            main_cls = self.main_cls_path(x)
            tags_amount = self.class_am_predict(x)
            return main_cls, tags_amount

        return x
    
    def main_cls_path(self, x):
        x = self.feat(x)
        x = self.feat_bn(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def class_am_predict(self, x):
        x = self.cls_feat_1(x)
        x = self.cls_bn_1(x)

        x = F.relu(x)
        if self.dropout > 0:
            x = self.drop_cls_1(x)
        x = self.cls_feat_2(x)
        x = self.cls_bn_2(x)

        x = F.relu(x)
        if self.dropout > 0:
            x = self.drop_cls_2(x)
        x = self.cls_feat_3(x)
        return x

def se_resnet50(weights = None, gpu = True, **kwargs):
    model = SE_ResNet(50, **kwargs)
    if weights is not None:
        if gpu:
            state_dict = load(weights)['state_dict']
        else:
            state_dict = load(weights, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model

def se_resnext50_32x4d(weights = None, gpu = True, **kwargs):
    model = SE_ResNetx4d(50, **kwargs)
    if weights is not None:
        if gpu:
            state_dict = load(weights)['state_dict']
        else:
            state_dict = load(weights, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model

def se_resnet_cls_50(weights = None, gpu = True, **kwargs):
    model = SE_ResNet_cls(50, **kwargs)
    if weights is not None:
        if gpu:
            state_dict = load(weights)['state_dict']
        else:
            state_dict = load(weights, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model

def se_resnet101(**kwargs):
    return SE_ResNet(101, **kwargs)

def se_resnet50_trained(**kwargs):
    model = SE_ResNet(50,n_classes=63,**kwargs)
    state_dict = load("/mnt/ssd1/easygold/challengerai-mlsv2018/logs/un_baseline_2018-10-10_15-47-12/checkpoint.pth.tar")["state_dict"]
    model.load_state_dict(state_dict)
    return model