from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import pretrainedmodels
from torch import load

__all__ = ['SE_ResNet', 'se_resnet50', 'se_resnet101', 'se_resnet50_trained']


class SE_ResNet(nn.Module):
    __factory = {
        50: pretrainedmodels.models.senet.se_resnet50,
        101: pretrainedmodels.models.senet.se_resnet101,
    }

    def __init__(self, depth, pretrained=True, dropout = 0, n_classes = 1000, cut_at_pooling=False, features = False):
        super(SE_ResNet, self).__init__()

        self.base = SE_ResNet.__factory[depth](pretrained='imagenet')
        self.stop_layer = SE_ResNet
        self.cut_at_pooling = cut_at_pooling
        self.features = features

        if not self.cut_at_pooling:
            self.dropout = dropout
            self.num_classes = n_classes
            
            self.out_planes = self.base.last_linear.in_features

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.out_planes, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)


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
        elif not self.training and not self.features:
            x = self.classifier(x)
        else:
            if self.dropout > 0:
                x = self.drop(x)
            if self.num_classes > 0:
                x = self.classifier(x)
        return x

def se_resnet50(weigths = None, gpu = True, **kwargs):
    model = SE_ResNet(50, **kwargs)
    if weigths is not None:
        if gpu:
            state_dict = load(weigths)['state_dict']
        else:
            state_dict = load(weigths, map_location= 'cpu')['state_dict']
        model.load_state_dict(state_dict)
    return model


def se_resnet101(**kwargs):
    return SE_ResNet(101, **kwargs)

def se_resnet50_trained(**kwargs):
    model = SE_ResNet(50,n_classes=63,**kwargs)
    state_dict = load("/mnt/ssd1/easygold/challengerai-mlsv2018/logs/un_baseline_2018-10-10_15-47-12/checkpoint.pth.tar")["state_dict"]
    model.load_state_dict(state_dict)
    return model