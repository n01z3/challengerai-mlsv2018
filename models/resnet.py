from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torch import load

__all__ = ['ResNet', 'resnet50', 'resnet101']


class ResNet(nn.Module):
    __factory = {
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
    }

    def __init__(self, depth, stop_layer = 'fc', pretrained=True):
        super(ResNet, self).__init__()

        self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.stop_layer = stop_layer

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == self.stop_layer:
                break
            x = module(x)

        return x

def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)