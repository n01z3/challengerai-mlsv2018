from __future__ import absolute_import

from .resnet import *
from .seresnet import *
from .inceptionv4 import *

__factory = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'se_resnet50': se_resnet50,
    'se_resnet101': se_resnet101,
    'se_resnet_cls50': se_resnet_cls_50,
    'se_resnet50_trained': se_resnet50_trained,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'inceptionv4': inceptionv4,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
