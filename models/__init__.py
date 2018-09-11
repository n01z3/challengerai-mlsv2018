from __future__ import absolute_import

from .resnet import *

__factory = {
    'resnet50': resnet50,
    'resnet101': resnet101,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
