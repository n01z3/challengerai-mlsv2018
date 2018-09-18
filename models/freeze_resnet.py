from torchvision import models
from torch import nn


class SimpleResNet18(nn.Module):
    def __init__(self, n_classes=63, freeze=True):
        self.base_model = models.resnet18(pretrained=True)
        if freeze:
            self.freeze_base_model(self.base_model)
        self.base_model.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        return self.base_model.forward(x)

    def freeze_base_model(self, model):
        for param in model.parameters:
            param.requires_grad = False


def resnet18(**kwargs):
    return SimpleResNet18(**kwargs)

