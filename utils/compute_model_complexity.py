import torch
import torch.nn as nn
from .MAdd.torchstat.statistics import stat


def compute_parameters(model):
    parameters = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parm = layer.weight
            parameters.append(parm.numel() / 1024 / 1024)
    return parameters


def compute_MACs(model):
    MACs = stat(model.cpu(), (3, 224, 224))#(3, 224, 224)
    MACs = [MAdd / 2 / 10**9 for MAdd in MACs]
    return MACs
