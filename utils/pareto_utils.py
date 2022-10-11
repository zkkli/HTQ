import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils import *


layer_a = []
def viz(module, input):
    layer_a.append(input[0].mean(axis=0,keepdim=True))
    #print(input[0].shape)
    #print('***********************************')

def get_layer_a(model):
    train_loader = getTrainData(dataset='imagenet', path='/dataset/imagenet/')
    img = next(iter(train_loader))[0]  # gain a input
    img = img.cuda()
    names = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.register_forward_pre_hook(viz)
            names.append(name)
    with torch.no_grad():
        model(img)

    return layer_a


def compute_perturbation(x, num_bits, metric=1):
    with torch.no_grad():
        n = 2 ** num_bits - 1
        saturation_max = x.max()
        saturation_min = x.min()
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        zero_point = -saturation_min / scale
        zero_point = zero_point.round()

        if len(x.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(x.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)

        q_x = torch.round(1. / scale * x + zero_point)
        q_x = (q_x - zero_point) * scale

        perturbation = torch.norm((x - q_x)) * metric #/sqrt(x.numel())
        perturbation = perturbation.cpu().detach().numpy()
        return perturbation
