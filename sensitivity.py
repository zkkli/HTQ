import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import types


def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def snip(model, train_loader):
    inputs, targets = next(iter(train_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(model)
    net = net.cuda()

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    scores = []
    names = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            score = torch.mean(torch.abs(layer.weight_mask.grad))
            scores.append(score.item())
            names.append(name)

    return scores, names
