import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import Parameter

import numpy as np
import os
import matplotlib.pyplot as plt


def quantization_on_weights(x, k):
    if k == 1:
        return SignMeanRoundFunction.apply(x)

    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5  # normalization weights
    n = torch.pow(2, k) - 1
    return 2 * RoundFunction.apply(x, n) - 1

def quantization_on_activations(x, k):
    n = torch.pow(2, k) - 1
    x = torch.clamp(x, 0, 1)  # normalization activations
    return RoundFunction.apply(x, n)

# clip_val means alpha in the PACT paper
# def pact_quantization_on_activations(x, k, clip_val=2):
#     n = (torch.pow(2, k) - 1) / clip_val
#     x = torch.clamp(x, 0, clip_val)  # normalization activations
#     # print("clip_val=", clip_val)
#     return RoundFunction.apply(x, n)

def normalization_on_activations(x):
    return torch.clamp(x, 0, 1)


class SignMeanRoundFunction(Function):

    @staticmethod
    def forward(ctx, x):
        # dorefa-Net : b*c*h*w
        # E = torch.mean(torch.abs(x))

        # BWN : c*h*w
        avg = nn.AdaptiveAvgPool3d(1)
        E = avg(torch.abs(x))
        return torch.where(x == 0, torch.ones_like(x), torch.sign(x / E)) * E

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        quantized_weight = quantization_on_weights(self.weight, self.k)
        if self.bias is not None:   # self.bias is None
            quantized_bias = self.bias
        else:
            quantized_bias = None
        return F.conv2d(input, quantized_weight, quantized_bias, self.stride, self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """
    def __init__(self, in_features, out_features, k, bias=True):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        quantized_weight = quantization_on_weights(self.weight, self.k)
        if self.bias is not None:  # self.bias is None
            quantized_bias = self.bias
        else:
            quantized_bias = None
        return F.linear(input, quantized_weight, quantized_bias)


class QReLU(nn.ReLU):
    """
    custom ReLU for quantization
    """
    def __init__(self, k, inplace=False):
        super(QReLU, self).__init__(inplace=inplace)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        out = quantization_on_activations(input, self.k)
        return out


class QMaskConv2d(nn.Conv2d):
    """
    Custom convolutional layers for channel pruning.
    Here we use mask to indicate the selected/unselected channels. 1 denotes selected and 0 denotes unselected.
    We store the original weight in weight and the pruned weight in pruned_weight.
    Before channel selection, the pruned weight is initialized to zero.
    In greedy algorithm for channel selection, we initialize the pruned weight with the original weight and
    then optimized pruned weight w.r.t. the selected channels by minimizing the problem (9).
    """

    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1, padding=0, bias=True):
        super(QMaskConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.register_buffer('k', torch.FloatTensor([k]))
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))   # self.weight.data.size(1) means input channel
        self.pruned_weight = Parameter(self.weight.clone())

    def forward(self, input):
        self.pruned_weight = quantization_on_weights(self.pruned_weight, self.k)
        return F.conv2d(input, self.pruned_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

