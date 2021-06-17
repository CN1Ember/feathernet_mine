#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 16:00
# @Author  : xiezheng
# @Site    : 
# @File    : insightface_mobilefacenet.py


import math
import torch
from torch import nn
from torchsummary import summary

# from insightface.models.mobilefacenetv2_width import Bottleneck_mobilefacenetv2
# from insightface.utils.model_analyse import ModelAnalyse


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class Bottleneck_mobilefacenetv2(nn.Module):
    def __init__(self, in_planes, exp_planes, out_planes, stride):
        super(Bottleneck_mobilefacenetv2, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        # planes = in_planes * expansion
        planes = exp_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.prelu1 = nn.ReLU(inplace = True)
        self.prelu1 = nn.PReLU(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.prelu2 = nn.ReLU(inplace = True)
        self.prelu2 = nn.PReLU(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.connect:
            return x + out
        else:
            return out


class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        kernel_size = w
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        x = maxpool(x)
        return x


class Mobilefacenetv2_width_wm(nn.Module):
    def __init__(self, embedding_size=512, mask_train=False, pruning_rate=0):
        super(Mobilefacenetv2_width_wm, self).__init__()
        print('mask_train',mask_train)
        block_setting = [
            # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        self.inplanes = 64
        self.last_planes = 512
        if not mask_train:
            self.last_planes = int(self.last_planes - math.floor(self.last_planes * pruning_rate))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.prelu1 = nn.ReLU(inplace = True)
        self.prelu1 = nn.PReLU(self.inplanes)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, groups=64, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        # self.prelu2 = nn.ReLU(inplace = True)
        self.prelu2 = nn.PReLU(self.inplanes)

        self.layers = self._make_layer(Bottleneck_mobilefacenetv2, block_setting, mask_train, pruning_rate=pruning_rate)

        # without pruning fc
        self.last_planes = 512
        self.conv3 = nn.Conv2d(self.inplanes, self.last_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.last_planes)
        # self.prelu3 = nn.ReLU(inplace = True)
        self.prelu3 = nn.PReLU(self.last_planes)
        if mask_train:
            self.conv4 = nn.Conv2d(self.last_planes, self.last_planes, kernel_size=(4, 7), groups=self.last_planes, stride=1,
                                   padding=0, bias=False)
        else:
            self.conv4 = nn.Conv2d(self.last_planes, self.last_planes, kernel_size=7, groups=self.last_planes, stride=1,
                                   padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.last_planes)
        self.linear = nn.Linear(self.last_planes, embedding_size)
        self.bn5 = nn.BatchNorm1d(embedding_size, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, setting, mask_train, pruning_rate):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                exp_planes = self.inplanes * t
                if not mask_train:
                    exp_planes = int(exp_planes - math.floor(exp_planes * pruning_rate))
                out_planes = c
                print(exp_planes,self.inplanes,t)
                if i == 0:
                    layers.append(block(self.inplanes, exp_planes, out_planes, s))
                else:
                    layers.append(block(self.inplanes, exp_planes, out_planes, 1))
                self.inplanes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.layers(out)
        out = self.prelu3(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out = self.bn4(out)
        out = out.view(out.size(0), -1)
        out = self.bn5(self.linear(out))
        return out


if __name__ == "__main__":
     model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
     # print(model)
     summary(model, (3, 56, 112), device='cpu')
#     # only widen: model layers_num = 49 ; model size=9.73 MB; model flops=995.61 M, 995605304.00
#     # model = Mobilefacenetv2_width_wm(embedding_size=512, pruning_rate=0.75)
#
#     print(model)
#     # summary(model, (3, 112, 112))
#
#     test_input = torch.randn(1, 3, 112, 112)
#     model_analyse = ModelAnalyse(model)
#     params_num = model_analyse.params_count()
#     flops = model_analyse.flops_compute(test_input)
#     count = 0
#     block_count = 0
#     for module in model.modules():
#         if isinstance(module, nn.Conv2d):
#             count = count + 1
#         if isinstance(module, Bottleneck_mobilefacenetv2):
#             block_count = block_count + 1
#
#     print("\nmodel layers_num = {}".format(count))
#     print("model block_count = {}".format(block_count))
#     print("model size = {:.2f} MB".format(params_num * 4 / 1024 / 1024))
#     print("model flops = {:.2f} MFLOPs".format(sum(flops) / 1e6))
