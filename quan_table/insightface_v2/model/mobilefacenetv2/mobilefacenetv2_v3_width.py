#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 16:00
# @Author  : xiezheng
# @Site    : 
# @File    : insightface_mobilefacenet.py


import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.nn import Parameter

from insightface_v2.utils.model_analyse import ModelAnalyse
from insightface_v2.utils.logger import get_logger
import os


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class Bottleneck_mobilefacenet(nn.Module):
    def __init__(self, in_planes, exp_planes, out_planes, stride):
        super(Bottleneck_mobilefacenet, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        # planes = in_planes * expansion
        planes = exp_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Mobilefacenetv2_v3_width(nn.Module):

    def __init__(self, embedding_size=512, width_mult=1.0):
        super(Mobilefacenetv2_v3_width, self).__init__()

        setting = [
            # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        self.inplanes = 64
        self.last_planes = 512
        self.last_planes = make_divisible(self.last_planes * width_mult) if width_mult > 1.0 else self.last_planes
        # print("self.last_planes={}".format(self.last_planes))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu1 = nn.PReLU(self.inplanes)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, groups=64, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.prelu2 = nn.PReLU(self.inplanes)

        self.layers = self._make_layer(Bottleneck_mobilefacenet, setting, width_mult=width_mult)

        self.conv3 = nn.Conv2d(self.inplanes, self.last_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.last_planes)
        self.prelu3 = nn.PReLU(self.last_planes)
        self.conv4 = nn.Conv2d(self.last_planes, self.last_planes, kernel_size=7, groups=self.last_planes,
                               stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.last_planes)
        self.linear = nn.Linear(self.last_planes, embedding_size)
        # self.bn5 = nn.BatchNorm1d(128, affine=False)
        self.bn5 = nn.BatchNorm1d(embedding_size, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, setting, width_mult):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                exp_planes = make_divisible(self.inplanes * t * width_mult)
                out_planes = make_divisible(c * width_mult)
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
        out = self.bn4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.bn5(self.linear(out))
        return out


if __name__ == "__main__":
    model = Mobilefacenetv2_v3_width(embedding_size=512, width_mult=1.315)
    # print(model.state_dict())
    # print("---------------------")
    # for key in model.state_dict().keys():
    #     print(key)
    print(model)
    # summary(model, (3, 112, 112))

    save_path = './finetune-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path, "finetune-test")
    test_input = torch.randn(1, 3, 112, 112)
    model_analyse = ModelAnalyse(model, logger)

    params_num = model_analyse.params_count()
    flops = model_analyse.flops_compute(test_input)
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count = count + 1

    print("\nmodel layers_num = {}".format(count))
    print("model size={} MB".format(params_num * 4 / 1024 / 1024))
    print("model flops={} M".format(sum(flops) / (10 ** 6)))
