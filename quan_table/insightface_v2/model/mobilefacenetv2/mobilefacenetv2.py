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


class Bottleneck_mobilefacenet(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expansion):
        super(Bottleneck_mobilefacenet, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        planes = in_planes * expansion

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


class Mobilefacenetv2(nn.Module):

    Mobilefacenet_bottleneck_setting = [
        # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
        [2, 64,  5, 2],
        [4, 128, 1, 2],
        [2, 128, 6, 1],
        [4, 128, 1, 2],
        [2, 128, 2, 1]
    ]

    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting, embedding_size=512):
        super(Mobilefacenetv2, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, groups=64, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)

        self.layers = self._make_layer(Bottleneck_mobilefacenet, bottleneck_setting)


        self.conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.prelu3 = nn.PReLU(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=7, groups=512, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, embedding_size)
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

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
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
    model = Mobilefacenetv2(embedding_size=512)
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