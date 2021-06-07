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

# from dcp.utils.model_analyse import ModelAnalyse
# from dcp.utils.logger import get_logger
from insightface_v2.utils.model_analyse import ModelAnalyse
from insightface_v2.utils.logger import get_logger

import os


class ZQBottleneck_mobilefacenet(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expansion):
        super(ZQBottleneck_mobilefacenet, self).__init__()

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


class ZQMobilefacenet(nn.Module):

    def __init__(self, embedding_size=512, blocks=[4,8,16,4]):
        super(ZQMobilefacenet, self).__init__()

        Mobilefacenet_bottleneck_setting = [
            # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
            [1, 64, blocks[0], 1],
            [2, 128, 1, 2],
            [1, 128, blocks[1], 1],
            [2, 256, 1, 2],
            [1, 256, blocks[2], 1],
            [2, 512, 1, 2],
            [1, 512, blocks[3], 1]
        ]

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, groups=64, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.prelu2 = nn.PReLU(64)

        self.layers = self._make_layer(ZQBottleneck_mobilefacenet, Mobilefacenet_bottleneck_setting)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.prelu2 = nn.PReLU(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=7, groups=512, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn4 = nn.BatchNorm1d(embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        out = self.layers(out)
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.bn4(self.linear(out))
        return out


if __name__ == "__main__":

    model = ZQMobilefacenet(blocks=[4, 8, 16, 4], embedding_size=128)
    # model = ZQMobilefacenet(blocks=[4,8,16,4], embedding_size=512)   # Total params: 5,701,632; Params size (MB): 21.75
    # model = ZQMobilefacenet(blocks=[4, 8, 16, 8], embedding_size=512)  # Total params: 7,833,600; Params size (MB): 29.88
    # print(model.state_dict())
    # print("---------------------")
    # for key in model.state_dict().keys():
    #     print(key)
    print(model)
    summary(model, (3, 112, 112))

    save_path = './finetune-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path, "finetune-test")
    test_input = torch.randn(1, 3, 112, 112)
    model_analyse = ModelAnalyse(model, logger)
    model_analyse.flops_compute(test_input)
    model_analyse.params_count()
