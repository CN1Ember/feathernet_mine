#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 16:39
# @Author  : xiezheng
# @Site    :
# @File    : insightface_resnet.py

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
from torchsummary import summary


__all__ = ['ResNet', 'LResNet18E_IR', 'LResNet34E_IR']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(channel // reduction),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, eps=2e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
        if stride != 1:
            self.downsample = True
            self.downsample_conv = nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False)
            self.downsample_bn = nn.BatchNorm2d(planes, eps=2e-5)
        else:
            self.downsample = False
        # self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.downsample_conv(x)
            residual = self.downsample_bn(residual)

        out += residual

        return out

# LResNetxxE-IR or SE-LResNetxxE-IR
class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=False):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5)
        self.prelu = nn.PReLU(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=2e-5)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512, eps=2e-5, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # downsample = None
        # if stride != 1:
        #     downsample_conv = nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False)
        #     downsample_bn = nn.BatchNorm2d(planes * block.expansion, eps=2e-5)
        #     downsample = nn.Sequential(
        #         downsample_conv,
        #         downsample_bn,
        #     )

        layers = []
        layers.append(block(self.inplanes, planes, stride, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        return x

# LResNet18E-IR
def LResNet18E_IR():
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=False)
    return model

# LResNet34E-IR
def LResNet34E_IR():
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False)
    return model

def LResNet50E_IR():
    model = ResNet(IRBlock, [3, 4, 14, 3], use_se=False)
    return model

def LResNet100E_IR():
    model = ResNet(IRBlock, [3, 13, 30, 3], use_se=False)
    return model

def LResNet74E_IR():
    model = ResNet(IRBlock, [3, 6, 24, 3], use_se=False)
    return model


if __name__ == "__main__":
    # model = LResNet34E_IR()   # Total params: 31,807,969
    model = LResNet100E_IR()
    print(model)
    # print("---------------------")
    # for key in model.state_dict().keys():
    #     print(key)
    summary(model, (3, 112, 112))
