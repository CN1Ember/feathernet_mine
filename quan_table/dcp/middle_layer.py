#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 19:22
# @Author  : xiezheng
# @Site    : 
# @File    : middle_layer.py

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class View(nn.Module):
    """
    Reshape data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input {:d}".format(x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


# for insightface_r34
class Middle_layer(nn.Module):
    """
    define auxiliary classifier:
    BN->PRELU->AVGPOOLING->FC->Arc
    """

    def __init__(self, in_channels=512, feature_size=7):
        super(Middle_layer, self).__init__()

        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(in_channels*feature_size*feature_size, in_channels)
        self.bn3 = nn.BatchNorm1d(in_channels)

        # init params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        forward propagation
        """
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        return x


# for mobilefacenet_v1
# new
class Middle_layer_mobilefacenetv1(nn.Module):
    """
    define auxiliary classifier:
    conv1x1-> linear GDConv7x7 -> linear conv1x1 ->Arc
    """

    def __init__(self, in_channels=512, feature_size=7):
        super(Middle_layer_mobilefacenetv1, self).__init__()
        self.last_channels = 512

        self.conv3 = nn.Conv2d(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.last_channels)
        self.prelu3 = nn.PReLU(self.last_channels)
        self.conv4 = nn.Conv2d(self.last_channels, self.last_channels, kernel_size=feature_size,
                               groups=self.last_channels, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.last_channels)
        self.linear = nn.Linear(self.last_channels, in_channels)
        self.bn5 = nn.BatchNorm1d(in_channels, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        """
        forward propagation
        """
        out = self.prelu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.bn5(self.linear(out))
        return out

# old
# class Middle_layer_mobilefacenetv1(nn.Module):
#     """
#     define auxiliary classifier:
#     conv1x1-> linear GDConv7x7 -> linear conv1x1 ->Arc
#     """
#
#     def __init__(self, in_channels=512, feature_size=7):
#         super(Middle_layer_mobilefacenetv1, self).__init__()
#         self.last_channels = 512
#         feature_size = (feature_size - 7)+1
#
#         self.conv3 = nn.Conv2d(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.last_channels)
#         self.prelu3 = nn.PReLU(self.last_channels)
#         self.conv4 = nn.Conv2d(self.last_channels, self.last_channels, kernel_size=7,
#                                groups=self.last_channels, stride=1, padding=0, bias=False)
#         self.bn4 = nn.BatchNorm2d(self.last_channels)
#         self.linear = nn.Linear(self.last_channels*feature_size*feature_size, in_channels, bias=False)
#         self.bn5 = nn.BatchNorm1d(in_channels)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 if m.affine:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#
#     def forward(self, x):
#         """
#         forward propagation
#         """
#         out = self.prelu3(self.bn3(self.conv3(x)))
#         out = self.bn4(self.conv4(out))
#         out = out.view(out.size(0), -1)
#         out = self.bn5(self.linear(out))
#         return out


# class NormAwareReweighting(nn.Module):
#     def forward(self, x):
#         # channel dim
#         x_norm = torch.norm(x, dim=1)
#         # print("x_norm shape={}".format(x_norm.shape))
#         x_mean = x_norm.mean(2).mean(1)
#         # print("x_mean shape={}".format(x_mean.shape))
#         weight = x_mean.unsqueeze(1).unsqueeze(2) / x_norm
#         x = x * weight.unsqueeze(1)
#         return x

# for gnap block
# class Middle_layer_gnap(nn.Module):
#     """
#     define auxiliary classifier:
#     conv1x1-> linear GDConv7x7 -> linear conv1x1 ->Arc
#     """
#
#     def __init__(self, in_channels=128, feature_size=7, width_mult=1.0):
#         super(Middle_layer_gnap, self).__init__()
#
#         # linear_size = int(feature_size - 7 + 1)
#         last_channel = 512
#         planes = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#
#         self.conv3 = nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.prelu3 = nn.PReLU(planes)
#
#         filter_in = 512
#         if in_channels > filter_in:
#             self.additional_conv1 = nn.Conv2d(filter_in, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
#             self.additional_bn1 = nn.BatchNorm2d(in_channels)
#             self.additional_prelu1 = nn.PReLU(in_channels)
#             self.additional_module1 = nn.Sequential(self.additional_conv1, self.additional_bn1, self.additional_prelu1)
#             filter_in = in_channels
#
#         self.bn3 = nn.BatchNorm2d(planes, affine=False)
#         self.norm_aware_reweighting = NormAwareReweighting()
#         self.avg_pool = nn.AvgPool2d(feature_size, stride=1)
#         self.view = View()
#         self.bn4 = nn.BatchNorm1d(in_channels, affine=False)
#
#         if in_channels > filter_in:
#             self.fc1 = nn.Sequential(self.additional_module1, self.bn3, self.norm_aware_reweighting, self.avg_pool,
#                                      self.view)
#         else:
#             self.fc1 = nn.Sequential(self.bn3, self.norm_aware_reweighting, self.avg_pool, self.view)
#         if in_channels < filter_in:
#             self.additional_fc2 = nn.Linear(planes, in_channels)
#             self.fc1.add_module(str(len(self.fc1)), self.additional_fc2)
#
#         self.fc1.add_module(str(len(self.fc1)), self.bn4)
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 if m.affine:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 # nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         """
#         forward propagation
#         """
#         # out = self.prelu3(self.bn3(self.conv3(x)))
#         # out = self.bn4(self.conv4(out))
#         # out = out.view(out.size(0), -1)
#         # out = self.bn5(self.linear(out))
#
#         out = self.prelu3(self.bn3(self.conv3(x)))
#         out = self.fc1(out)
#
#         return out

