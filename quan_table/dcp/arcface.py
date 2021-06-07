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

class Arcface(nn.Module):
    """
    define auxiliary classifier:
    BN->PRELU->AVGPOOLING->FC->Arc
    """

    def __init__(self, in_channels, num_classes, easy_margin=False, margin_m=0.5, margin_s=64.0):
        super(Arcface, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


    def forward(self, x, label):
        """
        forward propagation
        """

        x = F.normalize(x)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m   # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size()).cuda()
        one_hot = x.new_zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output