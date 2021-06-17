#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/11 22:21
# @Author  : xiezheng
# @Site    : 
# @File    : arcface.py

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
from torchsummary import summary
import os



class arcface(nn.Module):
    r"""Implement of large margin arc distance: : cos(theta + m)
        """
    def __init__(self, num_classes, emb_size):
        super(arcface, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = 0.5
        self.s = 64

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)

        if not self.training:
            return cosine

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output