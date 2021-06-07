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


class Bottleneck_mobilefacenetv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expansion):
        super(Bottleneck_mobilefacenetv2, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        planes = in_planes * expansion

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.0010000000474974513)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.0010000000474974513)
        self.prelu2 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=0.0010000000474974513)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.connect:
            return x + out
        else:
            return out

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

class Mobilefacenetv2_v2_depth(nn.Module):

    def __init__(self, embedding_size=512, blocks=[2,8,16,4]):
        super(Mobilefacenetv2_v2_depth, self).__init__()

        Mobilefacenetv2_bottleneck_setting = [
            # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
            [1, 64, blocks[0], 1],
            [2, 64, 1, 2],
            [2, 64, blocks[1], 1],
            [4, 128, 1, 2],
            [2, 128, blocks[2], 1],
            [4, 128, 1, 2],
            [2, 128, blocks[3], 1]
        ]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=0.0010000000474974513) #, track_running_stats=True)
        self.prelu1 = nn.PReLU(64)

        self.layers = self._make_layer(Bottleneck_mobilefacenetv2, Mobilefacenetv2_bottleneck_setting)

        self.conv2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512, eps=0.0010000000474974513)
        self.prelu2 = nn.PReLU(512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=7, groups=512, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(512, eps=0.0010000000474974513)
        # self.view = View()
        self.linear = nn.Linear(512, embedding_size)
        self.bn4 = nn.BatchNorm1d(embedding_size, affine=False, eps=2e-5)

        # self.fc1 = nn.Sequential(self.conv3, self.bn3, self.view, self.linear, self.bn4)
        # elif fc_type == 'gnap':
        #     print("gnap_type !!!")
        #     filter_in = 512
        #     if embedding_size > filter_in:
        #         self.additional_conv1 = nn.Conv2d(filter_in, embedding_size, kernel_size=1, stride=1, padding=0, bias=False)
        #         self.additional_bn1 = nn.BatchNorm2d(embedding_size)
        #         self.additional_prelu1 = nn.PReLU(embedding_size)
        #         self.additional_module1 = nn.Sequential(self.additional_conv1, self.additional_bn1, self.additional_prelu1)
        #         filter_in = embedding_size
        #
        #     self.bn3 = nn.BatchNorm2d(filter_in, affine=False)
        #     self.norm_aware_reweighting = NormAwareReweighting()
        #     self.avg_pool = nn.AvgPool2d(7, stride=1)
        #     self.view = View()
        #     self.bn4 = nn.BatchNorm1d(embedding_size, affine=False)
        #
        #     if embedding_size > filter_in:
        #         self.fc1 = nn.Sequential(self.additional_module1, self.bn3, self.norm_aware_reweighting, self.avg_pool, self.view)
        #     else:
        #         self.fc1 = nn.Sequential(self.bn3, self.norm_aware_reweighting, self.avg_pool, self.view)
        #     if embedding_size < filter_in:
        #         self.additional_fc2 = nn.Linear(filter_in, embedding_size)
        #         self.fc1.add_module(str(len(self.fc1)), self.additional_fc2)
        #
        #     self.fc1.add_module(str(len(self.fc1)), self.bn4)
        #
        # else:
        #     print('Not support fc type: {}'.format(fc_type))
        #     assert False

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
        out = self.layers(out)
        out = self.prelu2(self.bn2(self.conv2(out)))
        # out = self.fc1(out)
        out = self.bn3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.bn4(self.linear(out))
        return out


if __name__ == "__main__":

    # model = Mobilefacenetv2_v2_depth(blocks=[3, 8, 16, 5], embedding_size=512)  # FLOPs: 933M
    # model = Mobilefacenetv2_v2_depth(blocks=[2, 8, 16, 4], embedding_size=256)
    model = Mobilefacenetv2_v2_depth(blocks=[4, 8, 16, 8], embedding_size=256)

    # model = Mobilefacenetv2(blocks=[2,8,16,6], embedding_size=512)  # FLOPs: 946M
    # model = Mobilefacenetv2(blocks=[3, 8, 16, 6], embedding_size=512)  # FLOPs: 1001M
    # model = Mobilefacenetv2(blocks=[3, 8, 16, 5], embedding_size=512)  # FLOPs: 994M, Params size (MB): 7.31

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

    params_num = model_analyse.params_count()
    flops = model_analyse.flops_compute(test_input)
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count = count + 1

    print("\nmodel layers_num = {}".format(count))
    print("model size={} MB".format(params_num * 4 / 1024 / 1024))
    print("model flops={} M".format(sum(flops) / (10 ** 6)))