#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 11:10
# @Author  : xiezheng
# @Site    : 
# @File    : demo.py


import torch


if __name__ == '__main__':
    # # 读取的模型类型为DataParallel, 现将其读取到CPU上
    # check_point_params = torch.load(
    #     "/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/check_point/checkpoint_48.tar")
    #
    # # 取出DataParallel中的一般module类型
    # epoch = check_point_params['epoch']
    # print(epoch)
    # acc = check_point_params['lfw_acc']
    # print(acc)
    # model = check_point_params['model'].module  # 模型类型为：torch.nn.DataParallel
    # print(model)
    # metric_fc = check_point_params['metric_fc'].module  # 模型类型为：torch.nn.DataParallel
    # print(metric_fc)
    #
    # check_point_params = {}
    # check_point_params["model"] = model.state_dict()
    # check_point_params["arcface"] = metric_fc.state_dict()
    # torch.save(check_point_params, './insightface_r34_with_arcface_epoch48.pth')

    for i in range(3, -1, -1):
        print(i)
