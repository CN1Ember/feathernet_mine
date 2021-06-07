#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 10:28
# @Author  : xiezheng
# @Site    : 
# @File    : mobielfacenet_convert.py

import torch
import torch.nn as nn

# from dcp.models.insightface_mobilefacenet import MobileFaceNet
from dcp.models.mobilefacenet import Mobilefacenet

def main():
    # origin_model = MobileFaceNet()
    # 241 server
    origin_checkpoint_param = torch.load("/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/mobilefacenet_v1_res1-4-6-2/check_point/checkpoint_34.pth")
    origin_state_dict = origin_checkpoint_param['model']
    arcface_state_dict = origin_checkpoint_param['metric_fc']

    new_model = Mobilefacenet()
    new_state_dict = new_model.state_dict()

    # tmp_state_dict = new_state_dict

    key_list = []
    for key in new_state_dict.keys():
        key_list.append(key)

    num = 0
    for key, value in origin_state_dict.items():
        print("{} -- >  {}".format(key, [key_list[num]]))
        new_state_dict[key_list[num]] = value
        num += 1

    new_model.load_state_dict(new_state_dict)
    print("success !!!")

    checkpoint_param = {}
    checkpoint_param["model"] = new_state_dict
    checkpoint_param["arcface"] = arcface_state_dict
    torch.save(checkpoint_param, "/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/new_style_mobilefacenet_lfw_99.55.pth")
    print("program done!")

if __name__ == '__main__':
    main()