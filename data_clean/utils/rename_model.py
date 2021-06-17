#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/6 20:55
# @Author  : xiezheng
# @Site    : 
# @File    : rename_model.py

import torch

checkpoint_path = '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/' \
                  'insightface_r34_with_arcface_epoch48.pth'

params = torch.load(checkpoint_path)
model_state = params["model"]
metric_fc_state = params["arcface"]

check_point_params = {}
check_point_params["model"] = model_state
check_point_params["metric_fc"] = metric_fc_state

torch.save(check_point_params, '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/'
                               'insightface_r34_with_arcface_v2_epoch48.pth')