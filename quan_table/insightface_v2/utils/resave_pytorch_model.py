#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 14:44
# @Author  : xiezheng
# @Site    : 
# @File    : resave_pytorch_model.py


import torch


check_point_params = {}

model_path = '/home/xiezheng/program2019/Attention_transfer_face/new_style_mobilefacenet_lfw_99.55.pth'
arcface_path = '/home/xiezheng/program2019/Attention_transfer_face/mobilefacenet_epoch_34_best_lfw_0.996_checkpoint.pth'\

model_state = torch.load(model_path)
arcface_state = torch.load(arcface_path)


check_point_params['model'] = model_state['model']
check_point_params['metric_fc'] = arcface_state['metric_fc']

path = '/home/xiezheng/program2019/Attention_transfer_face/new_style_mobilefacenet_lfw_99.55.pth'
torch.save(check_point_params, path)