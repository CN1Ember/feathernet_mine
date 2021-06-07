#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/10 21:45
# @Author  : xiezheng
# @Site    : 
# @File    : demo.py


# import torch.nn as nn
# from torchvision.models import resnet18
# model = resnet18()
#
#
#
# def replace_layer_by_unique_name(module, unique_name, layer):
#     unique_names = unique_name.split(".")
#     if len(unique_names) == 1:
#         module._modules[unique_names[0]] = layer
#     else:
#         replace_layer_by_unique_name(
#             module._modules[unique_names[0]],
#             ".".join(unique_names[1:]),
#             layer
#         )
#
#
# def get_layer_by_unique_name(module, unique_name):
#     unique_names = unique_name.split(".")
#     if len(unique_names) == 1:
#         return module._modules[unique_names[0]]
#     else:
#         return get_layer_by_unique_name(
#             module._modules[unique_names[0]],
#             ".".join(unique_names[1:]),
#         )
#
#
#
# print(model)
# print("-"*50)
#
# for name,module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         temp_conv = nn.Conv2d(10,10,3)
#         replace_layer_by_unique_name(model,name,temp_conv)
#
# print('After replace:')
# print("-"*50)
# print(model)


from easydict import EasyDict as edict


activation_value_list = edict()

for line in open('./mobilefacenet_p0.5_activation.table'):
    line = line.strip()
    key, value = line.split(' ')

    # remove 'net.'
    if "net" in key:
        key = key.replace('net.', '')
    activation_value_list[key] = value

for key in activation_value_list:
    print(key, activation_value_list[key])