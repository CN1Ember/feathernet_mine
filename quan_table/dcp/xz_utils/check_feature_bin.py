#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/5 16:23
# @Author  : xiezheng
# @Site    : 
# @File    : check_feature_bin.py

import argparse
import os
import os.path
import struct
import sys
import numpy as np


feature_dim = 128
feature_ext = 1

def load_bin(path, fill=0.0):
    with open(path, 'rb') as f:
        bb = f.read(4 * 4)
        # print(len(bb))
        v = struct.unpack('4i', bb)
        # print(v[0])
        bb = f.read(v[0] * 4)
        v = struct.unpack("%df" % (v[0]), bb)
        feature = np.full((feature_dim + feature_ext,), fill, dtype=np.float32)
        # print("v=".format(v))
        feature[0:feature_dim] = v
        # feature = np.array( v, dtype=np.float32)
    # print(feature.shape)
    # print(np.linalg.norm(feature))
    return feature

bin_path = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/' \
           'pytorch_mobilefacenet_v1_model/pytorch_mobilefacenet_epoch34_feature-result/' \
           'fp_megaface_112x112_norm_features/870/8701948@N06/531397029_0.jpg_mobilefacenetPytorch_112x112.bin'

bin_path_2 = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_mobilefacenet_v1_model/' \
             'pytorch_mobilefacenet_epoch34_feature-result/fp_megaface_112x112_norm_features/870/' \
             '8701948@N06/531397029_1.jpg_mobilefacenetPytorch_112x112.bin'

# feature = load_bin(bin_path)
# print("feature={}".format(feature))

feature_2 = load_bin(bin_path_2)
print("feature_2={}".format(feature_2))


bin_path_3 = '/home/dataset/xz_datasets/Megaface/check/fp_megaface_112x112_norm_features' \
             '/870/8701948@N06/531397029_0.jpg_mobilefacenetPytorch_112x112.bin'

bin_path_4 = '/home/dataset/xz_datasets/Megaface/check/fp_megaface_112x112_norm_features' \
             '/870/8701948@N06/531397029_1.jpg_mobilefacenetPytorch_112x112.bin'

# feature_3 = load_bin(bin_path_3)
# print("feature_3={}".format(feature_3))

feature_4 = load_bin(bin_path_4)
print("feature_4={}".format(feature_4))