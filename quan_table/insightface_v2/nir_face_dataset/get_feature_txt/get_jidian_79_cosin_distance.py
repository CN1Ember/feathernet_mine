#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-20 下午5:16
# @Author  : xiezheng
# @Site    : 
# @File    : get_cosin_distance_face.py

import numpy as np
import sklearn
from sklearn import preprocessing
import os


def get_cosin_distance(txt_path1, txt_path2, feature_size):
    feature1 = np.loadtxt(txt_path1, dtype=np.float32)
    feature2 = np.loadtxt(txt_path2, dtype=np.float32)
    # print("real_feature shape: ", feature1.shape)
    # print("fake_feature shape: ", feature2.shape)
    feature1 = sklearn.preprocessing.normalize(feature1.reshape(feature_size))
    feature2 = sklearn.preprocessing.normalize(feature2.reshape(feature_size))
    cosin_distance = np.dot(feature1, feature2.T)
    return cosin_distance[0][0]

# nir: all_update
# path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/jidian_face_112x112_79_align_clear_nir_pytorch_mobilefacenet_gpu"

# nir: last1_update
path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/nir_face_part_update_pytorch_mobilefacenet/jidian_face_112x112_79_align_clear_last1_lr0.01"

# nir: last2_update
# path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/nir_face_part_update_pytorch_mobilefacenet/jidian_face_112x112_79_align_clear_last2_lr0.01"

# jidian_79_nir: algo_detection_crop_fill_align_feature
# path = "/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop-fill-aligned/jidian_face_79_algo_crop_fill_aligned_feature"
# ext = ".BMP.txt"

print("path={}".format(path))
ext = ".jpg.txt"
feature_size = (1, 128)

image_list = [["1/1 (1)", "1/1 (3)"],
              ["2/2 (1)", "2/2 (4)"],
              ["3/3 (3)", "3/3 (4)"],
              ["4/4 (3)", "4/4 (4)"],
              ["5/5 (1)", "5/5 (9)"],
              ["6/6 (3)", "6/6 (4)"],
              ["7/7 (3)", "7/7 (4)"],
              ["8/8 (3)", "8/8 (7)"],
              ["9/9 (2)", "9/9 (3)"],
              ["10/10 (3)", "10/10 (4)"],
              ["11/11 (4)", "11/11 (6)"],
              ["12/12 (1)", "12/12 (6)"],
              ]
for i in range(len(image_list)):
    print("i=", str(i))
    image1_path = os.path.join(path, image_list[i][0] + ext)
    image2_path = os.path.join(path, image_list[i][1] + ext)
    cosin_distance = get_cosin_distance(image1_path, image2_path, feature_size)
    print(image_list[i][0], "----", image_list[i][1], "--->", str(cosin_distance))

