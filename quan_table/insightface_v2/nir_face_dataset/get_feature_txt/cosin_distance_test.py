#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-12 下午8:40
# @Author  : xiezheng
# @Site    : 
# @File    : cosin_distance_test.py
import numpy as np
import sklearn
from sklearn import preprocessing
import os


def get_cosin_distance(real_txt_path, fake_txt_path):
    real_feature = np.loadtxt(real_txt_path, dtype=np.float32)

    fake_feature = np.loadtxt(fake_txt_path, dtype=np.float32)
    # fake_feature = np.loadtxt(fake_txt_path, dtype=np.float32, delimiter=',')
    # print(fake_feature)
    # assert False

    # print("real_feature shape: ", real_feature.shape)
    # print("fake_feature shape: ", fake_feature.shape)
    real_feature = sklearn.preprocessing.normalize(real_feature.reshape(1, 128))
    fake_feature = sklearn.preprocessing.normalize(fake_feature.reshape(1, 128))
    # print("real_feature shape: ", real_feature.shape)
    # print("fake_feature shape: ", fake_feature.shape)
    cosin_distance = np.dot(real_feature, fake_feature.T)
    return cosin_distance[0][0]



path = "D:\\2019Programs\jidian2019\mobilefacenet\误识\可见光量化模型_误识_20191204\\481"
part1_1 = os.path.join(path, "rec.bmp.rgb.txt")
part1_2 = os.path.join(path, "reg.bmp.rgb.txt")
cossin_distance1 = get_cosin_distance(part1_1, part1_2)
print("cosin-distance = ", cossin_distance1)

part1_1 = os.path.join(path, "0.jpg.rgb.txt")
part1_2 = os.path.join(path, "1.jpg.rgb.txt")
cossin_distance1 = get_cosin_distance(part1_1, part1_2)
print("cosin-distance = ", cossin_distance1)

part1_1 = os.path.join(path, "1.jpg.rgb.txt")
part1_2 = os.path.join(path, "reg.bmp.rgb.txt")
cossin_distance1 = get_cosin_distance(part1_1, part1_2)
print("cosin-distance = ", cossin_distance1)


# path = "./ku"
# part1_1 = os.path.join('./', "a1.bmp.1420.txt")
# for i in range(1,13):
#     img_name = "{}.txt".format(i)
#     part1_2 = os.path.join(path, img_name)
#     # print('i={}, part1_1={}, part1_2={}'.format(i, part1_1, part1_2))
#     cossin_distance1 = get_cosin_distance(part1_1, part1_2)
#     print("{}.txt, cosin-distance={}".format(i, cossin_distance1))
