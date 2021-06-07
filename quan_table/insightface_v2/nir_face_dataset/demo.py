#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/14 23:49
# @Author  : xiezheng
# @Site    : 
# @File    : demo.py


import numpy as np
import sklearn
from sklearn import preprocessing

# np.loadtxt('D:/2019Programs/jidian2019/极点智能/Draw ROC/jidian_face_112x112_499_align_result/jidian_face_112x112_499_align_clear_mxnet_resnet34_gpu/0008/20160831173809824.dat.bmp.txt')
# print('2345'=='2345')

# count = 0
# for i in range(1, 6261):
#     count += i
#     print(i, count)

a = np.array([[1,2,3,4]])
print(a.shape)
feat = sklearn.preprocessing.normalize(a)
print(feat)

b = a.reshape(4,1)
feat = sklearn.preprocessing.normalize(b, axis=0)
print(feat)