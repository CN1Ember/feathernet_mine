#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 22:52
# @Author  : xiezheng
# @Site    : 
# @File    : demo.py

import cv2
import numpy as np

origin_img = cv2.imread("./77.jpg")
origin_img = origin_img.astype(np.float32)
# origin_img_reshap = origin_img.reshape(-1)
# print("origin_img")
# for i in range(len(origin_img_reshap)):
#     print("i={}, data={}".format(i, origin_img_reshap[i]))

img = cv2.imread("./mxnet_77.jpg")
img = img.astype(np.float32)
# img_reshap = img.reshape(-1)
# print("img")
# for i in range(len(img_reshap)):
#     print("i={}, data={}".format(i, img_reshap[i]))

delta = origin_img-img
print("origin_img-img={}".format(delta))
delta_reshap = delta.reshape(-1)
for i in range(len(delta_reshap)):
    print("i={}, data={}".format(i, delta_reshap[i]))
print(np.sum(np.abs(delta)))
print(np.mean(np.abs(delta)))

