#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 22:18
# @Author  : xiezheng
# @Site    : 
# @File    : draw_img.py

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('0010.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(111)
plt.imshow(img)
plt.show()

img = img.reshape(-1)
print("length=", len(img))
for i in range(len(img)):
    print(img[i])



