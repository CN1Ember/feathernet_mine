#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 14:11
# @Author  : xiezheng
# @Site    : 
# @File    : opencv_pil.py

import cv2
from PIL import Image
import numpy
import accimage
import numpy as np

# cv2_img = cv2.imread('./3832.jpg')
# cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
# print(cv2_img.shape, type(cv2_img))
# print("cv2_img={}", cv2_img[0])
# print("cv2_img\n", cv2_img)
# cv2.imshow("OpenCV", cv2_img)
# cv2_arr = numpy.asarray(cv2_img)
# print(cv2_arr)
# cv2_arr = (cv2_arr - 127.5)/128
# print(cv2_arr)


print('*'*60)
pil_img = Image.open('./3832.jpg')
pil_img = pil_img.convert('RGB')
# print("pil_img\n", pil_img)
# pil_img.show()
pil_arr = numpy.asarray(pil_img)
print(pil_arr.shape)
pil_arr = np.transpose(pil_arr, (2, 0, 1))
print("pil_arr", pil_arr)
pil_arr = (pil_arr - 127.5)/128.0
print("pil_arr", pil_arr)

# img = cv2.cvtColor(numpy.asarray(pil_img),cv2.COLOR_RGB2BGR)
# cv2.imshow("OpenCV",img)
# cv2.waitKey()

def image_to_np(image):
    """
    Returns:
        np.ndarray: Image converted to array with shape (width, height, channels)
    """
    image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
    image.copyto(image_np)
    # image_np = np.transpose(image_np, (1, 2, 0))
    return image_np


acc_img = accimage.Image('./3832.jpg')
# acc_img = acc_img.convert('RGB')
# acc_img = numpy.asarray(acc_img)
acc_img = image_to_np(acc_img)
print("*"*60)
print(acc_img.shape)
print("acc_img\n", acc_img)
acc_img = (acc_img - 127.5)/128.0
print("acc_img\n", acc_img)