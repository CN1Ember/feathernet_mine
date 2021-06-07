#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 16:22
# @Author  : xiezheng
# @Site    : 
# @File    : read_np.py

import numpy as np

path = './pytorch_img.npy'
img = np.load(path)
print(img)