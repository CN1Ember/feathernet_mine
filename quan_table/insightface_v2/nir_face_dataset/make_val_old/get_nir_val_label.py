#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22 15:59
# @Author  : xiezheng
# @Site    : 
# @File    : get_nir_val_label.py

import numpy as np

val_data_txt= '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/val_dataset.txt'
val_label_txt = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/val_label'
labels = []
marks = []

f = open(val_data_txt,"r")
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
for line in lines :
    print(line)
    line = line.strip()
    line_str = line.split('/')
    label = line_str[-2]
    print("label={}".format(label))
    labels.append(label)
print("len(labels)={:d}".format(len(labels)))

for i in range(int(len(labels)/2)):
    label1 = labels[2 * i]
    label2 = labels[2 * i + 1]
    if label1 == label2:
        mark = True
    else:
        mark = False
    marks.append(mark)
    print("{},{},{}".format(label1, label2, mark))

np.save(val_label_txt + '_list', np.array(marks))




