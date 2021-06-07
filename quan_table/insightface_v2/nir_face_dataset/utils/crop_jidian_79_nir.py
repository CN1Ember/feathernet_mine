#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:47
# @Author  : xiezheng
# @Site    : 
# @File    : crop_jidian_79_nir.py

import os
import cv2

data_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_v2'
txt_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_detection_result.txt'
save_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop'

if not os.path.isdir(save_path):
    os.makedirs(save_path)

num = 1
for line in open(txt_path):
    line = line.strip()
    print("num={:d}, line={}".format(num, line))
    _line = line.rsplit(' ', 1)

    path = _line[-2]
    _path = path.split('/')
    label = _path[-2]
    img_name = _path[-1]

    points = _line[-1]
    _points = points.split(',')
    int_points = [int(_points[i]) for i in range(len(_points))]
    if int_points[0] == 0:
        continue
    # print("points={}, type={}".format(int_points, type(int_points[0])))

    img_path = os.path.join(data_path, label, img_name)
    img_save_dir = os.path.join(save_path, label)
    if not os.path.isdir(img_save_dir):
        os.makedirs(img_save_dir)
    img_save_path = os.path.join(img_save_dir, img_name)
    print("save_path={}".format(img_save_path))

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x,y,w,h = int_points[0],int_points[1],int_points[2],int_points[3]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    crop_img = img[y: y + h, x: x + w]
    resize_img = cv2.resize(crop_img, (100, 100))
    cv2.imwrite(img_save_path, resize_img)
    num += 1
    # break








