#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 16:05
# @Author  : xiezheng
# @Site    : 
# @File    : nir_img_fill.py

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os


def image_fill(img_path, outpath, fill_color=[0,0,0], fill_length=10):
    Black = fill_color
    str_split = img_path.split('/')
    img_name = str_split[-1]
    label = str_split[-2]
    img = cv2.imread(img_path)
    constant = cv2.copyMakeBorder(img, fill_length, fill_length, fill_length, fill_length, cv2.BORDER_CONSTANT, value=Black)
    img_output = os.path.join(outpath, label)

    if not os.path.exists(img_output):
        os.makedirs(img_output)

    img_path = os.path.join(img_output, img_name)
    print('img_output={}'.format(img_path))
    cv2.imwrite(img_path, constant)


if __name__ == '__main__':
    # For test
    # constant = image_fill(img_path="01.bmp")
    # cv2.imshow("fill", constant)
    # cv2.imwrite("01_fill.bmp", constant)
    # cv2.waitKey(0)

    # txt_path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/train_dataset.txt'
    # out_path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/train_dataset_v3'

    # txt_path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/test_dataset.txt'
    # out_path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/test_dataset_fill'

    txt_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop.txt'
    out_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop_fill'

    count = 0
    for line in open(txt_path, encoding='gbk'):
        line = line.strip()
        print('count={:d}'.format(count))
        print('line={}'.format(line))
        image_fill(line, out_path)
        count = count+1
