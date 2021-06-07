#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 20:43
# @Author  : xiezheng
# @Site    : 
# @File    : get_nir_val_data.py

import os
import random


if __name__ == '__main__':
    print("Test")
    root = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/test_dataset_fill_aligned'
    txt_path = "/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/val_dataset.txt"
    random.seed(123456)
    sample_from_one_dir = 4

    label = os.listdir(root)
    # For same
    for i in range(len(label)):
        path = os.path.join(root, label[i])
        img_name = os.listdir(path)
        if len(img_name) < sample_from_one_dir:
            print("warming: {} len(img_name)={}".format(path, img_name))
            continue
        sample = random.sample(img_name, sample_from_one_dir)
        for name in sample:
            img_path = os.path.join(path, name)
            print("img_path={}".format(img_path))
            with open(txt_path, 'a+') as w:
                w.write(img_path + '\n')

    # For issame
    for j in range(sample_from_one_dir):
        count = 1
        for i in range(len(label)):
            path = os.path.join(root, label[i])
            img_name = os.listdir(path)
            if len(img_name) < 1:
                print("warming: {} len(img_name)={}".format(path, img_name))
                continue
            sample = random.sample(img_name, 1)
            img_path = os.path.join(path, sample[0])
            print("img_path={}".format(img_path))
            with open(txt_path, 'a+') as w:
                w.write(img_path + '\n')
            count = count+1




