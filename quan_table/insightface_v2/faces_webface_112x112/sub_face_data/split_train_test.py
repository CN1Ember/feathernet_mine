#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/22 23:20
# @Author  : xiezheng
# @Site    : 
# @File    : split_train_test.py

import os
import random

# train_path = "/mnt/ssd/faces/faces_webface_112x112/train_0.3_imgs"
# train_txt = "0.3_imgs_train.txt"
# test_txt = "0.3_imgs_test.txt"

train_path = "/mnt/ssd/faces/faces_webface_112x112/train_0.1_imgs"
train_txt = "0.1_imgs_train.txt"
test_txt = "0.1_imgs_test.txt"

my_txt = "5.txt"
list_5 = []
train_list = []
test_list = []

split_rate = 0.2
folder_list = os.listdir(train_path)
print(folder_list)
folder_num = len(folder_list)
print("folder_num={}".format(folder_num))

for i in range(len(folder_list)):
    imgs_path = os.path.join(train_path, folder_list[i])
    imgs= os.listdir(imgs_path)
    if len(imgs) < 5:
        print("imgs<5, path={}".format(imgs_path))
        list_5.append(imgs_path)
        continue
    else:
        # train_list.append(folder_list[i])
        imgs_num = len(imgs)
        imgs_index_list = range(imgs_num)
        sample_imgs_index_list = random.sample(imgs_index_list, imgs_num)

        for j in range(len(sample_imgs_index_list)):
            path = os.path.join(imgs_path, imgs[sample_imgs_index_list[j]])
            print("path={}".format(path))
            if j < int(split_rate*imgs_num):
                test_list.append(path)
                # print("test")
            else:
                train_list.append(path)
                # print("train")

with open(train_txt, 'w+') as f:
    for i in range(len(train_list)):
        f.write(train_list[i] + '\n')

with open(test_txt, 'w+') as f:
    for i in range(len(test_list)):
        f.write(test_list[i] + '\n')

print("num<5:")
for i in range(len(list_5)):
    print(list_5[i])






