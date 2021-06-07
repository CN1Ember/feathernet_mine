#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 11:27
# @Author  : xiezheng
# @Site    : 
# @File    : sample_valid_list_40000.py

import os
import random

valid_large_data_txt = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/" \
                 "sub_face_data/sample_valid_list_160000_pairs.txt"

# valid_small_data_txt = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/" \
#                  "sub_face_data/sample_valid_list_40000_pairs.txt"
valid_small_data_txt = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/" \
                 "sub_face_data/sample_valid_list_1600_pairs.txt"

sample_rate = 0.01
f = open(valid_large_data_txt, "r")
lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
print("lines length={:d}".format(len(lines)))

lines_index_list = range(len(lines))
sample_num = int(sample_rate*len(lines))
print("sample={}".format(sample_num))

# sample_lines_index_list = random.sample(lines_index_list, sample_num)
# print("len sample_lines_index_list={}".format(sample_lines_index_list))

result_list = random.sample(range(0, len(lines)), sample_num)
print("len sample_lines_index_list={}".format(len(result_list)))

small_data_list = []
for i in range(len(result_list)):
    small_data_list.append(lines[result_list[i]].strip())


with open(valid_small_data_txt, 'w+') as f:
    for i in range(len(small_data_list)):
        print("{}".format(small_data_list[i]))
        f.write(small_data_list[i] + '\n')
