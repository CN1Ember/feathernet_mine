#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 16:56
# @Author  : xiezheng
# @Site    : 
# @File    : rewrite_txt.py

# from string import replace

txt_path = "/home/dataset/xz_datasets/jidian_face_499/jidian_face_112x112_499_align_clear.txt"
save_txt_path = "/home/dataset/xz_datasets/jidian_face_499/jidian_face_499_clear_list.txt"
f = open(save_txt_path, mode='w')

for line in open(txt_path):
    line = line.strip()
    print("line={}".format(line))
    mew_line = line.replace("_", "/")
    print("line={}".format(mew_line))
    f.write("{}\n".format(mew_line))

f.close()


