#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午10:32
# @Author  : xiezheng
# @Site    : 
# @File    : get_pathList.py

import glob
import os


def fun(path, txt_path):
    f = open(txt_path, 'a+')
    for fn in glob.glob(path + os.sep + '*'):  # '*'代表匹配所有文件
        if os.path.isdir(fn):  # 如果结果为文件夹
            fun(fn, txt_path)   # 递归
        else:
            if fn.endswith('.BMP') or fn.endswith('.jpg') or fn.endswith('.bmp') or fn.endswith('.png'):
                print(fn)
                # print(str(fn), len(str(fn)))
                # print("345", "23456")
                # assert False
                f.write(fn + '\n')

    f.close()


path = "/home/dataset/xz_datasets/nir_face_test/err_20191016_offical_align"
txt_path = "/home/dataset/xz_datasets/nir_face_test/err_20191016_offical_align.txt"

fun(path, txt_path)