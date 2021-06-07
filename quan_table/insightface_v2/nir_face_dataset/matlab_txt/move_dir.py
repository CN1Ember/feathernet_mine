#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/18 16:32
# @Author  : xiezheng
# @Site    : 
# @File    : move_dir.py

import shutil
import os
from pathlib import Path

def copy_dir(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        print('创建配置文件目录成功！\n')
    shutil.copytree(src_path, dst_path)


def move_dir(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        print('创建配置文件目录成功！\n')
    shutil.move(src_path, dst_path)


def move_dir_txt(txt_path, data_dir, store_dir):
    num = 1
    for line in open(txt_path, encoding='gbk'):
        line = line.strip()
        print('num={}, line={}'.format(num, line))
        src_dir = data_dir +'\\' + line
        print('data_dir={}\nstore_dir={}'.format(src_dir, store_dir))
        move_dir(src_dir, store_dir)
        num = num + 1
        # return



if __name__ == '__main__':
    txt = './select.txt'
    data_dir = 'D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727_aligned\\facedataset123_112_112_align\\face_dataset3_112_112_align\\face_test_200_align_split\\face_test_200_align'
    store_dir = 'D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727_aligned\\facedataset123_112_112_align\\face_dataset3_112_112_align\\face_test_200_align_split\\face_test_200_align_test'

    # data_dir = Path(data_dir)
    # store_dir = Path(store_dir)
    move_dir_txt(txt, data_dir, store_dir)

