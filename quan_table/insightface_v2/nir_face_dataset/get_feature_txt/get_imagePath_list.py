#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 17:24
# @Author  : xiezheng
# @Site    : 
# @File    : get_imagePath_list.py

import os
import glob

def fun(path, txt_path):
    f = open(txt_path, 'a+')
    for fn in glob.glob(path + os.sep + '*'):  # '*'代表匹配所有文件
        if os.path.isdir(fn):  # 如果结果为文件夹
            fun(fn, txt_path)   # 递归
        else:
            if fn.endswith('.BMP') or fn.endswith('.jpg') or fn.endswith('.bmp') or fn.endswith('.png'):
                print(fn)
                f.write(fn + '\n')
    f.close()


if __name__ == '__main__':
    # Get image path list

    # # 186
    # path = '/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2'
    # txt_path = '/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2.txt'

    path = '/home/dataset/xz_datasets/jidian_face_1178_20190727/face_test_200_align'
    txt_path = '/home/dataset/xz_datasets/jidian_face_1178_20190727/face_test_200_align.txt'

    # 241
    # path = '/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/jidian_test_107_illumination_190727_align'
    # txt_path = '/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/jidian_test_107_illumination_190727_align.txt'

    fun(path, txt_path)