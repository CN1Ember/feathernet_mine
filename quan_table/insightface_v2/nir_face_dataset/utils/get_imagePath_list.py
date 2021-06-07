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
    # path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/test_dataset_fill_aligned'
    # txt_path = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/test_dataset_fill_aligned.txt'

    # path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop_fill'
    # txt_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop_fill.txt'

    # path = '/home/dataset/xz_datasets/jidian_face_9784/clean_anno_jidian_190724_resnet100_aligned_nofilp'
    # txt_path = '/home/dataset/xz_datasets/jidian_face_9784/clean_anno_jidian_190724_resnet100_aligned_nofilp.txt'

    path = 'D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727\\facedataset123_112_112_align\\face_dataset3_112_112_align\\NIR_face_dataset'
    txt_path = 'D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727\\facedataset123_112_112_align\\face_dataset3_112_112_align\\NIR_face_dataset.txt'


    fun(path, txt_path)