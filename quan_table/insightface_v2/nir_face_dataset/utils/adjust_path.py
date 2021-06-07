#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 15:36
# @Author  : xiezheng
# @Site    : 
# @File    : adjust_path.py


from pathlib import Path
import shutil
import os


if __name__ == '__main__':
    txt_path = "D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727\\facedataset123_112_112_align\\face_dataset3_112_112_align\\NIR_face_dataset.txt"
    store_dir = "D:\\2019Programs\jidian2019\极点智能\Draw_ROC\datasets\jidian_data_20190727\\facedataset123_112_112_align\\face_dataset3_112_112_align\\face_test_200"

    name = '2019'
    num = 0

    for line in open(txt_path):
        save_dir = Path(store_dir)
        line = line.strip()
        print("line={}".format(line))
        img_name = line.split('\\')[-1]
        print('img_name={}'.format(img_name))
        img_class, new_img_name = img_name.split('-', 1)
        print('img_class={}, new_img_name={}'.format(img_class, new_img_name))

        save_dir = save_dir / (name + img_class)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        new_path = save_dir / new_img_name
        shutil.copy(line, new_path)
        print('new_path={}'.format(new_path))

        num = num + 1
        print('num={}'.format(num))
        # if num == 100:
        #     break


