#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 21:23
# @Author  : xiezheng
# @Site    : 
# @File    : label_aligned_400.py

import numpy as np
from skimage import transform as trans
from scipy import misc
import cv2
import os


def aligned_img(image_path, points, target_file):
    try:
        img = misc.imread(image_path)   # RGB
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)

    image_size = [112, 112]
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    if image_size[1] == 112:     # attention
        src[:, 0] += 8.0

    dst = points.reshape((5, 2))
    # print('dst={}'.format(dst))
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    # print('M={}'.format(M))
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
    bgr = warped[..., ::-1]
    # cv2.imshow('aligned', bgr)
    # cv2.waitKey(0)
    cv2.imwrite(target_file, bgr)
    # print('image_path={}, target_file={}'.format(image_path, target_file))


def get_data_txt(line):
    points = np.zeros((10))
    line_str = line.split('_190723/')[-1]
    img_path, p1, p2, p3,p4,p5,p6,p7,p8,p9,p10 = line_str.split(',')

    points[0] = float(p1)
    points[1] = float(p2)
    points[2] = float(p3)
    points[3] = float(p4)
    points[4] = float(p5)
    points[5] = float(p6)
    points[6] = float(p7)
    points[7] = float(p8)
    points[8] = float(p9)
    points[9] = float(p10)

    return img_path, points



if __name__ == '__main__':
    txt_path = 'D:/2019Programs/jidian2019/faceRecog_test/clean_anno_jidian_190723.txt'
    data_path = 'D:/2019Programs/jidian2019/faceRecog_test/clean_anno_jidian_190723'
    target_path = 'D:/2019Programs/jidian2019/faceRecog_test/clean_anno_jidian_190723_aligned'

    # if not os.path.exists(target_path):
    #     os.makedirs(target_path)

    count = 0
    for line in open(txt_path, 'r'):
        line = line.strip()
        print('count={}, line={}'.format(count, line))
        img_path, points = get_data_txt(line)
        # print(img_path, points)

        image_path = data_path + '/' + img_path
        target_file = target_path +'/'+ img_path

        target_dir, _ = target_file.rsplit('/', 1)
        # print(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        aligned_img(image_path, points, target_file)
        count += 1
        # if count == 10:
        #     assert False





