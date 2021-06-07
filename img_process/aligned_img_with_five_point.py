#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 21:23
# @Author  : xiezheng
# @Site    : 
# @File    : label_aligned_400.py
import numpy as np
from skimage import transform as trans
# from scipy import misc
import imageio
import cv2
import os


def aligned_img(image_path, points, target_file):
    try:
        # print("image_path: ", image_path)
        img = imageio.imread(image_path)   # RGB
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)



    # 112 * 112
    image_size = [112, 112]
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)


    dst = points.reshape((5, 2))
    # print('dst={}'.format(dst))
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    # tform.estimate(src, dst)
    M = tform.params[0:2, :]
    # print('M={}'.format(M))
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

    if len(warped.shape) == 3:
        if warped.shape[2] == 3:
            # reverse the last dimension
            bgr = warped[..., ::-1]
    else:
        bgr = warped

    # cv2.imshow('aligned', bgr)
    # cv2.waitKey(0)
    # print(target_file)
    res = cv2.imwrite(target_file, bgr)
    # print("res: ", res)
    # print('image_path={}, target_file={}'.format(image_path, target_file))


def get_data_txt(line):
    points = np.zeros((10))
    # split_list = line.split('~')
    split_list = line.split('~')
    img_path = split_list[0]

    if(len(split_list)>=2):
        point_list = split_list[1].split(',')
    else:
        return img_path, points, False

    for i in range(0, len(point_list)):
        points[i] = float(point_list[i])

    # print(img_path)

    return img_path, points, True



if __name__ == '__main__':

    # test
    txt_path = './depth_data/2.txt'
    data_path = './depth_data/2/depth'
    # target_path = r'D:\working\ir_dataset_labeled\rgb_challenge_0a_aligned_96_96_scale1.3_y15'
    target_path = './depth_data/2/align'

    # if not os.path.exists(target_path):
    #     os.makedirs(target_path)

    count = 0
    for line in open(txt_path, 'r'):
        line = line.strip()
        print('count={}, line={}'.format(count, line))
        img_path, points, flag = get_data_txt(line)
        print(img_path, points)

        image_path = data_path + '/' + img_path
        target_file = target_path +'/'+ img_path
        print("image_path: ", image_path)
        print("target_file: ", target_file)

        target_dir, _ = target_file.rsplit('/', 1)
        print("target_dir: ", target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if flag:
            aligned_img(image_path, points, target_file)
        else:
            print(image_path," detect no face!")
        count += 1
        # if count == 10:
        #     assert False





