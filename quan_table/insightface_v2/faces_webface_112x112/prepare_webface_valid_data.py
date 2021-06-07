#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 10:48
# @Author  : xiezheng
# @Site    : 
# @File    : prepare_webface_valid_data.py

from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return carray, issame


def load_val(lines, data_path, store_path, image_size=[112, 112]):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    print("lines = {:d}".format(len(lines)))
    data = bcolz.fill([len(lines)*2, 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=store_path, mode='w')
    marks = []

    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        # print("i={:d}, {}".format(i, lines[i]))
        img1_path, img2_path, label = lines[i].split(' ')
        # print("{} - {} - {}".format(img1_path, img2_path, label))

        img1_path = os.path.join(data_path, img1_path)
        img2_path = os.path.join(data_path, img2_path)

        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # plt.subplot(121)
        # plt.imshow(img)
        # print("1\n", img.shape, type(img))
        # # print("2\n", img.astype(np.uint8))
        data[2 * i, ...] = val_transform(img1)
        # if i == 1:
        #     print(img, '\n', data[i, ...])
        # print("3\n", data[i, ...].shape, type(data[i, ...]))
        # plt.show()
        # break

        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        data[2 * i+1, ...] = val_transform(img2)

        i += 2
        if i % 1000 == 0:
            print('loading bin', i)

        if label == "1":
            mark = True
        elif label == "0":
            mark = False
        else:
            assert False
        marks.append(mark)
        print("{} - {} - {} - {}".format(img1_path, img2_path, label, mark))
        # if i == 10:
        #     break
    print(data.shape)

    np.save(store_path + '_list', np.array(marks))
    return data


if __name__ == '__main__':
    print("start")
    data_path = "/mnt/ssd/faces/faces_webface_112x112/verfication_0.7_imgs"

    valid_data_txt = '/home/xiezheng/program2019/insightface_DCP/insightface_v2/' \
                     'faces_webface_112x112/sub_face_data/sample_valid_list_160000_pairs.txt'
    store_dir = "/mnt/ssd/faces/faces_webface_112x112/160000_imgs_verfication_data"

    # valid_data_txt = '/home/xiezheng/program2019/insightface_DCP/insightface_v2/' \
    #                  'faces_webface_112x112/sub_face_data/sample_valid_list_1600_pairs.txt'
    # store_dir = "/mnt/ssd/faces/faces_webface_112x112/1600_imgs_verfication_data"

    f = open(valid_data_txt, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    print("lines length={:d}".format(len(lines)))
    load_val(lines, data_path, store_dir)