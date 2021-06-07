#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22 21:22
# @Author  : xiezheng
# @Site    : 
# @File    : get_val_nir_face.py

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


def load_val(lines, rootdir, image_size=[112, 112]):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)

    print("lines = {:d}".format(len(lines)))
    data = bcolz.fill([len(lines), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')

    for i in range(len(lines)):
        print("i={:d}, {}".format(i, lines[i]))
        img = cv2.imread(lines[i].strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.subplot(121)
        # plt.imshow(img)
        # print("1\n", img.shape, type(img))
        # # print("2\n", img.astype(np.uint8))
        data[i, ...] = val_transform(img)
        # if i == 1:
        #     print(img, '\n', data[i, ...])
        # print("3\n", data[i, ...].shape, type(data[i, ...]))
        # plt.show()
        # break
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
        # if i == 10:
        #     break
    print(data.shape)
    return data


if __name__ == '__main__':
    print("start")
    val_data_txt = '/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test/val_dataset.txt'
    rootdir = os.path.join('/home/dataset/Face/xz_FaceRecognition/nir_face_dataset/Test', 'val')

    f = open(val_data_txt, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    print("lines length={:d}".format(len(lines)))
    load_val(lines, rootdir)


