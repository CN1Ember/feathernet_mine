#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 19:39
# @Author  : xiezheng
# @Site    : 
# @File    : mxnet_read_img.py

import cv2
import numpy as np
import mxnet as mx
import sklearn
from sklearn.preprocessing import normalize


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])

def get_feature(net, buffer, use_flip, image_shape=[3,112,112]):
    global emb_size
    if use_flip:
        input_blob = np.zeros((len(buffer) * 2, 3, image_shape[1], image_shape[2]))
    else:
        input_blob = np.zeros((len(buffer), 3, image_shape[1], image_shape[2]))
    idx = 0
    for item in buffer:
        img = cv2.imread(item)[:, :, ::-1]  # to rgb
        img = np.transpose(img, (2, 0, 1))
        attempts = [0, 1] if use_flip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            if flipid == 1:
                do_flip(_img)
            input_blob[idx] = _img
            idx += 1
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    net.model.forward(db, is_train=False)
    _embedding = net.model.get_outputs()[0].asnumpy()

    # print("_embedding[0]={}".format(_embedding[0]))
    # assert False

    if emb_size == 0:
        emb_size = _embedding.shape[1]
        print('set emb_size to ', emb_size)
    embedding = np.zeros((len(buffer), emb_size), dtype=np.float32)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1 + embedding2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)
    return embedding