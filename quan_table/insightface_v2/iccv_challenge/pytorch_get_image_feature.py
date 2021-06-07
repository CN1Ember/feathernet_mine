#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 20:33
# @Author  : xiezheng
# @Site    : 
# @File    : get_image_feature.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from time import time

import os.path
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn import preprocessing

import torch
from torch import nn

from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v2_depth import Mobilefacenetv2_v2_depth
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v2_depth_width import Mobilefacenetv2_v2_depth_width
from insightface_v2.model.mobilefacenetv2_width import Mobilefacenetv2_width
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v3_width import Mobilefacenetv2_v3_width
from insightface_v2.iccv_challenge.insightface_resnet import LResNet100E_IR

image_shape = None
# net = None
data_size = 1862120
emb_size = 0
use_flip = True


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_feature(buffer, model):
    global emb_size
    if use_flip:
        input_blob = np.zeros((len(buffer) * 2, 3, image_shape[1], image_shape[2]))
    else:
        input_blob = np.zeros((len(buffer), 3, image_shape[1], image_shape[2]))
    idx = 0

    img_process_start_time = time()
    for item in buffer:
        # img = cv2.imread(item)[:, :, ::-1]  # to rgb

        img = cv2.imread(item, cv2.IMREAD_COLOR)
        if img is None:
            print('parse image', item, 'error')
            return None
        assert img.shape == (image_shape[1], image_shape[2], image_shape[0])

        # print(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape((1, 1, 3))
        img = img.astype(np.float32) - v_mean
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        attempts = [0, 1] if use_flip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            # print("_img={}".format(_img))
            # assert False
            if flipid == 1:
                do_flip(_img)
            input_blob[idx, ...] = _img
            idx += 1
    img_process_end_time = time()
    img_process_time = img_process_end_time-img_process_start_time
    print('img process time={}'.format(img_process_time))

    batch_img = torch.tensor(input_blob).float()
    # print("batch_img[0], {}".format(batch_img[0]))
    # print("batch_img[0], {}".format(batch_img[0].numpy()))
    # assert False

    forward_start_time = time()
    with torch.no_grad():
        _embedding = model(batch_img.cuda())
    _embedding = _embedding.data.cpu().numpy()
    forward_end_time = time()
    forward_time = forward_end_time - forward_start_time
    print('forward time={}'.format(forward_time))

    if emb_size == 0:
        emb_size = _embedding.shape[1]
        print('set emb_size to ', emb_size)

    normalize_start_time = time()
    embedding = np.zeros((len(buffer), emb_size), dtype=np.float32)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1 + embedding2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)
    normalize_end_time = time()
    normalize_time = normalize_end_time - normalize_start_time
    print('normalize time={}'.format(normalize_time))

    total_time = normalize_end_time - img_process_start_time
    print('img process time take {:.3f}'.format(float(img_process_time/total_time)))
    print('forward time take {:.3f}'.format(float(forward_time / total_time)))
    print('normalize time take {:.3f}'.format(float(normalize_time / total_time)))
    assert False
    return embedding


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows, cols, cols * 4, 5))
        f.write(m.data)


def main(args):
    global image_shape
    # global net
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    image_shape = [int(x) for x in args.image_size.split(',')]
    print("image_shape={}".format(image_shape))
    model_state = torch.load(args.model)

    # model = Mobilefacenetv2_v2_depth(blocks=[3, 8, 16, 5], embedding_size=512, fc_type="gdc")
    model = Mobilefacenetv2_v2_depth_width(blocks=[3, 8, 16, 5], embedding_size=512, fc_type="gdc", width_mult=1.0)
    # model = Mobilefacenetv2_width(embedding_size=512)
    # model = Mobilefacenetv2_v3_width(embedding_size=512, width_mult=1.315)
    # model = LResNet100E_IR()

    # model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)

    print("model={}".format(model))
    model_state_param = model_state['model']
    # model.load_state_dict(model_state)
    model.load_state_dict(model_state_param)
    # lfw_acc = model_state["lfw_best_acc"]
    # print("epoch={}, lfw acc={:4f}".format(model_state['epoch'], lfw_acc))

    print("|===> model load success !!!")
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    features_all = None
    i = 0
    fstart = 0
    buffer = []
    output_path = args.output.rsplit('/', 1)[0]
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for line in open(os.path.join(args.input, 'filelist.txt'), 'r'):
        if i % 1000 == 0:
            print("processing ", i)
        i += 1
        line = line.strip()
        image_path = os.path.join(args.input, line)
        buffer.append(image_path)
        if len(buffer) == args.batch_size:
            embedding = get_feature(buffer, model)
            buffer = []
            fend = fstart + embedding.shape[0]
            if features_all is None:
                features_all = np.zeros((data_size, emb_size), dtype=np.float32)
            # print('writing', fstart, fend)
            features_all[fstart:fend, :] = embedding
            fstart = fend
        # if i == 100:
        #     break

    if len(buffer) > 0:
        embedding = get_feature(buffer, model)
        fend = fstart + embedding.shape[0]
        print('writing', fstart, fend)
        features_all[fstart:fend, :] = embedding

    write_bin(args.output, features_all)
    # os.system("bypy upload %s"%args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=64)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    # 241
    # parser.add_argument('--input', type=str, help='', default='/mnt/ssd/faces/iccv_challenge/val/iccv19-challenge-data')
    # 186
    parser.add_argument('--input', type=str, help='', default='/home/xiezheng/dataset/face/iccv_challenge/'
                                                              'val/iccv19-challenge-data')

    # mobilefacenet_r2-8-16-4_256
    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/program2019/'
    #                      'insightface_DCP/iccv_challenge_image_feature/mobilefacenet_r2-8-16-4_256_iccv_ms1m_feature.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/iccv_challenge/'
    #                             'mobilefacenet_r2-8-16-4_256/insightface_mobilefacenetv2_epoch27.pth')

    # r34
    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/program2019/'
    #                     'insightface_DCP/iccv_challenge_image_feature/r34_512_iccv_ms1m_feature.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/log/insightface_r34_epoch48.pth')


    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/programs2019/insightface_DCP/'
    #                     'insightface_v2/iccv_challenge/iccv_result/pytorch_zq_mobilefacenet_r4-8-16-4_512_softmax_checkpoint_7.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/iccv_challenge/model/'
    #                             'pytorch_zq_mobilefacenet_r4-8-16-4_512_softmax_checkpoint_7.pth')

    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/programs2019/insightface_DCP/'
    #                     'insightface_v2/iccv_challenge/sub_iccv_1000_result/mobilenfacenet_r1-4-6-2_gdc_512_440MFLOPs_softmax.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_1000_log_gdc/'
    #                             'mobilenfacenet_r1-4-6-2_gdc_512_440MFLOPs_softmax/check_point/checkpoint_28.pth')

    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/programs2019/insightface_DCP/'
    #                     'insightface_v2/iccv_challenge/sub_iccv_1000_result/mobilenfacenet_r3-8-16-5_gdc_512_995MFLOPs_softmax.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_1000_log_gdc/'
    #                             'mobilenfacenet_r3-8-16-5_gdc_512_995MFLOPs_softmax/check_point/checkpoint_22.pth')

    # parser.add_argument('--output', type=str, help='', default='/home/xiezheng/programs2019/insightface_DCP/'
    #                                                            'insightface_v2/iccv_challenge/sub_iccv_1000_result/'
    #                                                            'mobilenfacenet_r2-8-16-8_depth_150_gdc_5x5_se_512_993MFLOPs_softmax.bin')
    # parser.add_argument('--model', type=str, help='',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_1000_log_gdc/'
    #                             'mobilenfacenet_r2-8-16-8_depth_150_gdc_5x5_se_512_993MFLOPs_softmax/check_point/checkpoint_22.pth')

    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--gpu', type=str, help='', default='0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

