#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 20:12
# @Author  : xiezheng
# @Site    : 
# @File    : get_jidian_499_features.py

import os
import cv2
import numpy as np
from torchvision import transforms
import torch

from insightface_v2.model.models import MobileFaceNet

# method 1:
def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_feature(model, image_path, image_shape=[3,112,112]):
    model = model.cuda()
    img = read_image(image_path)
    # print(img.shape)
    if img is None:
        print('parse image', image_path, 'error')
        return None
    assert img.shape == (image_shape[1], image_shape[2], image_shape[0])

    v_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape((1, 1, 3))
    img = img.astype(np.float32) - v_mean
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    img = torch.tensor(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
    # print('图片大小：', img.size())
    F = model(img.cuda())
    F = F.data.cpu().numpy()
    # _norm = np.linalg.norm(F)
    # F /= _norm
    # print(F.shape)
    return F

# method 2:
def get_pytorch_model_feature(model, img_path):
    model = model.cuda()
    aligned = cv2.imread(img_path, cv2.IMREAD_COLOR)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0
    aligned = val_transform(aligned)
    # print("aligned size={}, {}".format(aligned.size(), type(aligned)))
    aligned = aligned.reshape(1, aligned.size()[0], aligned.size()[1], aligned.size()[2])
    # print("aligned size={}, {}".format(aligned.size(), type(aligned)))
    # aligned = torch.tensor(aligned)
    feature = model(aligned.cuda()).data.cpu().numpy()
    return feature


# nir_mobielfacenet_v1_lr0.1: all_update
# trained_model_path = '/home/dataset/xz_datasets/jidian_face_499/model/nir_mobilefacenet_lr0.1_epoch17.pth'
# save_path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/jidian_face_112x112_79_align_clear_nir_pytorch_mobilefacenet_gpu"


# nir_mobielfacenet_v1_lr0.01: last1_update
trained_model_path = '/home/dataset/xz_datasets/jidian_face_499/model/nir_mobilefacenet_part_update_last1_lr0.01_epoch29.pth'
save_path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/" \
            "nir_face_part_update_pytorch_mobilefacenet/jidian_face_112x112_79_align_clear_last1_lr0.01"

# save_path = "/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop-fill-aligned/jidian_face_79_algo_crop_fill_aligned_feature"
# txt_path = "/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop-fill-aligned/jidian_face_79_algo_crop_fill_aligned.txt"
# data_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_79_algo_crop-fill-aligned/jidian_face_79_algo_crop_fill_aligned'

# nir_mobielfacenet_v1_lr0.01: last2_update
# trained_model_path = '/home/dataset/xz_datasets/jidian_face_499/model/nir_mobilefacenet_part_update_last2_lr0.01_epoch27.pth'
# save_path = "/home/dataset/xz_datasets/jidian_face_79/feature_result/" \
#             "nir_face_part_update_pytorch_mobilefacenet/jidian_face_112x112_79_align_clear_last2_lr0.01"

if not os.path.isdir(save_path):
    os.makedirs(save_path)

# rgb_mobilefacenet_v1
# trained_model_path = '/home/dataset/xz_datasets/jidian_face_499/model/mobilefacenet_v1_epoch34.pth'
# save_path = "/home/dataset/xz_datasets/jidian_face_499/feature_result/jidian_face_112x112_499_align_clear_pytorch_mobilefacenet_gpu"

txt_path = "/home/dataset/xz_datasets/jidian_face_79/jidian_face_112x112_align_79_v2.txt"
data_path = '/home/dataset/xz_datasets/jidian_face_79/jidian_face_112x112_align_79_v2/jidian_face_v2'

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
model = MobileFaceNet(embedding_size=128, blocks=[1,4,6,2])
check_point_params = torch.load(trained_model_path)
model.load_state_dict(check_point_params['model'])
# print(model)
model.eval()
print("trained_model_path={}".format(trained_model_path))
print("load pretrained model state!!!")

num = 1
for line in open(txt_path):
    line = line.strip()
    _str = line.split('/')
    label = _str[-2]
    img_name = _str[-1]
    img_path = os.path.join(data_path, label, img_name)
    print("num= {}\n img_path= {} ".format(str(num), img_path))
    # feature = get_feature(model, img_path).reshape(1, 128)
    feature = get_pytorch_model_feature(model, img_path).reshape(1, 128)
    # print(feature)

    newpath = os.path.join(save_path, label)
    if not os.path.isdir(newpath):
        os.makedirs(newpath)
    storePath = os.path.join(newpath, img_name + ".txt")
    print("storePath= {}".format(storePath))
    np.savetxt(storePath, feature, fmt='%.8f')
    num = num + 1
    # break