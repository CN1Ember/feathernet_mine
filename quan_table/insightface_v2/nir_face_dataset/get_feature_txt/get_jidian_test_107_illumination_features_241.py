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
from dcp.models.mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_v2.model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm

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


# rgb_mobilefacenet_v1_p0.5
trained_model_path = "/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/" \
                     "test_107_illumination_feature_result/" \
                     "mobilefacenet_cos_lr_pruned_0.5_nowith_fc_checkpoint_026.pth"
save_path = "/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/test_107_illumination_feature_result/" \
            "pytorch_mf_p0.5_fp_python_checkpoint_026"


# # ms1m_v2: mf_p0.5_nir_finetune_last1
# trained_model_path = '/home/xiezheng/program2019/insightface_DCP/insightface_v2/' \
#                      'nir_log/nir_face_part_update_1224_19027/' \
#                      'mobilefacenet_v1_res1-4-6-2_p0.5_last1_lr0.01_epoch30/' \
#                      'check_point/checkpoint_29.pth'
# save_path = "/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/test_107_illumination_feature_result/" \
#             "mobilefacenet_v1_res1-4-6-2_p0.5_nir_last1_update_lr0.01_checkpoint29"


# ms1m_v2: mf_p0.5_nir_finetune_last3
# trained_model_path = '/home/xiezheng/program2019/insightface_DCP/insightface_v2/' \
#                      'nir_log/nir_face_part_update_1224_19027/' \
#                      'mobilefacenet_v1_res1-4-6-2_p0.5_last3_lr0.01_epoch30/' \
#                      'check_point/checkpoint_29.pth'
# save_path = "/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/test_107_illumination_feature_result/" \
#             "mobilefacenet_v1_res1-4-6-2_p0.5_nir_last3_update_lr0.01_checkpoint29"


txt_path = "/home/datasets/Face/FaceRecognition/nir_face_dataset_1224/jidian_test_107_illumination_190727_align.txt"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# model = MobileFaceNet(embedding_size=128, blocks=[1,4,6,2])
model = pruned_Mobilefacenet(pruning_rate=0.5)  # old
# model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)

check_point_params = torch.load(trained_model_path)
model.load_state_dict(check_point_params['model'])
# print(model)
model.eval()
print("trained_model_path={}".format(trained_model_path))
print("load pretrained model state success!!!")

if not os.path.isdir(save_path):
    os.makedirs(save_path)

num = 1
for line in open(txt_path):
    line = line.strip()
    # img_path = os.path.join(data_dir, line)
    img_path = line
    print("num= {}\n img_path= {} ".format(str(num), img_path))
    # feature = get_feature(model, img_path).reshape(1, 128)
    feature = get_pytorch_model_feature(model, img_path).reshape(1, 128)
    # print(feature)
    # print("feature size=".format(feature.size()))
    str_array = line.split('/')
    newpath = os.path.join(save_path, str_array[-2])
    if not os.path.isdir(newpath):
        os.makedirs(newpath)
    storePath = os.path.join(newpath, str_array[-1] + ".txt")
    print("storePath= {}".format(storePath))
    np.savetxt(storePath, feature, fmt='%.8f')
    num = num + 1
    # break