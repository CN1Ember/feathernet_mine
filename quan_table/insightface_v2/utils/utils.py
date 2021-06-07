import argparse
import logging
import math
import os
import sys
import io

import cv2 as cv
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

# from insightface_v2.utils.align_faces import get_reference_facial_points, warp_and_crop_face
# from insightface_v2.mtcnn.detector import detect_faces

# from align_faces import get_reference_facial_points, warp_and_crop_face
# from mtcnn.detector import detect_faces

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during back-propagation to avoid explosion of gradients. 避免梯度爆炸
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def accuracy(outputs, labels, topk=(1,)):
#     """
#     Computes the precision@k for the specified values of k
#     :param outputs: the outputs of the model
#     :param labels: the ground truth of the data
#     :param topk: the list of k in top-k
#     :return: accuracy
#     """
#
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = labels.size(0)
#
#         _, pred = outputs.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(labels.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size).item())
#         return res

def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.long().view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (1.0 / batch_size)

# def align_face(img_fn, facial5points, image_h=112, image_w=112):
#     raw = cv.imread(img_fn, True)   # BGR
#     facial5points = np.reshape(facial5points, (2, 5))
#
#     crop_size = (image_h, image_w)
#
#     default_square = True
#     inner_padding_factor = 0.25
#     outer_padding = (0, 0)
#     output_size = (image_h, image_w)
#
#     # get the reference 5 landmarks position in the crop settings
#     reference_5pts = get_reference_facial_points(
#         output_size, inner_padding_factor, outer_padding, default_square)
#
#     # dst_img = warp_and_crop_face(raw, facial5points)
#     dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
#     return dst_img

# def get_face_attributes(full_path):
#     try:
#         img = Image.open(full_path).convert('RGB')
#         bounding_boxes, landmarks = detect_faces(img)
#
#         if len(landmarks) > 0:
#             landmarks = [int(round(x)) for x in landmarks[0]]
#             return True, landmarks
#
#     except KeyboardInterrupt:
#         raise
#     except:
#         pass
#     return False, None

# def select_central_face(im_size, bounding_boxes):
#     width, height = im_size
#     nearest_index = -1
#     nearest_distance = 100000
#     for i, b in enumerate(bounding_boxes):
#         x_box_center = (b[0] + b[2]) / 2
#         y_box_center = (b[1] + b[3]) / 2
#         x_img = width / 2
#         y_img = height / 2
#         distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
#         if distance < nearest_distance:
#             nearest_distance = distance
#             nearest_index = i
#
#     return nearest_index

# def get_central_face_attributes(full_path):
#     try:
#         img = Image.open(full_path).convert('RGB')
#         bounding_boxes, landmarks = detect_faces(img)
#
#         if len(landmarks) > 0:
#             i = select_central_face(img.size, bounding_boxes)
#             return True, [bounding_boxes[i]], [landmarks[i]]
#
#     except KeyboardInterrupt:
#         raise
#     except:
#         pass
#     return False, None, None

# def get_all_face_attributes(full_path):
#     img = Image.open(full_path).convert('RGB')
#     bounding_boxes, landmarks = detect_faces(img)
#     return bounding_boxes, landmarks

def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img


# for zqmobilefacenet_r4-8-16-4_512
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='iccv_emore', help='train_data mode: emore, emore_debug, iccv_emore')
#
#     # # 241
#     parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1', help='train data folder')
#
#     # 230
#     # parser.add_argument('--emore_folder', default='/home/dataset/Face/xz_FaceRecognition/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#
#     parser.add_argument('--gpu', default='3,5,6,7', help='ngpu use')
#     parser.add_argument('--outpath', default='./iccv_log/zq_mobilefacenet_res4-8-16-4_512_gdc_softmax_', help='output path')
#     parser.add_argument('--step', default=[], type=list, help='ngpu use')
#
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#     parser.add_argument('--network', default='zq_mobilefacenet', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--block_setting', default=[4,8,16,4], type=list, help='block num')
#     parser.add_argument('--fc_type', default='gnap', type=str, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=36, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=400, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--cos_lr', type=bool, default=True, help='full logging')
#     parser.add_argument('--loss_type', type=str, default="softmax", help='full logging')
#
#     parser.add_argument('--checkpoint', type=str, default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/'
#                         'iccv_log/zq_mobilefacenet_res4-8-16-4_512_gdc_softmax/check_point/checkpoint_4.pth', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=8, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# zq_mobilefacenet_r4-8-16-4_softmax_epoch8 大数据集
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='iccv_emore',
#                         help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore')
#
#     # 241
#     parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#                         help='train data folder')
#     # 246
#     # parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#     # 230
#     # parser.add_argument('--emore_folder',
#     #                     default='/home/dataset/Face/xz_FaceRecognition/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#
#     parser.add_argument('--gpu', default='0,1', help='ngpu use')
#
#     # parser.add_argument('--outpath', default='./sub_iccv_log/new_zq_mobilefacenetv1_r4-8-16-4_128_softmax', help='output path')
#     parser.add_argument('--outpath', default='./iccv_log/zq_mobilefacenet_r4-8-16-4_gnap_128_softmax', help='output path')
#     parser.add_argument('--step', default=[], type=list, help='ngpu use')
#
#     # 241
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#
#     # 230
#     # parser.add_argument('--pretrained', type=str, default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/'
#     #                      'sub_iccv_log/mobilefacenet_v1_softmax/check_point/checkpoint_25.pth',help='pretrained model')
#
#     parser.add_argument('--network', default='zq_mobilefacenet', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilenet_v3', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--fc_type', default='gnap', type=str, help='fc type')
#
#     # parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#     parser.add_argument('--block_setting', default=[4,8,16,4], type=list, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=8, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=160, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--cos_lr', type=bool, default=False, help='full logging')
#     parser.add_argument('--loss_type', type=str, default="softmax", help='arcface or softmax')
#
#     parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=8, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='sub_iccv_emore',
#                         help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore')
#
#     # 241
#     # parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#     # 246
#     parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#                         help='train data folder')
#     # 230
#     # parser.add_argument('--emore_folder',
#     #                     default='/home/dataset/Face/xz_FaceRecognition/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#
#     parser.add_argument('--gpu', default='0,1', help='ngpu use')
#
#     parser.add_argument('--outpath', default='./sub_iccv_log/new_zq_mobilefacenetv1_r4-8-16-4_128_softmax', help='output path')
#     parser.add_argument('--step', default=[8,16,24,28], type=list, help='ngpu use')
#
#     # 241
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#
#     # 230
#     # parser.add_argument('--pretrained', type=str, default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/'
#     #                      'sub_iccv_log/mobilefacenet_v1_softmax/check_point/checkpoint_25.pth',help='pretrained model')
#
#     parser.add_argument('--network', default='zq_mobilefacenet', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilenet_v3', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--fc_type', default='gnap', type=str, help='fc type')
#
#     # parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#     parser.add_argument('--block_setting', default=[4,8,16,4], type=list, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=200, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--cos_lr', type=bool, default=False, help='full logging')
#     parser.add_argument('--loss_type', type=str, default="softmax", help='arcface or softmax')
#
#     parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=8, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# mobilefacenet_softmax_finetune_arcface
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='sub_iccv_emore',
#                         help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore')
#
#     # 241
#     parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#                         help='train data folder')
#     # 246
#     # parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#     # 230
#     # parser.add_argument('--emore_folder',
#     #                     default='/home/dataset/Face/xz_FaceRecognition/iccv_challenge/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#
#     parser.add_argument('--gpu', default='2,3,6,7', help='ngpu use')
#
#     parser.add_argument('--outpath', default='./sub_iccv_log/mobilefacenetv1_softmax_finetune_arcface_bs512', help='output path')
#
#     parser.add_argument('--step', default=[8,16,24,28], type=list, help='ngpu use')
#
#     # 241
#     parser.add_argument('--pretrained', type=str, default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/'
#                         'sub_iccv_log/mobilefacenet_v1_softmax/check_point/checkpoint_25.pth', help='pretrained model')
#
#     # 230
#     # parser.add_argument('--pretrained', type=str, default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/'
#     #                      'sub_iccv_log/mobilefacenet_v1_softmax/check_point/checkpoint_25.pth',help='pretrained model')
#
#     # parser.add_argument('--network', default='zq_mobilefacenet', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilenet_v3', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#
#     parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[4,8,16,4], type=list, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--cos_lr', type=bool, default=False, help='full logging')
#     parser.add_argument('--loss_type', type=str, default="arcface", help='arcface or softmax')
#
#     parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=8, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='sub_iccv_1000',
#                         help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore, sub_iccv_1000')
#     # parser.add_argument('--data_mode', default='iccv_emore',
#     #                     help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore, sub_iccv_1000')
#     # parser.add_argument('--data_mode', default='sub_webface_0.1',
#     #                     help='train_data mode: emore, emore_debug, iccv_emore, sub_iccv_emore, sub_iccv_1000')
#
#     # 241
#     # parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#     # help='train data folder')
#
#     # 245
#     parser.add_argument('--emore_folder', default='/mnt/ssd/Datasets/faces/ms1m-retinaface-t1',
#                         help='train data folder')
#
#     # 170
#     # parser.add_argument('--emore_folder', default='/mnt/ssd/datasets/Faces/train/ms1m-retinaface-t1',
#     #                     help='train data folder')
#     # 230
#     # parser.add_argument('--emore_folder',
#     #                     default='/home/dataset/Face/xz_FaceRecognition/faces_webface_112x112', help='train data folder')
#
#     parser.add_argument('--gpu', default='0,1', help='ngpu use')
#
#     # parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_log_gdc'
#     #                                          '/mobilenfacenet_r3-8-16-4_gdc_512_988MFLOPs_softmax', help='output path')
#
#     # parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_log_gdc'
#     #                                          '/mobilenfacenet_r3-8-16-5_gdc_512_994MFLOPs_softmax', help='output path')
#
#     # parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_iccv_1000_log_gdc'
#     #                                          '/mobilenfacenet_r2-6-9-5_depth104_width1.1_gdc_5x5_se_512_995MFLOPs_softmax', help='output path')
#     # parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/sub_webface_0.1_train_dataset'
#     #                                          '/mobilenfacenetv2_r1-4-6-2_gdc_512_440MFLOPs_arcface_[15,25]_no_margin',
#     #                     help='output path')
#
#     parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/inception_resnet_sub_iccv_log/'
#                                              'InceptionResNetV2_512_baseline_softmax', help='output path')
#     # parser.add_argument('--outpath', default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/data/'
#     #                                          'pytorch_mobilenfacenet_r3-8-16-5_gdc_512_995MFLOPs_softmax_rec', help='output path')
#
#     parser.add_argument('--step', default=[15,25], type=list, help='ngpu use')
#     # parser.add_argument('--step', default=[], type=list, help='ngpu use')
#     # parser.add_argument('--step', default=[18,24,30], type=list, help='ngpu use')
#
#     # parser.add_argument('--pretrained', type=str,
#     #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/reload_mxnet.pkl', help='pretrained model')
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#
#     # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--network', default='InceptionResNetV2', help='specify network: r34, mobilefacenet')
#     # parser.add_argument('--network', default='mobilefacenet_v2', help='specify network: r34, mobilefacenet')
#
#     # parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[2, 8, 16, 4], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[3,8,16,5], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[4, 6, 2], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[], type=list, help='block num')
#     # parser.add_argument('--block_setting', default=[2, 8, 16, 8], type=list, help='block num')
#     parser.add_argument('--block_setting', default=[3, 8, 16, 5], type=list, help='block num')
#
#     parser.add_argument('--fc_type', default="gdc", type=str, help='fc_type')
#     # parser.add_argument('--fc_type', default="gnap", type=str, help='fc_type')
#
#     parser.add_argument('--width_mult', type=int, default=1.0, help='training epoch size.')
#
#     # parser.add_argument('--end-epoch', type=int, default=36, help='training epoch size.')
#     parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#     # parser.add_argument('--end-epoch', type=int, default=8, help='training epoch size.')
#
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
#
#     parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context: 512, 256')
#     # parser.add_argument('--batch-size', type=int, default=160, help='batch size in each context: 512, 256')
#
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--cos_lr', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--loss_type', type=str, default="softmax", help='arcface or softmax')
#     # parser.add_argument('--loss_type', type=str, default="arcface", help='arcface or softmax')
#
#     parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=8, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# for mobilenet_v3_dali
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='iccv_emore', help='train_data mode: emore, emore_debug, iccv_emore')
#
#     # 241
#     parser.add_argument('--emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
#                         help='train data folder')
#
#     parser.add_argument('--gpu', default='3,6', help='ngpu use')
#     parser.add_argument('--outpath', default='./iccv_log/mobilenet_v3_512', help='output path')
#     parser.add_argument('--step', default=[], type=list, help='ngpu use')
#
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#     parser.add_argument('--network', default='mobilenet_v3', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--block_setting', default=[], type=list, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=40, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=320, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#     parser.add_argument('--cos_lr', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--opt_level', type=str, default='01', help='torch.distributed')
#
#     parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=16, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# insightface mobilefacenet_v2
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='emore', help='train_data mode: emore, emore_debug')
#     # 186
#     # parser.add_argument('--emore_folder', default='/home/dataset/Face/FaceRecognition/'
#     #                                               'faces_ms1m-refine-v2_112x112/faces_emore', help='train data folder')
#     # 230
#     # parser.add_argument('--emore_folder', default='/home/dataset/Face/xz_FaceRecognition/'
#     #                                               'faces_ms1m-refine-v2_112x112/faces_emore', help='train data folder')
#
#     # # 241
#     parser.add_argument('--emore_folder', default='/home/datasets/Face/FaceRecognition/'
#                                                   'faces_ms1m-refine-v2_112x112/faces_emore', help='train data folder')
#
#     parser.add_argument('--gpu', default='2,3,4,5', help='ngpu use')
#     parser.add_argument('--outpath', default='./log/mobilefacenet_v2_res4-8-16-8_gdc_todo', help='output path')
#
#     parser.add_argument('--pretrained', type=str, default='', help='pretrained model')
#     parser.add_argument('--network', default='mobilefacenet_v2', help='specify network: r34, mobilefacenet_v1, mobilefacenet_v2')
#     parser.add_argument('--block_setting', default=[4,8,16,8], type=list, help='ngpu use')
#
#     parser.add_argument('--emb-size', type=int, default=256, help='embedding length')
#     parser.add_argument('--end-epoch', type=int, default=48, help='training epoch size.')
#     parser.add_argument('--step', default=[10,20,30,40], type=list, help='ngpu use')
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--batch-size', type=int, default=448, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--checkpoint', type=str, default='/home/xiezheng/program2019/insightface_DCP/'
#                         'insightface_v2/log/mobilefacenet_v2_res4-8-16-8_gdc/check_point/checkpoint_25.pth', help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=4, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# # insightface mobilefacenet_v1 nir_face_part_update: nir_data_1000
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='nir_face', help='train_data mode: emore, emore_debug, nir_face')
#     # 241
#     parser.add_argument('--emore_folder', default='/home/datasets/Face/FaceRecognition/nir_face_dataset', help='train data folder')
#
#     parser.add_argument('--gpu', default='0', help='ngpu use')
#     parser.add_argument('--outpath', type=str, default='./log/nir_face_part_update/mobilefacenet_v1_res1-4-6-2_nir_face_last3_lr0.01_epoch30', help='output path')
#     parser.add_argument('--step', default=[10,20], type=list, help='step')
#
#     parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/"
#                                                           "mobilefacenet_v1_pretrained_epoch34/checkpoint_34.pth", help='pretrained model')
#     parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#
#     parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#     parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=4e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=256, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--checkpoint', type=str, help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=4, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args


# # insightface mobilefacenet_v1 nir_face_part_update: nir_data_1224_190727
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='nir_face_1224', help='train_data mode: emore, emore_debug, nir_face, nir_face_1224')
#     # 241
#     parser.add_argument('--emore_folder', default='/home/datasets/Face/FaceRecognition/nir_face_dataset_1224',
#                         help='train data folder')
#
#     parser.add_argument('--gpu', default='4', help='ngpu use')
#
#     # mobilefacent_baseline
#     # parser.add_argument('--outpath', type=str,
#     #                     default='./nir_log/nir_face_part_update_1224_19027/mobilefacenet_v1_res1-4-6-2_last3_lr0.01_epoch30',
#     #                     help='output path')
#     # mobilefacent_p0.5
#     parser.add_argument('--outpath', type=str,
#                         default='./nir_log/nir_face_part_update_1224_19027_ICCV_without_p_fc/'
#                                 'mobilefacenet_v1_res1-4-6-2_p0.5_last3_lr0.1_epoch40',
#                         help='output path')
#     # parser.add_argument('--outpath', type=str,
#     #                     default='./nir_log/nir_face_part_update_1224_19027_ICCV_without_p_fc/'
#     #                             'mobilefacenet_v1_res1-4-6-2_p0.5_last3_lr0.01_epoch30',
#     #                     help='output path')
#     parser.add_argument('--last_type', type=str, default='last3', help='part update type')
#
#     # parser.add_argument('--step', default=[10,20], type=list, help='step')
#     parser.add_argument('--step', default=[10, 20, 30], type=list, help='step')
#
#     # 241
#     # mobilefacent_baseline
#     # parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/"
#     #                                                       "mobilefacenet_v1_pretrained_epoch34_lfw0.9955/checkpoint_34.pth",
#     #                     help='pretrained model')
#
#     # mobilefacent_p0.5_iccv_new
#     parser.add_argument('--pretrained', type=str, default="/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/"
#                         "mobilefacenet/2stage/dim128/finetune/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs512_e36_"
#                          "lr0.010_step[15, 25, 31]_bs512_cosine_finetune_20190804/check_point/checkpoint_033.pth",
#                         help='pretrained model')
#
#     # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--network', default='mobilefacenet_p0.5', help='specify network: r34, mobilefacenet')
#
#     parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#
#     # parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#     parser.add_argument('--end-epoch', type=int, default=40, help='training epoch size.')
#
#     parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
#     # parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
#     # parser.add_argument('--lr', type=float, default=0.002, help='start learning rate')
#
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=4e-4, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--checkpoint', type=str, help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=4, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args

# insightface mobilefacenet_v1 nir_face_part_update: nir_data_1224_190727 + int8_fintune
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train face network')
#     # general
#     parser.add_argument('--data_mode', default='nir_face_1224', help='train_data mode: emore, emore_debug, nir_face, nir_face_1224')
#     # 241
#     parser.add_argument('--emore_folder', default='/home/datasets/Face/FaceRecognition/nir_face_dataset_1224',
#                         help='train data folder')
#
#     parser.add_argument('--gpu', default='6', help='ngpu use')
#     # mobilefacent_baseline
#     # parser.add_argument('--outpath', type=str,
#     #                     default='./nir_log/nir_face_part_update_1224_19027/mobilefacenet_v1_res1-4-6-2_last3_lr0.01_epoch30',
#     #                     help='output path')
#
#     # mobilefacent_p0.5
#     parser.add_argument('--outpath', type=str,
#                         default='./nir_log/nir_face_part_update_1224_19027/int8_train/'
#                                 'mobilefacenet_v1_res1-4-6-2_p0.5_nir_last1_lr0.01_epoch30_int8_train_wd4e-5',
#                         help='output path')
#
#     parser.add_argument('--step', default=[10,20], type=list, help='step')
#
#     # 241
#     # mobilefacent_baseline
#     # parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/"
#     #                                                       "mobilefacenet_v1_pretrained_epoch34_lfw0.9955/checkpoint_34.pth",
#     #                     help='pretrained model')
#
#     # mobilefacent_p0.5
#     parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/"
#                         "nir_log/nir_face_part_update_1224_19027/int8_train/log_ft_mobilefacenet_v1_p0.5_ms1m_v2_bs256_"
#                         "e18_lr0.001_step[]_p0.5_cos_lr_w_a_int8_finetune_20190723/checkpoint_017.pth",
#                         help='pretrained model')
#
#     parser.add_argument('--last_type', type=str, default='last1', help='part update type')
#     # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
#     parser.add_argument('--network', default='mobilefacenet_p0.5', help='specify network: r34, mobilefacenet')
#
#     parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
#     parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
#
#     parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
#     # parser.add_argument('--lr', type=float, default=0.002, help='start learning rate')
#
#     parser.add_argument('--optimizer', default='sgd', help='optimizer')
#     parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
#     parser.add_argument('--mom', type=float, default=0.9, help='momentum')
#     parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
#     parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context: 512, 256')
#     parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
#     parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
#     parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
#     parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
#     parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
#     parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
#     parser.add_argument('--full-log', type=bool, default=False, help='full logging')
#
#     parser.add_argument('--checkpoint', type=str, help='checkpoint')
#     parser.add_argument('--num_workers', type=int, default=4, help='num workers')
#     parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
#     parser.add_argument('--print_freq', type=int, default=10, help='print freq')
#
#     args = parser.parse_args()
#     ensure_folder(args.outpath)
#     return args


# insightface mobilefacenet_v1 nir_face: learn for scratch
def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data_mode', default='nir_face', help='train_data mode: emore, emore_debug, nir_face')
    # 241
    parser.add_argument('--emore_folder', default='/home/datasets/Face/FaceRecognition/nir_face_dataset', help='train data folder')
    parser.add_argument('--gpu', default='5', help='ngpu use')
    parser.add_argument('--end-epoch', type=int, default=28, help='training epoch size.')
    parser.add_argument('--step', default=[6,12,18,24], type=list, help='step')
    parser.add_argument('--outpath', type=str, default='./nir_scratch_log/nir_face_mobilefacenet_v1_res1-4-6-2_lr0.1_epoch28', help='output path')
    parser.add_argument('--pretrained', type=str, default="", help='pretrained model')
    parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
    parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')

    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size in each context: 512, 256')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--full-log', type=bool, default=False, help='full logging')

    parser.add_argument('--checkpoint', type=str, help='checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--grad_clip', type=float, default=5., help='grad clip')
    parser.add_argument('--print_freq', type=int, default=10, help='print freq')

    args = parser.parse_args()
    ensure_folder(args.outpath)
    return args


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger

def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.makedirs(folder)

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr[0]

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


# def separate_bn_paras(modules):
#     if not isinstance(modules, list):
#         modules = [*modules.modules()]
#     paras_only_bn = []
#     paras_wo_bn = []
#     paras_bias = []
#     for layer in modules:
#         if 'model' in str(layer.__class__):
#             continue
#         if 'container' in str(layer.__class__):
#             continue
#         else:
#             if 'batchnorm' in str(layer.__class__):
#                 paras_only_bn.extend([*layer.parameters()])
#             else:
#                 for name, param in layer.named_parameters():
#                     if 'bias' in name:
#                         print('bias: {}'.format(name))
#                         print(param)
#                         paras_bias.append(param)
#                     else:
#                         paras_wo_bn.append(param)
#                 # paras_wo_bn.extend([*layer.parameters()])
#     return paras_only_bn, paras_wo_bn, paras_bias



def update_lr(epoch, optimizer, args):
    """
        Update learning rate of optimizers
        :param epoch: index of epoch
    """
    gamma = 0
    for step in args.step:
        if epoch + 1.0 > int(step):
            gamma += 1
    lr = args.lr * math.pow(0.1, gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def update_lr(epoch, optimizer, args):
#     """
#         Update learning rate of optimizers
#         :param epoch: index of epoch
#     """
#     gamma = 0
#     for step in args.step:
#         if epoch + 1.0 > int(step):
#             gamma += 1
#     lr = args.lr * math.pow(0.1, gamma)
#     for param_group in optimizer.param_groups:
#         if hasattr(param_group, 'is_bias'):
#             param_group['lr'] = lr * 2
#         else:
#             param_group['lr'] = lr




# if __name__ == "__main__":
#     args = parse_args()
#     print(args.step, type(args.step))
#     for step in args.step:
#         print(step)
