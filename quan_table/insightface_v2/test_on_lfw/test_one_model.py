#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 21:40
# @Author  : xiezheng
# @Site    : 
# @File    : test_model.py

import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn

from prefetch_generator import BackgroundGenerator

from insightface_v2.utils.model_analyse import ModelAnalyse
from insightface_v2.utils.logger import get_logger
from insightface_v2.insightface_data.data_pipe import get_train_loader, get_all_val_data, get_test_loader
from insightface_v2.utils.verifacation import evaluate
from insightface_v2.model.focal_loss import FocalLoss
from insightface_v2.model.models import resnet34, resnet_face18, ArcMarginModel
from insightface_v2.utils.utils import parse_args, AverageMeter, clip_gradient, \
    accuracy, get_logger, get_learning_rate, separate_bn_paras, update_lr
from insightface_v2.utils.checkpoint import save_checkpoint, load_checkpoint
from insightface_v2.utils.ver_data import val_verification


from torchvision import transforms
from torch.utils.data import DataLoader

from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v3_width import Mobilefacenetv2_v3_width
from insightface_v2.model.models import MobileFaceNet
from insightface_v2.model.mobilefacenet_pruned import  pruned_Mobilefacenet


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def de_preprocess(tensor):
    return tensor * 0.501960784 + 0.5

hflip = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.functional.hflip,
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def val_evaluate(model, carray, issame, nrof_folds=10, use_flip=True, emb_size=512, batch_size=200):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), emb_size])
    print('use_flip={}'.format(use_flip))
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            out = model(batch.cuda())
            if use_flip:
                fliped_batch = hflip_batch(batch)
                fliped_out = model(fliped_batch.cuda())
                out = out + fliped_out
                # print('flip success !!')

            embeddings[idx:idx + batch_size] = l2_norm(out).cpu()  # xz: add l2_norm
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            out = model(batch.cuda())

            if use_flip:
                fliped_batch = hflip_batch(batch)
                fliped_out = model(fliped_batch.cuda())
                out = out + fliped_out

            embeddings[idx:] = l2_norm(out).cpu()  # xz: add l2_norm
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean()


def board_val(logger, db_name, accuracy, best_threshold, step):
    logger.info('||===>>Test Epoch: [[{:d}]\t\tVal:{}_accuracy={:.4f}%\t\tbest_threshold={:.4f}'
                .format(step, db_name, accuracy*100, best_threshold))
    # writer.add_scalar('val_{}_accuracy'.format(db_name), accuracy*100, step)
    # writer.add_scalar('val_{}_best_threshold'.format(db_name), best_threshold, step)
    # writer.add_image('val_{}_roc_curve'.format(db_name), roc_curve_tensor, step)


def test_flip(model, epoch, emb_size):
    agedb_accuracy, agedb_best_threshold = val_evaluate(model, agedb_30, agedb_30_issame, emb_size=emb_size)
    board_val(logger, 'agedb_30', agedb_accuracy, agedb_best_threshold, epoch)
    lfw_accuracy, lfw_best_threshold = val_evaluate(model, lfw, lfw_issame, emb_size=emb_size)
    board_val(logger, 'lfw', lfw_accuracy, lfw_best_threshold, epoch)
    cfp_accuracy, cfp_best_threshold = val_evaluate(model, cfp_fp, cfp_fp_issame, emb_size=emb_size)
    board_val(logger, 'cfp_fp', cfp_accuracy, cfp_best_threshold, epoch)



if __name__ == "__main__":

    # only widen: model layers_num = 49 ; model size=9.73 MB; model flops=995.61 M
    # emb_size = 512
    # model = Mobilefacenetv2_v3_width(embedding_size=emb_size, width_mult=1.315)

    emb_size = 128
    # model = MobileFaceNet(embedding_size=128, blocks = [1,4,6,2])
    # model = pruned_Mobilefacenet(pruning_rate=0.25)
    model = pruned_Mobilefacenet(pruning_rate=0.5)


    outpath = '/home/xiezheng/program2019/insightface_DCP/dcp/insightface_dcp_log/' \
              'fine-tuning/test_pami_face_flip_result'
    # # 246
    # emore_folder = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1'
    # 241
    # iccv_ms1m
    # emore_folder = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1'
    # ms1mv2
    emore_folder = '/mnt/ssd/faces/faces_emore'

    # writer = SummaryWriter(outpath)
    logger = get_logger(outpath, 'insightface')
    # logger.info("model={}".format(model))
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_all_val_data(emore_folder)
    logger.info("train dataset and val dataset are ready!")


    # model_path = '/home/xiezheng/program2019/insightface_DCP/' \
    #              'mobilefacenet_v1_pretrained_epoch34_lfw0.9955/checkpoint_34.pth'

    # model_path = '/home/xiezheng/program2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/' \
    #              'log_ft_mobilefacenet_v1_ms1m_v2_bs512_e28_lr0.010_step[6, 12, 18, 24]_20190603_conv3_p0.25_99.03/' \
    #              'check_point/checkpoint_022.pth'
    #
    model_path = '/home/xiezheng/program2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/' \
                 'log_ft_mobilefacenet_v1_ms1m_v2_bs768_e28_lr0.010_step[6, 12, 18, 24]_20190603_conv3_p0.5_97.00/' \
                 'check_point/checkpoint_025.pth'

    chechpoint = torch.load(model_path)
    model.load_state_dict(chechpoint['model'])
    print('model_path={}'.format(model_path))
    logger.info("model load success!!!")
    model = model.cuda()
    test_flip(model, 0, emb_size)







