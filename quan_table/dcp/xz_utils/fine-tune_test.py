#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 11:35
# @Author  : xiezheng
# @Site    : 
# @File    : fine-tune_test.py

import torch
import os
import bcolz
import numpy as np

from dcp.utils.verifacation import evaluate, l2_norm
from dcp.utils.model_analyse import ModelAnalyse
from dcp.utils.logger import get_logger

from dcp.models.insightface_resnet_pruned import pruned_LResNet34E_IR
from dcp.models.insightface_resnet import LResNet34E_IR, LResNet18E_IR
from dcp.models.mobilefacenet import Mobilefacenet

from dcp.models.insightface_mobilefacenet import MobileFaceNet, ZQMobileFaceNet

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return carray, issame

def get_ms1m_testloader(data_path='/home/dataset/Face/xz_FaceRecognition/faces_ms1m-refine-v2_112x112/faces_emore/'):
    # test_loader
    test_loader = []
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    test_loader.append({'agedb_30':agedb_30, 'agedb_30_issame':agedb_30_issame})
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    test_loader.append({'cfp_fp': cfp_fp, 'cfp_fp_issame': cfp_fp_issame})
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    test_loader.append({'lfw': lfw, 'lfw_issame': lfw_issame})
    return test_loader

def val_evaluate(model, carray, issame, emb_size=512, batch_size=512, nrof_folds=5):
    idx = 0
    embeddings = np.zeros([len(carray), emb_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            out = model(batch.cuda())
            embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
            idx += batch_size

        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            out = model(batch.cuda())
            embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    return accuracy.mean()

def face_val(pruned_model, testdata_path):

    pruned_model.eval()
    val_loader = get_ms1m_testloader(testdata_path)
    print("testdata prepare!!!")

    with torch.no_grad():
        lfw_accuracy = val_evaluate(pruned_model, val_loader[-1]['lfw'], val_loader[-1]['lfw_issame'])
        cfp_fp_accuracy = val_evaluate(pruned_model, val_loader[-2]['cfp_fp'], val_loader[-2]['cfp_fp_issame'])
        agedb_30_accuracy = val_evaluate(pruned_model, val_loader[-3]['agedb_30'], val_loader[-3]['agedb_30_issame'])

    print("|===>Validation lfw acc: {:4f}".format(lfw_accuracy))
    print("|===>Validation cfp_fp acc: {:4f}".format(cfp_fp_accuracy))
    print("|===>Validation agedb_30 acc: {:4f}".format(agedb_30_accuracy))



if __name__ == '__main__':

    pruned_rate = 0.5
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    save_path = './finetune-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path, "finetune-test")

    test_input = torch.randn(1, 3, 112, 112).cuda()
    # model = pruned_LResNet34E_IR(pruning_rate=pruned_rate)
    # model = LResNet34E_IR()
    # model = LResNet18E_IR()

    # model = Mobilefacenet()

    # model = MobileFaceNet(embedding_size=128, blocks=[1,4,6,2])
    # model = MobileFaceNet(256, blocks=[2, 8, 16, 4])                  # 7.61 M
    # model = MobileFaceNet(embedding_size=256, blocks=[4, 8, 16, 8])

    # model = ZQMobileFaceNet(embedding_size=256, blocks=[4,8,16,4])
    model = ZQMobileFaceNet(embedding_size=512, blocks=[4, 8, 16, 4])
    # model = ZQMobileFaceNet(embedding_size=512, blocks=[4, 8, 16, 8])
    # model = ZQMobileFaceNet(embedding_size=512, blocks=[8, 16, 32, 8])

    model = model.cuda()

    model_analyse = ModelAnalyse(model, logger)
    model_analyse.flops_compute(test_input)
    model_analyse.params_count()

    # 230
    # testdata_path = '/home/dataset/Face/xz_FaceRecognition/faces_ms1m-refine-v2_112x112/faces_emore/'
    # # # r34_lr0.1_checkpoint27_lfw_0.9975
    # # check_point_params = torch.load('/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    # #                                 'log_ft_LResnetxE-IR_ms1m_v2_bs512_e48_lr0.100_step[10, 20, 30, 40]_20190527_dcp0.5_93.10_todo/'
    # #                                 'check_point/best_model_without_aux_fc.pth')
    #
    # # 241
    # # testdata_path = '/home/datasets/Face/FaceRecognition/faces_ms1m-refine-v2_112x112/faces_emore/'
    # # r34_lr0.1_checkpoint27_lfw_0.9975
    #
    # model_list =[
    #     # lr = 0.1
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs512_e48_lr0.100_step[10, 20, 30, 40]_20190527_dcp0.5_93.10_todo/check_point/checkpoint_037.pth',
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs512_e48_lr0.100_step[10, 20, 30, 40]_20190527_dcp0.5_93.10_todo/check_point/checkpoint_038.pth',
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs512_e48_lr0.100_step[10, 20, 30, 40]_20190527_dcp0.5_93.10_todo/check_point/checkpoint_047.pth',
    #     # lr = 0.01
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs512_e28_lr0.010_step[8, 16, 24]_20190527_dcp0.5_93.10/check_point/checkpoint_014.pth',
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs480_e28_lr0.010_step[8, 16, 24]_20190527_dcp0.5_93.10_todo/check_point/checkpoint_025.pth',
    #     '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/fine-tuning/'
    #     'log_ft_LResnetxE-IR_ms1m_v2_bs480_e28_lr0.010_step[8, 16, 24]_20190527_dcp0.5_93.10_todo/check_point/checkpoint_027.pth'
    # ]
    #
    # # check_point_params = torch.load('/home/xiezheng/program2019/insightface_DCP/dcp/insightface_dcp_log/best_r34_without_aux_fc_checkpoint27.pth')
    # for i in range(len(model_list)):
    #     print("model_path={}".format(model_list[i]))
    #     check_point_params = torch.load(model_list[i])
    #     model_state = check_point_params['model']
    #     # print(model)
    #     model.load_state_dict(model_state)
    #     print("model load finished !!!")
    #     model = model.cuda()
    #     # model.eval()
    #     face_val(model, testdata_path)