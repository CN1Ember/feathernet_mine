#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 20:51
# @Author  : xiezheng
# @Site    : 
# @File    : get_model.py


import pickle
import torch
import torch.nn as nn
# from torchvision.models import resnet50, resnet101, inception_v3
from model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm
from utils.attentionTransfer_util.arcface import arcface
from ncnn_quantization import get_activation_value, replace


def get_base_model(base_model_name, emb_size, logger):
    if base_model_name == 'mobilefacenet_p0.5':
        model = Mobilefacenetv2_width_wm(embedding_size=emb_size, pruning_rate=0.5)    # 128
    else:
        assert False, logger.info("invalid base_model_name={}".format(base_model_name))
    return model


def load_state(model_source, model_target, source_metric_fc, target_metric_fc, pretrain_path, outpath, logger):
    if pretrain_path:
        state = torch.load(pretrain_path)
        model_state = state["model"]
        if 'asat+acat' in outpath:
            source_metric_fc_state = state['metric_fc']
    else:
        assert False, logger.info('pretrain_path is None!!!')
    model_source.load_state_dict(model_state)
    model_target.load_state_dict(model_state)
    #   source_metric_fc.load_state_dict(source_metric_fc_state)
    if 'asat+acat' in outpath:
        target_metric_fc.load_state_dict(source_metric_fc_state)
        logger.info('asat+acat: target_metric_fc load success!!!')
    return model_source, model_target, source_metric_fc, target_metric_fc


def get_model(base_model_name, pretrain_path, emb_size, target_class_num, outpath, table_path, logger):
    model_source = get_base_model(base_model_name, emb_size, logger)
    model_target = get_base_model(base_model_name, emb_size, logger)

    source_metric_fc = arcface(num_classes=85742, emb_size=emb_size)   # For ms1m_v2: 85742
    target_metric_fc = arcface(num_classes=target_class_num, emb_size=emb_size)

    model_source, model_target, source_metric_fc, target_metric_fc = \
        load_state(model_source, model_target, source_metric_fc, target_metric_fc, pretrain_path, outpath, logger)

    for param in model_source.parameters():
        param.requires_grad = False
    # source_model 只用来做预测
    model_source.eval()
    # replace model quantization
    activation_value_list = get_activation_value(table_path)
    model_target = replace(model_target, activation_value_list, logger)

    model_target = model_target.cuda()
    model_source = model_source.cuda()
    target_metric_fc = target_metric_fc.cuda()

    logger.info('get model_source and model_target = {}'.format(base_model_name))
    logger.info('source_metric_fc={}, target_metric_fc={}'.format(source_metric_fc, target_metric_fc))

    # assert False
    return model_source, source_metric_fc, model_target, target_metric_fc



if __name__ == '__main__':
    # model = inception_v3(pretrained=False, aux_logits=False)
    # model = resnet101(pretrained=False)
    #
    # count = 1
    # for name, param in model.named_parameters():
    #     print('count={}, name={}'.format(count, name))
    #     # if 'conv' in name or 'downsample.0' in name:
    #     #     print('count={}\tname={}\tsize={}'.format(count, name,
    #     #                                                      param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3]))
    #     count += 1
    print('')
