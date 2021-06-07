#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 10:11
# @Author  : xiezheng
# @Site    : 
# @File    : checkpoint.py
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


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.makedirs(folder)

# def save_checkpoint(args, epoch, model, metric_fc, optimizer, acc, is_best):
def save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, acc):
    check_point_params = {}
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    if isinstance(metric_fc, nn.DataParallel):
        check_point_params["metric_fc"] = metric_fc.module.state_dict()
    else:
        check_point_params["metric_fc"] = metric_fc.state_dict()

    check_point_params["lfw_best_acc"] = acc
    check_point_params["optimizer"] = optimizer
    # check_point_params["scheduler"] = scheduler
    check_point_params['epoch'] = epoch

    output_path = os.path.join(args.outpath, "check_point")
    ensure_folder(output_path)
    filename = 'checkpoint_{:d}.pth'.format(epoch)
    torch.save(check_point_params, os.path.join(output_path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # if is_best:
    #     torch.save(check_point_params["model"],
    #                os.path.join(output_path, 'epoch_{:d}_best_val_tpr_{:.3f}_checkpoint.pth'.format(epoch, acc)))


def save_checkpoint_soft(args, epoch, model, rgb_metric_fc, nir_metric_fc, optimizer, scheduler, acc):
    check_point_params = {}
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    if isinstance(rgb_metric_fc, nn.DataParallel):
        check_point_params["rgb_metric_fc"] = rgb_metric_fc.module.state_dict()
    else:
        check_point_params["rgb_metric_fc"] = rgb_metric_fc.state_dict()

    if isinstance(nir_metric_fc, nn.DataParallel):
        check_point_params["nir_metric_fc"] = nir_metric_fc.module.state_dict()
    else:
        check_point_params["nir_metric_fc"] = nir_metric_fc.state_dict()

    check_point_params["nir_top1_accs"] = acc
    check_point_params["optimizer"] = optimizer
    check_point_params["scheduler"] = scheduler
    check_point_params['epoch'] = epoch

    output_path = os.path.join(args.outpath, "check_point")
    ensure_folder(output_path)
    filename = 'checkpoint_{:d}_nir_acc{:.3f}.pth'.format(epoch, acc)
    torch.save(check_point_params, os.path.join(output_path, filename))


def load_checkpoint(checkpoint_path, model, metric_fc, logger):
    if checkpoint_path is not None:
        check_point_params = torch.load(checkpoint_path)
        model_state = check_point_params["model"]
        metric_fc_state = check_point_params["metric_fc"]
        start_epoch = check_point_params['epoch']
        optimizer = check_point_params["optimizer"]
        scheduler = check_point_params["scheduler"]
        lfw_best_acc = check_point_params["lfw_best_acc"]

        model = load_state(model, model_state, logger)
        metric_fc = load_state(metric_fc, metric_fc_state, logger)
        return model, metric_fc, start_epoch, optimizer, scheduler, lfw_best_acc
        # return model, metric_fc, start_epoch, optimizer, lfw_best_acc


def load_state(model, state_dict, logger):
    """
    load state_dict to model
    :params model:
    :params state_dict:
    :return: model
    """
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    logger.info("load state finished !!!")
    return model