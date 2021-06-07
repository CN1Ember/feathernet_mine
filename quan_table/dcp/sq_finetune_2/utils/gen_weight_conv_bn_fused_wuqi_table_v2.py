#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/10 18:58
# @Author  : xiezheng
# @Site    : 
# @File    : gen_weight_table.py
import torch
from torch import nn
from collections import OrderedDict

import sys
sys.path.insert(1,'/home/xiezheng/lidaiyuan/insightface_dcp/')
import argparse

from dcp.models.mobilefacenet_pruned import pruned_Mobilefacenet
from dcp.models.mobilefacenetv2_width_wm_fused import Mobilefacenetv2_width_wm_fuse
import numpy as np

import math
import time


def save_weight_table(net, calibration_path):
    print("------save the scale table----")
    calibration_file = open(calibration_path, 'w')
    print("------quant weight----")

    weight_table = OrderedDict()
    bias_table = OrderedDict()

    layer_order_list = list(net.named_modules())
    layer_name_list = []
    for i in range(len(layer_order_list)):
        layer_name_list.append(layer_order_list[i][0])

    for name, layer in net.named_modules():
        # find the convolution layers to get out the weight_scale
        if isinstance(layer, nn.Conv2d):
            per_ch_w = []
            per_ch_b = []

            next_layer_id = layer_name_list.index(name) + 1
            bn_model = layer_order_list[next_layer_id][1]
            fused_weight = layer.weight.data
            fused_bias = layer.bias.data

            # print(bn_mean)
            
            for i in range(fused_weight.shape[0]):
                # print(fused_weight[i].shape)
                data_ = fused_weight[i]
                maxv = data_.abs().max()
                # print(fused_bias[i])
                if maxv.cpu().numpy() == 0:
                    scale_frac = 0
                    print(name)
                else:
                    scale_frac = math.floor(math.log(127.0 / maxv.cpu().numpy(),2))


                # print(fused_bias[i].abs().cpu())
                if fused_bias[i].abs().cpu() == 0:
                    print(name)
                    bias_frac = 0
                else:
                    bias_frac = math.floor(math.log(127.0 / fused_bias[i].abs().cpu().numpy(),2))

                if name not in weight_table.keys():
                    weight_table[name] = per_ch_w
                    weight_table[name].append(scale_frac)
                else:
                    weight_table[name].append(scale_frac)
                
                if name not in bias_table.keys():
                    bias_table[name] = per_ch_b
                    bias_table[name].append(bias_frac)
                else:
                    bias_table[name].append(bias_frac)

            # print(bias_table[name])
            # time.sleep(1)

					
        # find the activation layers to get out the weight_scale
        # if isinstance(layer, nn.BatchNorm2d):
        #     per_ch_1 = []
        #     per_ch_2 = []
        #     per_ch_3 = []
        #     per_ch_4 = []

        #     for i in range(layer.weight.shape[0]):
        #         data_ = layer.weight[i].data
        #         bias_data_ = layer.bias[i].data
        #         moving_mean_ = layer.running_mean[i].data
        #         moving_var_ = layer.running_var[i].data


        #         maxv = data_.abs()
        #         scale = math.floor(math.log(127.0 / maxv.cpu().numpy(),2))

        #         maxv = bias_data_.abs()
        #         bias = math.floor(math.log(127.0 / maxv.cpu().numpy(),2))

        #         # max_mean = moving_mean_.abs()
        #         # max_var = moving_var_.abs()
        #         #转换成物奇bn格式
        #         var = 1 / moving_var_.cpu().numpy()
        #         mean = - moving_mean_.cpu().numpy() / moving_var_.cpu().numpy()
                
        #         mean = math.floor(math.log(127.0 / np.abs(mean),2))
        #         var = math.floor(math.log(127.0 / np.abs(var),2))

                
        #         if name not in weight_table.keys():
        #             weight_table[name] = per_ch_1
        #             weight_table[name].append(scale)
        #         else:
        #             weight_table[name].append(scale)

        #         if name not in bias_table.keys():
        #             bias_table[name] = per_ch_2
        #             bias_table[name].append(bias)
        #         else:
        #             bias_table[name].append(bias)

        #         if name not in moving_mean_table.keys():
        #             moving_mean_table[name] = per_ch_3
        #             moving_mean_table[name].append(mean)
        #         else:
        #             moving_mean_table[name].append(mean)

        #         if name not in moving_var_table.keys():
        #             moving_var_table[name] = per_ch_4
        #             moving_var_table[name].append(var)
        #         else:
        #             moving_var_table[name].append(var)

        # if name in moving_var_table.keys(): 
        #     print(moving_mean_table[name])
        #     print(moving_var_table[name])


        # find the activation layers to get out the weight_scale
        if isinstance(layer, nn.PReLU):
            per_ch = []
            for i in range(layer.weight.shape[0]):
                data_ = layer.weight[i].data
                maxv = data_.abs()
                scale = int(math.log(127.0 / maxv.cpu().numpy(),2))
                
                if name not in weight_table.keys():
                    weight_table[name] = per_ch
                    weight_table[name].append(scale)
                else:
                    weight_table[name].append(scale)
    print("------quant weight done----")

    # save the weight blob scales
    for key, value_list in weight_table.items():

        # remove 'net.'
        key_str = key
        if "net" in key:
            key = key.replace('net.', '')

        calibration_file.write(key + '_w' + " ")
        for value in value_list:
            calibration_file.write(str(value) + " ")


        if key_str in bias_table.keys():
            calibration_file.write("\n")
            calibration_file.write(key + '_bias' + " ")
            for value in bias_table[key_str]: 
                calibration_file.write(str(value) + " ")

        # if key_str in moving_mean_table.keys():
        #     calibration_file.write("\n")
        #     calibration_file.write(key + '_moving_mean' + " ")
        #     # print('key_str',moving_mean_table[key_str])
        #     for value in moving_mean_table[key_str]:            
        #         calibration_file.write(str(value) + " ")
      
        # if key_str in moving_var_table.keys():
        #     calibration_file.write("\n")
        #     calibration_file.write(key + '_moving_var' + " ")  
        #     # print('key_str',moving_var_table[key_str])
        #     for value in moving_var_table[key_str]:
        #         calibration_file.write(str(value) + " ")

        calibration_file.write("\n")

    # save the botom blob scales
    # for key, value in scale_table.items():
    #     if key == 'quant' or key == 'test' or key == 'quant_bp_coef':
    #         continue
    #     name = key.split('net.')
    #     calibration_file.write(name[1] + " " + str(value) + "\n")
    #     print("%-24s : %f" % (name[1], value))

    calibration_file.close()
    print("====> Save calibration table success...")

def main(args):
    print(args)
    net = Mobilefacenetv2_width_wm_fuse(embedding_size=128, pruning_rate=0)

    check_point_params = torch.load(args.model_path)
    model_state = check_point_params['model']
    print(net)
    net.load_state_dict(model_state)
    print("net load success !!!")

    save_weight_table(net, args.calibration_path)
    print('Success get weight scale for int8-quantization !!!')

    # net.eval()
    # input = torch.ones([1, 3, 112, 112])
    # out = net(input)
    # print('out={}'.format(out))
    print(args)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, help='model_path', default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq-step[20]/check_point/checkpoint_25.pth')
    # parser.add_argument('--calibration_path', type=str, help='calibration_path', default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_checkpoint25_weight.table')

    parser.add_argument('--model_path', type=str, help='model_path',
                        default='/home/xiezheng/lidaiyuan/Face_integration/train_log/Soft-finetuning-p0-noquan-lr0.0001-[30]-1224-data_aug-ft128/check_point/checkpoint_1_fused.pth')
    parser.add_argument('--calibration_path', type=str, help='calibration_path',
                        default='/home/xiezheng/lidaiyuan/Face_integration/train_log/Soft-finetuning-p0-noquan-lr0.0001-[30]-1224-data_aug-ft128/check_point/checkpoint_1_fused.pth_w.table')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


