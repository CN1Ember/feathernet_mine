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
sys.path.insert(1,'/home/lidaiyuan/code/insightface_dcp/')

import argparse

from dcp.models.FaceFeatherNet import FaceFeatherNetA


def save_weight_table(net, calibration_path):
    print("------save the scale table----")
    calibration_file = open(calibration_path, 'w')
    print("------quant weight----")
    weight_table = OrderedDict()
    for name, layer in net.named_modules():
        # find the convolution layers to get out the weight_scale
        if isinstance(layer, nn.Conv2d):
            per_ch = []
            for i in range(layer.weight.shape[0]):
                data_ = layer.weight[i].data
                maxv = data_.abs().max()
                scale = 127.0 / maxv.cpu().numpy()
                if name not in weight_table.keys():
                    weight_table[name] = per_ch
                    weight_table[name].append(scale)
                else:
                    weight_table[name].append(scale)
    print("------quant weight done----")

    # save the weight blob scales
    for key, value_list in weight_table.items():

        # remove 'net.'
        if "net" in key:
            key = key.replace('net.', '')

        calibration_file.write(key + " ")
        for value in value_list:
            calibration_file.write(str(value) + " ")
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
    net = FaceFeatherNetA()

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
    # parser.add_argument('--model_path', type=str, help='model_path', default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_74_rm_block.pth.tar')
    # parser.add_argument('--calibration_path', type=str, help='calibration_path', default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_74_rm_block.table')

    parser.add_argument('--model_path', type=str, help='model_path',
                        default='/home/lidaiyuan/feathernet2020/FeatherNet/checkpoints/FaceFeatherNetA_nir_1104_ft_quan/_172_best.pth.tar')
    parser.add_argument('--calibration_path', type=str, help='calibration_path',
                        default='/home/lidaiyuan/feathernet2020/FeatherNet/checkpoints/FaceFeatherNetA_nir_1104_ft_quan/_172_best_block_w.table')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


