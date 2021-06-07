#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 16:43
# @Author  : xiezheng
# @Site    : 
# @File    : adjust_table.py


import sys
import argparse

QUANTIZE_NUM = 127.0

def replace_weight_name(weight_path, calibration_path):
    count = 1
    calibration_file = open(calibration_path, 'a+')
    for line in open(weight_path, 'r'):
        line = line.strip()
        title = 'conv{}_param_0'.format(count)
        str_arr = line.split(' ', maxsplit=1)

        print('{}-->{}'.format(str_arr[0], title))
        new_line = '{} {}\n'.format(title, str_arr[-1])
        calibration_file.write(new_line)
        count += 1
    calibration_file.close()


def replace_activation_name(activation_path, calibration_path):
    count = 1
    calibration_file = open(calibration_path, 'a+')
    for line in open(activation_path, 'r'):
        line = line.strip()
        title = 'conv{}'.format(count)
        str_arr = line.split(' ', maxsplit=1)

        print('{}-->{}'.format(str_arr[0], title))

        threshold = float(str_arr[-1])
        scale = QUANTIZE_NUM / threshold
        new_line = '{} {}\n'.format(title, str(scale))
        calibration_file.write(new_line)
        count += 1
    calibration_file.close()


def main(args):
    print(args)
    replace_weight_name(args.weight_path, args.calibration_path)
    replace_activation_name(args.activation_path, args.calibration_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001_int8
    # parser.add_argument('--weight_path', type=str, help='weight_path',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_checkpoint25_weight.table')
    # parser.add_argument('--activation_path', type=str, help='activation_path',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/qqc_sq_finetune/iccv_mobilefacenet_p0.5_our_sq_activation.table')
    # parser.add_argument('--calibration_path', type=str, help='calibration_path',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/iccv_mf_p0.5_wa.table')

    #
    parser.add_argument('--weight_path', type=str, help='weight_path',
                        default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_nir_120_rm_block_w.table')
    parser.add_argument('--activation_path', type=str, help='activation_path',
                        default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_nir_120_rm_block_act.table')
    parser.add_argument('--calibration_path', type=str, help='calibration_path',
                        default='/home/lidaiyuan/feathernet2020/model/caffemodel/_nir_120_rm_block.table')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))