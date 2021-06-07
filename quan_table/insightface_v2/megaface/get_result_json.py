#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 17:42
# @Author  : xiezheng
# @Site    : 
# @File    : get_result_json.py


import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath
import argparse
import sys

def read_json(json_path):
    with open(json_path, 'r') as f:
        temp = json.loads(f.read())
        # print(type(temp), temp)
        # print(type(temp['cmc']), temp['cmc'])
        # print(type(temp['roc']),temp['roc'])
        cmc = np.array(temp['cmc'])
        roc = np.array(temp['roc'])
        # print(type(cmc), cmc)
        # print(type(roc), roc)
    return cmc, roc


def cal_cmc(cmc):
    # fig = plt.figure()
    # plt.plot(cmc[0]+1, cmc[1]*100, color='Blue',label='CMC')
    # plt.grid(linestyle='--', linewidth=1)
    # plt.xlabel('Rank')
    # plt.ylabel('Identification Rate %')
    # plt.title('Identification @ 1e6 distractors ={:.5f}%'.format(cmc[1][0]*100))
    # plt.legend(loc="lower right")
    # plt.show()
    # fig.savefig('cmc.pdf')
    print('Identification @ 1e6 distractors ={:.5f}%'.format(cmc[1][0]*100))


def cal_roc(roc, x_labels=[10**-6]):
    fpr, tpr = roc[0], roc[1]
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr

    tpr_result = 0
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_result =  tpr[min_index]

    # plt.plot(fpr, tpr, color='red', label='ROC')
    # plt.grid(linestyle='--', linewidth=1)
    # # plt.xlim([10 ** -6, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Verification @ 1e-6 ={:.5f}%'.format(tpr_result * 100))
    # plt.legend(loc="lower right")
    # plt.show()
    # fig.savefig('roc.pdf')
    print('Verification @ 1e-6 ={:.5f}%'.format(tpr_result * 100))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, help='', default='/home/dataset/xz_datasets/Megaface/'
    'pytorch_mobilefacenet/pytorch_mobilefacenet_v1_model/pytorch_mobilefacenet_epoch38_feature-result/'
    'megaface_result/cmc_facescrub_megaface_mobilefacenetPytorch_112x112_1000000_1.json')

    return parser.parse_args(argv)


def main(args):
    # print(json_path)
    correct_json_path = Path(args.json_path)
    # print(correct_json_path)
    cmc, roc = read_json(correct_json_path)
    cal_cmc(cmc)
    cal_roc(roc=roc)


if __name__ == '__main__':
    # json_path = './cmc.json'
    main(parse_arguments(sys.argv[1:]))
