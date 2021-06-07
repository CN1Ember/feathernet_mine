#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 17:52
# @Author  : xiezheng
# @Site    : 
# @File    : draw_log.py

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_acc(line):
    line_arr = line.split(':')
    arr = line_arr[-1].split('+-')
    return float(arr[0])

def plot_img(acc0, acc1, acc2, name):
    plt.figure()
    plt.grid(ls='--')
    plt.xlabel('Iteration')
    plt.ylabel('{} Accuracy'.format(name))
    # ax.set_xscale("log")
    plt.yscale("log")
    # plt.title("ResNet32 with SGD on Cifar10")
    # for i in range(len(opt_list)):
    plt.plot(range(len(acc0)), acc0, linewidth=1.6, label=name + "_width")
    plt.plot(range(len(acc1)), acc1, linewidth=1.6, label=name+"_deep")
    plt.plot(range(len(acc2)), acc2, linewidth=1.6, label=name + "_deep_width")
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(os.path.join('./{}_accuracy.pdf'.format(name)))

def get_path_data(log_path):
    lfw_acc = []
    cfp_acc = []
    age_acc = []

    for line in open(log_path, 'r'):
        line = line.strip()
        if 'lfw' in line and 'Accuracy-Flip' in line:
            lfw_acc.append(get_acc(line))

        elif 'cfp_fp' in line and 'Accuracy-Flip' in line:
            cfp_acc.append(get_acc(line))

        elif 'agedb_30' in line and 'Accuracy-Flip' in line:
            age_acc.append(get_acc(line))

    return lfw_acc, cfp_acc, age_acc


def append_path_data(log_path, lfw_acc, cfp_acc, age_acc):

    for line in open(log_path, 'r'):
        line = line.strip()
        if 'lfw' in line and 'Accuracy-Flip' in line:
            lfw_acc.append(get_acc(line))

        elif 'cfp_fp' in line and 'Accuracy-Flip' in line:
            cfp_acc.append(get_acc(line))

        elif 'agedb_30' in line and 'Accuracy-Flip' in line:
            age_acc.append(get_acc(line))

    return lfw_acc, cfp_acc, age_acc


width_log_path = './width_log'

deep_width_log_path = './deep_width_log'
deep_width_todo_log_path = './deep_width_todo_log'

deep_log_log = './deep_arcface_log'
deep_todo_log_log = './deep_arcface_todo_log'

width_lfw_acc, width_cfp_acc, width_age_acc = get_path_data(width_log_path)

deep_width_lfw_acc, deep_width_cfp_acc, deep_width_age_acc = get_path_data(deep_width_log_path)
deep_width_lfw_acc, deep_width_cfp_acc, deep_width_age_acc = append_path_data(deep_width_todo_log_path,
                                                                              deep_width_lfw_acc, deep_width_cfp_acc,
                                                                              deep_width_age_acc)

deep_lfw_acc, deep_cfp_acc, deep_age_acc = get_path_data(deep_log_log)
deep_lfw_acc, deep_cfp_acc, deep_age_acc = append_path_data(deep_todo_log_log, deep_lfw_acc, deep_cfp_acc, deep_age_acc)

plot_img(width_lfw_acc, deep_lfw_acc, deep_width_lfw_acc, 'LFW')
plot_img(width_cfp_acc, deep_cfp_acc, deep_width_cfp_acc, 'CFP')
plot_img(width_age_acc, deep_age_acc, deep_width_age_acc, 'AGE')







