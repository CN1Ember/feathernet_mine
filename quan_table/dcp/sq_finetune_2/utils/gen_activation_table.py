# -*- coding: utf-8 -*-

# BUG1989 is pleased to support the open source community by supporting ncnn available.
#
# Copyright (C) 2019 BUG1989. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
sys.path.insert(1,'/home/lidaiyuan/code/insightface_dcp/')
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

import argparse
import torch
import torch.backends.cudnn as cudnn

from dcp.sq_finetune.utils.quant_replace import *
from dcp.sq_finetune.utils.quant_net import *
import dcp.sq_finetune.utils.quant_op as qfp

from dcp.models.mobilefacenet_pruned import pruned_Mobilefacenet
from dcp.models.mobilefacenetv2_width_wmss import Mobilefacenetv2_width_wm
from dcp.models.FaceFeatherNet import FaceFeatherNetA

# from dcp.dataloader import get_quantization_lfw_dataloader
from read_data.read_data import CASIA


def train(net, trainloader, optimizer, epoch, calibration_path):
    print('\nEpoch: %d' % epoch)
    calibration_file = open(calibration_path, 'a+')

    # step 1: get the range of data_in
    quant_hist['step'] = 1
    print('-----step 1-----')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            input_var = Variable(inputs).float().cuda()
            # target_var = Variable(targets).long().cuda()
            # inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(input_var)
            print(batch_idx)
            # print(quant_hist)

    # step 2: get the hist of data_in 
    quant_hist['step'] = 2
    print('-----step 2-----')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            input_var = Variable(inputs).float().cuda()
            # target_var = Variable(targets).long().cuda()
            optimizer.zero_grad()
            outputs = net(input_var)
            print(batch_idx)

    print('-----step 2 done-----')
    print(quant_hist)

    # step 3: get quant_table 
    print('-----step 3-----')
    print(quant_hist.keys())

    for key in quant_hist.keys():
        if key == 'step':
            continue
        quant_table[key] = qfp.get_best_activation_intlength(quant_hist[key]['hist'], quant_hist[key]['hist_edges'])

        # # remove 'net.'
        # if "net" in key:
        #     key = key.replace('net.', '')

        lines = "{} {}".format(key, quant_table[key])
        print(lines)
        calibration_file.write("{}\n".format(lines))

    # output quant_table
    calibration_file.close()
    print(quant_table)
    print("====> Save calibration activation scale table success...")

    # print('save quant_table...')
    # cPickle.dump(quant_table, open('quant_table.bin', 'wb'), protocol=2)
    # print('save quant_table done. ')



def main(args):
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    normalize = transforms.Normalize(mean=[0.21735254], std=[0.21561144])  # 0224/nir_train_01 KL
    # Data
    print('==> Preparing data..')
    val_dataset = CASIA(
        'train_set_21030601_20210307212802',
        transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=False, sampler=None)

    # Model
    print('==> Building model..')

    # old
    # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # net = pruned_Mobilefacenet(pruning_rate=0.25)

    # new
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    net = FaceFeatherNetA()
    check_point_params = torch.load(args.model_path)
    model_state = check_point_params
    print(net)
    net.load_state_dict(model_state)
    print("net load success !!!")

    net = net.cuda()
    net.eval()

    # no quant_table now
    quant_table['quant'] = False

    net = rep_layer(net)
    print("----After Replace Conv2d with QuantConv2d----")
    # print(net)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=5e-4)



    for epoch in range(start_epoch, start_epoch + 1):
        train(net, val_loader, optimizer, epoch, args.table_path)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--n_threads', type=int, help='n_threads', default=2)

    # ms1mv2_mobilefacenet_p0.5
    # parser.add_argument('--model_path', type=str, help='model_path', default='/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.5_mobilefacenet_v1/checkpoint_025.pth')
    # parser.add_argument('--table_path', type=str, help='table_path', default='./mobilefacenet_p0.5.table')

    # ms1mv2_mobilefacenet_p0.25
    # parser.add_argument('--model_path', type=str, help='model_path', default='/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/p0.25_mobilefacenet_v1_without_fc_checkpoint_022.pth')
    # parser.add_argument('--table_path', type=str, help='table_path', default='./mobilefacenet_p0.25.table')


    # iccv_mobilefacenet_p0.5
    # parser.add_argument('--model_path', type=str, help='model_path',
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/finetune/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs512_e36_lr0.010_step[15, 25, 31]_p0.5_bs512_cosine_finetune_without_fc_20190809/check_point/checkpoint_035.pth')
    # parser.add_argument('--table_path', type=str, help='table_path', default='./iccv_mobilefacenet_p0.5_activation.table')

    # iccv_mobilefacenet_p0.5_our_sq_17
    # parser.add_argument('--model_path', type=str, help='model_path',
    #                     default='/home/xiezheng/lidaiyuan/Face_integration/pretrain_model/checkpoint_25.pth')
    # parser.add_argument('--table_path', type=str, help='table_path',
    #                     default='/home/xiezheng/lidaiyuan/Face_integration/pretrain_model/checkpoint_25.pth.table')

    parser.add_argument('--model_path', type=str, help='model_path',
                        default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_nir_120_rm_block.pth.tar')
    parser.add_argument('--table_path', type=str, help='table_path',
                        default='/home/lidaiyuan/feathernet2020/model/pytorchmodel/_nir_120_rm_block_act.table')   

    # 186
    parser.add_argument('--gpu', type=str, help='gpu', default='7')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
