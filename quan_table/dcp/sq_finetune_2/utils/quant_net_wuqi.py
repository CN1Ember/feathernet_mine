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

import torch.nn as nn
import sys

import numpy as np
import _pickle as cPickle
from collections import *

import dcp.sq_finetune.utils.quant_op_wuqi as qfp

# quant_table show the set when quantizing('quant' in phase)
# 'quant': whether it already has a quant_table
# 'test': it is used when eval the net
quant_table = OrderedDict({'quant': True, 'test': True, 'quant_bp_coef': 2})
quant_hist = OrderedDict()

class PreluQuant(nn.Module):
    def __init__(self, bn, phase='quant', name='', mode='before'):
        super(PreluQuant, self).__init__()

        self.name = name
        self.mode= mode
        self.add_module('bn', bn)
        if 'quant' in phase:
            self.fw_hook= self.bn.register_forward_pre_hook(self.pre_hook) 


    def rmvhooks(self):
        if self.fw_hook != 0:
            self.fw_hook.remove()


    def pre_hook(self, mdl, datain):
        if quant_table['quant']:
            qfp.quantize_data(datain[0], intlen=quant_table[self.name])  # quantize input data inplace
        else:
            if quant_hist['step'] == 1:
                if self.name not in quant_hist.keys():
                    quant_hist[self.name] = {'max_data': 0}
                quant_hist['layer_order'].append(self.name)
                quant_hist[self.name]['max_data'] = max(quant_hist[self.name]['max_data'], datain[0].data.abs().max())

            if quant_hist['step'] == 2:
                th = quant_hist[self.name]['max_data'].cpu()
                th_cp= th.cpu().numpy()
                hist, hist_edges = np.histogram(datain[0].data.cpu().numpy(), bins=8192, range=(0, th_cp))
                if 'hist' in quant_hist[self.name].keys():
                    quant_hist[self.name]['hist'] += hist
                else:
                    quant_hist[self.name]['hist'] = hist
                    quant_hist[self.name]['hist_edges'] = hist_edges

            if quant_hist['step'] == 3: # original test
                pass

            if quant_hist['step'] == 4:
                qfp.quantize_data(datain[0], intlen=quant_table[self.name])


    def forward(self, x):
        out= self.bn(x)
        return out

class BnQuant(nn.Module):
    def __init__(self, bn, phase='quant', name='', mode='before'):
        super(BnQuant, self).__init__()

        self.name = name
        self.mode= mode
        self.add_module('bn', bn)
        # print(bn)
        if 'quant' in phase:
            self.fw_hook= self.bn.register_forward_pre_hook(self.pre_hook) 


    def rmvhooks(self):
        if self.fw_hook != 0:
            self.fw_hook.remove()


    def pre_hook(self, mdl, datain):
        # print(self.name)
        if quant_table['quant']:
            qfp.quantize_data(datain[0], intlen=quant_table[self.name])  # quantize input data inplace
        else:
            if quant_hist['step'] == 1:
                # print(datain[0].data.shape)
                if self.name not in quant_hist.keys():
                    quant_hist[self.name] = {'max_data': 0}
                quant_hist['layer_order'].append(self.name)
                quant_hist[self.name]['max_data'] = max(quant_hist[self.name]['max_data'], datain[0].data.abs().max())

            if quant_hist['step'] == 2:
                th = quant_hist[self.name]['max_data'].cpu()
                th_cp= th.cpu().numpy()
                hist, hist_edges = np.histogram(datain[0].data.cpu().numpy(), bins=8192, range=(0, th_cp))
                if 'hist' in quant_hist[self.name].keys():
                    quant_hist[self.name]['hist'] += hist
                else:
                    quant_hist[self.name]['hist'] = hist
                    quant_hist[self.name]['hist_edges'] = hist_edges

            if quant_hist['step'] == 3: # original test
                pass

            if quant_hist['step'] == 4:
                qfp.quantize_data(datain[0], intlen=quant_table[self.name])


    def forward(self, x):
        out= self.bn(x)
        return out

class ConvQuant(nn.Module):
    def __init__(self, conv, phase='quant', name='', mode='before'):
        super(ConvQuant, self).__init__()

        self.name = name
        self.mode= mode
        self.add_module('conv', conv)
        if 'quant' in phase:
            self.fw_hook= self.conv.register_forward_pre_hook(self.pre_hook) 


    def rmvhooks(self):
        if self.fw_hook != 0:
            self.fw_hook.remove()


    def pre_hook(self, mdl, datain):
        if quant_table['quant']:
            qfp.quantize_data(datain[0], intlen=quant_table[self.name])  # quantize input data inplace
        else:
            if quant_hist['step'] == 1:
                if self.name not in quant_hist.keys():
                    quant_hist[self.name] = {'max_data': 0}
                quant_hist['layer_order'].append(self.name)
                quant_hist[self.name]['max_data'] = max(quant_hist[self.name]['max_data'], datain[0].data.abs().max())

            if quant_hist['step'] == 2:
                th = quant_hist[self.name]['max_data'].cpu()
                th_cp= th.cpu().numpy()
                hist, hist_edges = np.histogram(datain[0].data.cpu().numpy(), bins=8192, range=(0, th_cp))
                if 'hist' in quant_hist[self.name].keys():
                    quant_hist[self.name]['hist'] += hist
                else:
                    quant_hist[self.name]['hist'] = hist
                    quant_hist[self.name]['hist_edges'] = hist_edges

            if quant_hist['step'] == 3: # original test
                pass

            if quant_hist['step'] == 4:
                qfp.quantize_data(datain[0], intlen=quant_table[self.name])


    def forward(self, x):
        out= self.conv(x)
        return out

# including the whole net to finetune the net with quantized parameters
class QuantNet(nn.Module):
    def __init__(self, net):
        super(QuantNet, self).__init__()

        # load the quant_table only if it exists
        if quant_table['quant']:
            print('load quant_table.bin......')
            try:
                qtemp = cPickle.load(open('quant_table.bin', 'rb'))
                quant_table.update(qtemp)
            except Exception:
                sys.exit("Error! Can't load quant_table.bin!")
            quant_table['quant'] = True
            quant_table['quant_bp_coef'] = 1e-3

        self.add_module('net', net)

    # must rewrite this function, or this class will be added a needless name
    def forward(self, x):
        net = self.net
        res = self.net(x)
        return res


def save_params(optimizer):
    qfp.add_pshadow(optimizer)


def load_params(optimizer):
    qfp.copy_pshadow2param(optimizer)
