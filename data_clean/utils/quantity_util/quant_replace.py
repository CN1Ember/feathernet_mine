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

from dcp.sq_finetune.utils.quant_net import *
# from resnet import *
import torch.nn as nn


def rep_layer(net, layer_tpyes=(nn.Conv2d)):
        # print("----Start replace conv----")
        c_module= ('net',net)
        queue= [c_module]
        order= []
        count= 0
        while queue:
                c_module= queue.pop(0)
                if isinstance(c_module[1],layer_tpyes):
                        count+= 1
                        index= c_module[0].split(',')[1:]
                        express='net'
                        name= 'net'
                        for i in range(len(index)):
                                express+= "._modules['{}']".format(index[i])
                                name+= ".{}".format(index[i])
                        exec(express+'=ConvQuant('+express+',name='+"'"+name+"'"+')')

                if c_module[1] not in order:
                        order.append(c_module)
                        if c_module[1] is None:
                            pass
                        else:
                            for sub_module in c_module[1]._modules.items():
                                    queue.append((c_module[0]+','+sub_module[0],sub_module[1]))

        print('total replaced layer number:{}'.format(count))
        net= QuantNet(net)
        return net


def recover_layer(net,layer_tpyes=ConvQuant):
#         print("----Start recover conv----")
        c_module= ('net',net)
        queue= [c_module]
        order= []
        count= 0
        while queue:
                c_module= queue.pop(0)
                if isinstance(c_module[1],layer_tpyes):
                        count+= 1
                        index= c_module[0].split(',')[1:]
                        express='net'
                        for i in range(len(index)):
                                express+= "._modules['{}']".format(index[i])
                        if count == 1:
                                exec(express+'='+express+'.conv')
                        else:
                                exec(express+'.rmvhooks()')
                                exec(express+'='+express+'.conv')
                if c_module[1] not in order:
                        order.append(c_module)
                        if c_module[1] is None:
                            pass
                        else:
                            for sub_module in c_module[1]._modules.items():
                                    queue.append((c_module[0]+','+sub_module[0],sub_module[1]))
        return net.net


# if __name__ == '__main__':
#         net= ResNet18()
#         net= rep_layer(net)
#         print(net)
#         print('----------------------------------------------')
#         net= recover_layer(net)
#         print(net)
