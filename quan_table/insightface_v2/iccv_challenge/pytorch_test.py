#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 0:22
# @Author  : xiezheng
# @Site    : 
# @File    : pytorch_test.py

# import mxnet as mx
#
# data = mx.nd.NDArray([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
# label = mx.nd.NDArray([1, 0, 2, 3])
# ignore_label = 1
# a = mx.symbol.SoftmaxOutput(data=mx.sym.Variable('data'), label=mx.sym.Variable('label'), name='softmax', normalization='valid')
# # 通过把两个输出组成一个group来得到自己需要查看的中间层输出结果
# # group = mx.symbol.Group([a])
# # print(group.list_outputs())
#
# ex = a.bind(ctx=mx.cpu(), args={'data' : data, 'label':label})
# ex.forward()
# print('number of outputs = %d\nthe first output = \n%s' % ( len(ex.outputs), ex.outputs[0].asnumpy()))

import torch
from torch import nn
from torch.autograd import Variable

ce_loss = nn.CrossEntropyLoss()
softmax_loss = nn.Softmax()

# input = Variable(torch.Tensor([[1, 2, 3], [11, 7, 5]]) ,requires_grad=True)
# target = Variable(torch.Tensor([2, 0]),requires_grad=True).long()

input = Variable(torch.Tensor([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]) ,requires_grad=True)
target = Variable(torch.Tensor([1, 0, 2, 3]),requires_grad=True).long()

softmax_out = softmax_loss(input)
print("softmax_out={}".format(softmax_out))

ce_output = ce_loss(input, target)
ce_output.backward()
print("ce_output={}".format(ce_output))
print("input grad={}".format(input.grad))

