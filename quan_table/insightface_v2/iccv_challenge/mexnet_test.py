#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 8:37
# @Author  : xiezheng
# @Site    : 
# @File    : mexnet_test.py

import mxnet as mx
import numpy as np

data = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
print(data.shape)
# label = np.array([1, 0, 2, 3])
#
# # data = np.array([[1, 2, 3],[11, 7, 5]])
# # label = np.array([2, 0])
#
# data_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=1, data_name='data', label_name='le')
#
# dataout = mx.sym.var("data")
# labelout = mx.sym.var('le')
#
# # dataout = mx.nd.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
# # label = mx.nd.array([1, 0, 2, 3])
# #
# # X = mx.sym.Variable("X")
# # Label = mx.sym.var('Label')
#
# # fc1 = mx.sym.FullyConnected(dataout, num_hidden=4, no_bias=False, flatten=True, name='fc1')
# # sym = mx.sym.SoftmaxOutput(dataout, labelout, name='smax')
# # sym = mx.sym.softmax_cross_entropy(X, Label, name='smax')
# # executor = sym.bind(ctx=mx.cpu(0),args={"X" : X}, args_grad= {"X": mx.nd.zeros((1, 4))})
#
# sym = mx.sym.softmax_cross_entropy(dataout, labelout, name='smax')
# executor = sym.bind(ctx=mx.cpu(0),args={"data" : data}, args_grad= {"data": mx.nd.zeros((4, 4))})
#
# out = executor.forward(is_train=True).copy()
#
# executor.backward(out)
# print(executor.grad_arrays)





#
# # print(sym.gradient(data))
# mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=['data'], label_names=['le'])
#
# mod.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
# mod.init_params(initializer=mx.init.Uniform(scale=.1))
#
# # for batch in data_iter:
# #     out = mod.forward(batch, is_train=True)
# #     print(out.asnumpy())
#
# # out = mod.predict(data_iter)
# # print("out={}".format(out))
# # print(out.asnumpy())
# # print(out.asnumpy().sum())
#
# for nbatch, eval_batch in enumerate(data_iter):
#     mod.forward_backward(eval_batch)
# print(mod._exec_group.grad_arrays)
#     # print(mod._exec_group.grad_arrays)
#
# # out_list = []
# # out_list.append(out)
# # out = mx.symbol.Group(out_list)
# # out.backward()
# # print(data.grad)
#

data = mx.sym.var(name = 'data', shape = (4, 4), dtype = 'float64')
label = mx.sym.var(name = 'label', shape = (4,), dtype = 'float64')

# s = mx.sym.make_loss(mx.sym.SoftmaxOutput(data, label, name='s', normalization='valid'))
s = mx.sym.make_loss(mx.sym.softmax_cross_entropy(data, label, name='s'))

bind = s.simple_bind(ctx = mx.cpu(1), grad_req = {'data': 'write', 's':'write'})
bind.forward(data = mx.nd.array([[1, 2, 3, 4], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]), label = mx.nd.array([1, 0, 2, 3]))
outputs = bind.outputs[0]
# print(outputs)
bind.backward()
print(bind.grad_dict)