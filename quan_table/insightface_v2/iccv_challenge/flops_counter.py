#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 15:33
# @Author  : xiezheng
# @Site    : 
# @File    : flops_counter.py


'''
@author: insightface
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse
import numpy as np
import mxnet as mx


def is_no_bias(attr):
  ret = False
  if 'no_bias' in attr and (attr['no_bias']==True or attr['no_bias']=='True'):
    ret = True
  return ret


def count_fc_flops(input_filter, output_filter, attr):
  #print(input_filter, output_filter ,attr)
  ret = 2*input_filter*output_filter
  if is_no_bias(attr):
    ret -= output_filter
  return int(ret)


def count_conv_flops(input_shape, output_shape, attr):
  # print('iccv challenge count_conv_flops_type!')
  kernel = attr['kernel'][1:-1].split(',')
  kernel = [int(x) for x in kernel]

  #print('kernel', kernel)
  if is_no_bias(attr):
    ret = (2*input_shape[1]*kernel[0]*kernel[1]-1)*output_shape[2]*output_shape[3]*output_shape[1]
  else:
    ret = 2*input_shape[1]*kernel[0]*kernel[1]*output_shape[2]*output_shape[3]*output_shape[1]
  num_group = 1
  if 'num_group' in attr:
    num_group = int(attr['num_group'])
  ret /= num_group
  return int(ret)

# Ours
def count_conv_flops_ours(input_shape, output_shape, attr):
  # print('our dcp count_conv_flops_type!')
  kernel = attr['kernel'][1:-1].split(',')
  kernel = [int(x) for x in kernel]

  #print('kernel', kernel)
  num_group = 1
  if 'num_group' in attr:
      num_group = int(attr['num_group'])

  input_channel = input_shape[1] // num_group

  if is_no_bias(attr):
    ret = (2*input_channel*kernel[0]*kernel[1]-1)*output_shape[2]*output_shape[3]*output_shape[1]
  else:
    ret = 2*input_channel*kernel[0]*kernel[1]*output_shape[2]*output_shape[3]*output_shape[1]

  return int(ret)


def count_flops(sym, count_conv_flops_type, **data_shapes):
  all_layers = sym.get_internals()
  #print(all_layers)
  arg_shapes, out_shapes, aux_shapes = all_layers.infer_shape(**data_shapes)
  out_shape_dict = dict(zip(all_layers.list_outputs(), out_shapes))

  nodes = json.loads(sym.tojson())['nodes']
  nodeid_shape = {}
  for nodeid, node in enumerate(nodes):
    name = node['name']
    layer_name = name+"_output"
    if layer_name in out_shape_dict:
      nodeid_shape[nodeid] = out_shape_dict[layer_name]
  #print(nodeid_shape)
  FLOPs = 0
  for nodeid, node in enumerate(nodes):
    flops = 0
    if node['op']=='Convolution':
      output_shape = nodeid_shape[nodeid]
      name = node['name']
      attr = node['attrs']
      input_nodeid = node['inputs'][0][0]
      input_shape = nodeid_shape[input_nodeid]
      if count_conv_flops_type == 'dcp':
        flops = count_conv_flops_ours(input_shape, output_shape, attr)
      elif count_conv_flops_type == 'iccv':
        flops = count_conv_flops(input_shape, output_shape, attr)
      else:
          assert False

    elif node['op']=='FullyConnected':
      attr = node['attrs']
      output_shape = nodeid_shape[nodeid]
      input_nodeid = node['inputs'][0][0]
      input_shape = nodeid_shape[input_nodeid]
      output_filter = output_shape[1]
      input_filter = input_shape[1]*input_shape[2]*input_shape[3]
      #assert len(input_shape)==4 and input_shape[2]==1 and input_shape[3]==1
      flops = count_fc_flops(input_filter, output_filter, attr)
    #print(node, flops)
    FLOPs += flops

  return FLOPs


def flops_str(FLOPs):
  preset = [ (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K') ]

  for p in preset:
    if FLOPs//p[0]>0:
      N = FLOPs/p[0]
      ret = "%.1f%s"%(N, p[1])
      return ret
  ret = "%.1f"%(FLOPs)
  return ret


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='flops counter')
  # general
  #parser.add_argument('--model', default='../models2/y2-arcface-retinat1/model,1', help='path to load model.')
  #parser.add_argument('--model', default='../models2/r100fc-arcface-retinaa/model,1', help='path to load model.')
  # parser.add_argument('--model', default='../models2/r50fc-arcface-emore/model,1', help='path to load model.')

  # 170: 933M
  # parser.add_argument('--model', default='/home/xiezheng/2019Programs/insightface/models/'
  #                                        'MobileFaceNet_r2-8-16-4_512_933M/model-emore-test,45', help='path to load model.')
  # parser.add_argument('--model', default='/home/xiezheng/2019Programs/insightface/models/'
  #                                        'MobileFaceNet_r3-8-16-5_512_994M/model-emore-test,34', help='path to load model.')

  # 246:
  # parser.add_argument('--model', default='/home/xiezheng/program2019/insightface/models/'
  #                                        'MobileFaceNet_todo/model-emore-softmax,82', help='path to load model.')
  # parser.add_argument('--model', default='/home/xiezheng/program2019/insightface/models/'
  #                     'MobileFaceNet_r3-8-16-5_gdc_512_994M_softmax_then_arcface/model-emore-softmax,1', help='path to load model.')

  # 186
  parser.add_argument('--model', default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/'
                                         'log/mxnet_insightface_model/model-r100-ii/model,0', help='path to load model.')

  args = parser.parse_args()
  _vec = args.model.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

  params = 0
  count = 0
  for key, value in arg_params.items():
      count += 1
      print('count={}, key={}, value={}'.format(count, key, value))
      if isinstance(value, int):
          layer_param = 1
      elif len(value.shape) == 1:
          layer_param = value.shape[0]
      elif len(value.shape) > 1:
          layer_param = len(value.reshape(-1))
      else:
          assert False, 'value type={}'.format(type(value))
      params += layer_param
      print('count={}, params={}'.format(count, params))
  print('model #param.={}, {}M'.format(params, params/1e6))

  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  FLOPs = count_flops(sym, count_conv_flops_type='iccv', data=(1,3,112,112))
  print('ICCV Challenge: FLOPs:', FLOPs)
  FLOPs = count_flops(sym, count_conv_flops_type='dcp', data=(1, 3, 112, 112))
  print('DCP: FLOPs:', FLOPs)


