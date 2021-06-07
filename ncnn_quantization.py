import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import Parameter

import numpy as np
import os
import matplotlib.pyplot as plt

# global params
QUANTIZE_NUM = 127.0


def replace_layer_by_unique_name(module, unique_name, layer):
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        module._modules[unique_names[0]] = layer
    else:
        replace_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
            layer
        )


def quantization_on_weights(x):
    c_out, c_in, k, w = x.shape
    x_reshape = x.reshape(c_out, -1)
    # print('x={}'.format(x))
    # print('x_reshape={}'.format(x_reshape))
    threshold, _ = torch.max(torch.abs(x_reshape), 1)  # threshold shape=c_out*1
    weight_scale = QUANTIZE_NUM / threshold
    weight_scale = weight_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)

    x = x*weight_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / weight_scale
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantization_on_input(x):
    threshold = torch.max(torch.abs(x))  # threshold shape=1
    activations_scale = QUANTIZE_NUM / threshold
    x = x*activations_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / activations_scale
    return x


def quantization_on_input_fix_scale(x, activation_value):
    threshold = activation_value
    activations_scale = QUANTIZE_NUM / threshold
    x = x*activations_scale
    x = RoundFunction.apply(x)
    x = torch.clamp(x, min=-127, max=127)
    x = x / activations_scale
    return x


class Conv2d_Ncnn_int8(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, activation_value=0.0):
        super(Conv2d_Ncnn_int8, self).__init__(in_channels, out_channels, kernel_size,
                                               stride, padding, dilation, groups, bias)
        self.activation_value = activation_value


    def forward(self, input):
        # quantized_input = quantization_on_input(input)
        quantized_input = quantization_on_input_fix_scale(input, self.activation_value)
        quantized_weight = quantization_on_weights(self.weight)
        if self.bias is not None:   # not quantization self.bias
            quantized_bias = self.bias
        else:
            quantized_bias = None
        return F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride, self.padding,
                        self.dilation, self.groups)

# class PReLU_Ncnn_int8(nn.PReLU):
#     """
#     custom ReLU for quantization
#     """
#     def __init__(self, num_parameters=1):
#         super(PReLU_Ncnn_int8, self).__init__(num_parameters=num_parameters)
#
#     def forward(self, input):
#         out = quantization_on_activations(input)
#         return out

def get_activation_value(table_path):
    activation_value_list = []
    for line in open(table_path):
        line = line.strip()
        key, value = line.split(' ')

        # remove 'net.'
        if "net" in key:
            key = key.replace('net.', '')
        activation_value_list.append(value)

    # for key in activation_value_list:
    #     print('{} {}'.format(key, activation_value_list[key]))
    return activation_value_list



def replace(model, activation_value_list):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("count={}, name={}, activation_value_list[{}]={}".format(count, name,
                                                                           count, activation_value_list[count]))
            temp_conv = Conv2d_Ncnn_int8(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                groups=module.groups,
                bias=(module.bias is not None),
                activation_value=float(activation_value_list[count]))
            temp_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                temp_conv.bias.data.copy_(module.bias.data)
            replace_layer_by_unique_name(model, name, temp_conv)
            count += 1
    # print("After replace:\n {}".format(model))
    return model


# if __name__ == "__main__":
#     x = torch.rand(2,2,2,2)
#     print('x={}'.format(x))
#     q_x = quantization_on_weights(x)
#     print('q_x={}'.format(q_x))