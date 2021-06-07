import numpy as np
import torch.nn as nn
from prettytable import PrettyTable
# from dcp.models.mobilenetv1 import MobileNetV1
# from dcp.models.mobilenetv2_cifar import MobileNetV2Cifar
from dcp.mask_conv import MaskConv2d, MaskLinear

from torch import nn

from . import model_transform as mt

__all__ = ["ModelAnalyse"]


class ModelAnalyse(object):
    def __init__(self, model, logger, setting=None):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.flops = []
        self.madds = []
        self.weight_shapes = []
        self.layer_names = []
        self.channel_nums = []
        self.bias_shapes = []
        self.output_shapes = []
        self.pruned = None
        self.setting = setting

    def params_count(self, pruned=False, include_conv1=False):
        params_num_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Shape", "Dim"]

        self.logger.info("------------------------number of parameters------------------------\n")
        for name, param in self.model.named_parameters():
            if 'pruned_weight' in name:
                if not pruned:
                    continue
                else:
                    param_num = param.numel()
                    param_equal_zero = param.eq(0).sum().item()
                    param_num = param_num - param_equal_zero
                    param_shape = [shape for shape in param.shape]
                    params_num_list.append(param_num)
            else:
                if pruned and 'conv1' in name and include_conv1:
                    continue
                if pruned and (('fc.weight' in name or 'linear.weight' in name) and 'pruned_weight' in name):
                    continue
                param_num = param.numel()
                param_shape = [shape for shape in param.shape]
                params_num_list.append(param_num)

            output.add_row([name, param_shape, param_num])

        self.logger.info(output)

        params_num_list = np.array(params_num_list)
        params_num = params_num_list.sum()
        self.logger.info("|===>Number of parameters is: {}".format(params_num))
        return params_num

    def zero_count(self):
        weights_zero_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Zero Num"]

        self.logger.info("------------------------number of zeros in parameters------------------------\n")
        for name, param in self.model.named_parameters():
            weight_zero = param.data.eq(0).sum().item()
            weights_zero_list.append(weight_zero)
            output.add_row([name, weight_zero])
        self.logger.info(output)

        weights_zero_list = np.array(weights_zero_list)
        zero_num = weights_zero_list.sum()
        self.logger.info("|===>Number of zeros is: {}".format(zero_num))
        return zero_num

    def _flops_conv_hook(self, layer, x, out):
        # https://www.zhihu.com/question/65305385

        # compute number of floating point operations
        if hasattr(layer, 'd') and self.pruned:
            in_channels = layer.d.sum().item()
        else:
            in_channels = layer.in_channels
        groups = layer.groups

        # ours_flops
        channels_per_filter = in_channels // groups
        # iccv_flops
        # channels_per_filter = in_channels

        if layer.bias is not None:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * channels_per_filter * layer.weight.size(2) * layer.weight.size(3)) \
                          * layer.weight.size(0)
        else:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * channels_per_filter * layer.weight.size(2) * layer.weight.size(3) - 1.) \
                          * layer.weight.size(0)

        # layer_flops = int(layer_flops / groups)

        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.channel_nums.append(in_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.flops.append(layer_flops)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = out.size(2)*out.size(3)*layer.weight.size(1)*layer.weight.size(2)*layer.weight.size(0)
        """

    def _flops_linear_hook(self, layer, x, out):
        # compute number floating point operations

        if self.setting is not None:
            # for r34
            if self.setting.net_type in ["LResnetxE-IR"]:
                if isinstance(layer, MaskLinear):
                    q_zero = layer.d.eq(0).sum().item()
                    input_channels = len(layer.d) - q_zero

                    if layer.bias is not None:
                        layer_flops = (2 * input_channels) * layer.weight.size(0)
                    else:
                        layer_flops = (2 * input_channels) * layer.weight.size(0)

                    self.weight_shapes.append([layer.weight.size(0), input_channels])
                    self.channel_nums.append(input_channels)

                    self.output_shapes.append(list(out.shape))
                    if layer.bias is not None:
                        self.bias_shapes.append(list(layer.bias.shape))
                    else:
                        self.bias_shapes.append([0])

                else:
                    if layer.bias is not None:
                        layer_flops = (2 * layer.weight.size(1)) * layer.weight.size(0)
                    else:
                        layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)

                    self.weight_shapes.append(list(layer.weight.shape))
                    self.channel_nums.append(x[0].shape[1])
                    self.output_shapes.append(list(out.shape))
                    if layer.bias is not None:
                        self.bias_shapes.append(list(layer.bias.shape))
                    else:
                        self.bias_shapes.append([0])

            # for mobilefacenet_v1
            elif self.setting.net_type in ["mobilefacenet_v1"]:
                if isinstance(layer, MaskLinear) and self.pruned:
                    q_zero = layer.d.eq(0).sum().item()
                    input_channels = len(layer.d) - q_zero

                    if layer.bias is not None:
                        layer_flops = (2 * input_channels) * layer.weight.size(0)
                    else:
                        layer_flops = (2 * input_channels) * layer.weight.size(0)

                    self.weight_shapes.append([layer.weight.size(0), input_channels])
                    self.channel_nums.append(input_channels)

                    self.output_shapes.append(list(out.shape))
                    if layer.bias is not None:
                        self.bias_shapes.append(list(layer.bias.shape))
                    else:
                        self.bias_shapes.append([0])

                else:
                    if layer.bias is not None:
                        layer_flops = (2 * layer.weight.size(1)) * layer.weight.size(0)
                    else:
                        layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)

                    self.weight_shapes.append(list(layer.weight.shape))
                    self.channel_nums.append(x[0].shape[1])
                    self.output_shapes.append(list(out.shape))
                    if layer.bias is not None:
                        self.bias_shapes.append(list(layer.bias.shape))
                    else:
                        self.bias_shapes.append([0])

            else:
                assert False

        else:
            if layer.bias is not None:
                layer_flops = (2 * layer.weight.size(1)) * layer.weight.size(0)
            else:
                layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)

            self.weight_shapes.append(list(layer.weight.shape))
            self.channel_nums.append(x[0].shape[1])
            self.output_shapes.append(list(out.shape))
            if layer.bias is not None:
                self.bias_shapes.append(list(layer.bias.shape))
            else:
                self.bias_shapes.append([0])

        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = layer.weight.size(1)*layer.weight.size(0)
        """
        self.flops.append(layer_flops)

    def _madds_conv_hook(self, layer, x, out):
        input = x[0]
        batch_size = input.shape[0]
        output_height, output_width = out.shape[2:]

        kernel_height, kernel_width = layer.kernel_size
        if hasattr(layer, 'd'):
            in_channels = layer.d.sum().item()
        else:
            in_channels = layer.in_channels
        out_channels = layer.out_channels
        groups = layer.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

        active_elements_count = batch_size * output_height * output_width

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.channel_nums.append(in_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        # layer_madds = layer.weight.size(0) * layer.weight.size(1)
        # if layer.bias is not None:
        #     layer_madds += layer.weight.size(0)
        input = x[0]
        batch_size = input.shape[0]
        overall_flops = int(batch_size * input.shape[1] * out.shape[1])

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out.shape[1]
        overall_flops = overall_flops + bias_flops
        self.weight_shapes.append(list(layer.weight.shape))
        self.channel_nums.append(input.shape[1])
        self.output_shapes.append(list(out.shape))
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def madds_compute(self, x):
        """
        Compute number of multiply-adds of the model
        """

        hook_list = []
        self.madds = []
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._madds_conv_hook))
                self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._madds_linear_hook))
                self.layer_names.append(layer_name)
        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        madds_np = np.array(self.madds)
        madds_sum = float(madds_np.sum())
        percentage = madds_np / madds_sum

        output = PrettyTable()
        output.field_names = ["Layer", "Weight Shape", "#Channels", "Bias Shape", "Output Shape", "Madds", "Percentage"]

        self.logger.info("------------------------Madds------------------------\n")
        for i in range(len(self.madds)):
            output.add_row([self.layer_names[i], self.weight_shapes[i], self.channel_nums[i], self.bias_shapes[i],
                            self.output_shapes[i], madds_np[i], percentage[i]])
        self.logger.info(output)
        repo_str = "|===>Total MAdds: {:e}".format(madds_sum)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return madds_np

    def flops_compute(self, x, pruned=False):
        """
        Compute number of flops of the model
        """

        hook_list = []
        self.flops = []
        self.pruned = pruned
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._flops_conv_hook))
                self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._flops_linear_hook))
                self.layer_names.append(layer_name)

        # run forward for computing FLOPs
        self.model.eval()
        # if isinstance(self.model, MobileNetV1):
        #     output = self.model.features(x)
        #     output = self.model.classifier(output)
        # else:
        self.model(x)

        flops_np = np.array(self.flops)
        flops_sum = float(flops_np.sum())
        percentage = flops_np / flops_sum

        output = PrettyTable()
        output.field_names = ["Layer", "Weight Shape", "#Channels", "Bias Shape", "Output Shape", "FLOPs", "Percentage"]

        self.logger.info("------------------------FLOPs------------------------\n")
        for i in range(len(self.flops)):
            output.add_row([self.layer_names[i], self.weight_shapes[i], self.channel_nums[i], self.bias_shapes[i],
                            self.output_shapes[i], flops_np[i], percentage[i]])
        self.logger.info(output)
        repo_str = "|===>Total FLOPs: {:e}".format(flops_sum)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return flops_np