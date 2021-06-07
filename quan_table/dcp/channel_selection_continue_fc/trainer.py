import math
import time
import numpy as np

import torch.autograd
import torch.nn as nn

import dcp.utils as utils
from dcp.mask_conv import MaskConv2d
from dcp.aux_classifier import AuxClassifier
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck

from dcp.utils.verifacation import evaluate, l2_norm
from dcp.models.insightface_resnet import IRBlock
from dcp.middle_layer import Middle_layer
from dcp.arcface import Arcface

from dcp.models.mobilefacenet import Bottleneck_mobilefacenet
from dcp.middle_layer import Middle_layer_mobilefacenetv1

class View(nn.Module):
    """
    Reshape data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input {:d}".format(x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out


class SegmentWiseTrainer(object):
    """
        Segment-wise trainer for channel selection
    """

    def __init__(self, original_model, pruned_model, train_loader, val_loader, settings, logger, tensorboard_logger,
                 run_count=0):

        self.original_model = original_model
        self.pruned_model = pruned_model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.pruned_last_fc_module = None
        self.origin_last_fc_module = None

        self.lr = self.settings.lr
        self.ori_segments = []
        self.pruned_segments = []
        self.seg_optimizer = []

        self.middle_layer = []
        self.aux_fc = []
        self.fc_optimizer = []
        self.run_count = run_count

        # insightface_dcp
        self.channel_array = []
        if self.settings.net_type == "LResnetxE-IR":
            if self.settings.n_losses == 2:
                self.feature_size = [28, 14, 7]

        elif self.settings.net_type == "mobilefacenet_v1":
            if self.settings.n_losses == 2:
                self.feature_size = [14, 14, 7]

        # run pre-processing
        self.insert_additional_losses()

    def create_segments(self):
        net_origin = None
        net_pruned = None

        # For head_layers
        if self.settings.net_type in ["preresnet", "resnet", "LResnetxE-IR", "mobilefacenet_v1"]:
            if self.settings.net_type == "preresnet":
                net_origin = nn.Sequential(self.original_model.conv)
                net_pruned = nn.Sequential(self.pruned_model.conv)
            elif self.settings.net_type == "resnet":
                net_head = nn.Sequential(
                    self.original_model.conv1,
                    self.original_model.bn1,
                    self.original_model.relu,
                    self.original_model.maxpool)
                net_origin = nn.Sequential(net_head)
                net_head = nn.Sequential(
                    self.pruned_model.conv1,
                    self.pruned_model.bn1,
                    self.pruned_model.relu,
                    self.pruned_model.maxpool)
                net_pruned = nn.Sequential(net_head)

            elif self.settings.net_type == "LResnetxE-IR":
                net_head = nn.Sequential(
                    self.original_model.conv1,
                    self.original_model.bn1,
                    self.original_model.prelu,
                    self.original_model.maxpool)
                net_origin = nn.Sequential(net_head)
                net_head = nn.Sequential(
                    self.pruned_model.conv1,
                    self.pruned_model.bn1,
                    self.pruned_model.prelu,
                    self.pruned_model.maxpool)
                net_pruned = nn.Sequential(net_head)

            elif self.settings.net_type == "mobilefacenet_v1":
                net_head = nn.Sequential(
                    self.original_model.conv1,
                    self.original_model.bn1,
                    self.original_model.prelu1,
                    self.original_model.conv2,
                    self.original_model.bn2,
                    self.original_model.prelu2)
                net_origin = nn.Sequential(net_head)
                net_head = nn.Sequential(
                    self.pruned_model.conv1,
                    self.pruned_model.bn1,
                    self.pruned_model.prelu1,
                    self.pruned_model.conv2,
                    self.pruned_model.bn2,
                    self.pruned_model.prelu2)
                net_pruned = nn.Sequential(net_head)

        elif self.settings.net_type in ["vgg"]:
            net_origin = None
            net_pruned = None
        else:
            self.logger.info("unsupported net_type: {}".format(self.settings.net_type))
            assert False, "unsupported net_type: {}".format(self.settings.net_type)
        self.logger.info("init shallow head done!")

        # For middle_layers
        block_count = 0
        if self.settings.net_type in ["resnet", "preresnet"]:
            for ori_module, pruned_module in zip(self.original_model.modules(), self.pruned_model.modules()):
                if isinstance(ori_module, (PreBasicBlock, Bottleneck, BasicBlock)):
                    self.logger.info("enter block: {}".format(type(ori_module)))
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        net_origin = nn.Sequential(ori_module)

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        net_pruned = nn.Sequential(pruned_module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.ori_segments.append(net_origin)
                        self.pruned_segments.append(net_pruned)
                        net_origin = None
                        net_pruned = None

        if self.settings.net_type in ["mobilefacenet_v1"]:
            for ori_module, pruned_module in zip(self.original_model.modules(), self.pruned_model.modules()):
                if isinstance(ori_module, Bottleneck_mobilefacenet):
                    self.logger.info("enter block: {}".format(type(ori_module)))
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        net_origin = nn.Sequential(ori_module)

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        net_pruned = nn.Sequential(pruned_module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.ori_segments.append(net_origin)
                        self.pruned_segments.append(net_pruned)
                        net_origin = None
                        net_pruned = None

            # Add last_fc for net_pruned
            self.pruned_last_fc_module = Middle_layer_mobilefacenetv1()
            self.pruned_last_fc_module.conv3 = self.pruned_model.conv3
            self.pruned_last_fc_module.bn3 = self.pruned_model.bn3
            self.pruned_last_fc_module.prelu3 = self.pruned_model.prelu3
            self.pruned_last_fc_module.conv4 = self.pruned_model.conv4
            self.pruned_last_fc_module.bn4 = self.pruned_model.bn4
            self.pruned_last_fc_module.linear = self.pruned_model.linear
            self.pruned_last_fc_module.bn5 = self.pruned_model.bn5
            net_pruned.add_module(str(len(net_pruned)), self.pruned_last_fc_module)

            # Add last_fc for net_origin
            self.origin_last_fc_module = Middle_layer_mobilefacenetv1()
            self.origin_last_fc_module.conv3 = self.original_model.conv3
            self.origin_last_fc_module.bn3 = self.original_model.bn3
            self.origin_last_fc_module.prelu3 = self.original_model.prelu3
            self.origin_last_fc_module.conv4 = self.original_model.conv4
            self.origin_last_fc_module.bn4 = self.original_model.bn4
            self.origin_last_fc_module.linear = self.original_model.linear
            self.origin_last_fc_module.bn5 = self.original_model.bn5
            net_origin.add_module(str(len(net_origin)), self.origin_last_fc_module)
            block_count += 1
            self.logger.info("enter block: {}".format(type(self.origin_last_fc_module)))

        if self.settings.net_type in ["LResnetxE-IR"]:
            for ori_module, pruned_module in zip(self.original_model.modules(), self.pruned_model.modules()):
                if isinstance(ori_module,  IRBlock):
                    self.logger.info("enter block: {}".format(type(ori_module)))
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        net_origin = nn.Sequential(ori_module)

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        net_pruned = nn.Sequential(pruned_module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.ori_segments.append(net_origin)
                        self.pruned_segments.append(net_pruned)
                        net_origin = None
                        net_pruned = None

            # Add last_fc for net_pruned
            self.pruned_last_fc_module = Middle_layer()
            self.pruned_last_fc_module.bn2 = self.pruned_model.bn2
            self.pruned_last_fc_module.dropout = self.pruned_model.dropout
            self.pruned_last_fc_module.fc = self.pruned_model.fc
            self.pruned_last_fc_module.bn3 = self.pruned_model.bn3
            net_pruned.add_module(str(len(net_pruned)), self.pruned_last_fc_module)

            # Add last_fc for net_origin
            self.origin_last_fc_module = Middle_layer()
            self.origin_last_fc_module.bn2 = self.original_model.bn2
            self.origin_last_fc_module.dropout = self.original_model.dropout
            self.origin_last_fc_module.fc = self.original_model.fc
            self.origin_last_fc_module.bn3 = self.original_model.bn3
            net_origin.add_module(str(len(net_origin)), self.origin_last_fc_module)
            block_count += 1
            self.logger.info("enter block: {}".format(type(self.origin_last_fc_module)))

        elif self.settings.net_type in ["vgg"]:
            for ori_module, pruned_module in zip(self.original_model.modules(), self.pruned_model.modules()):
                if isinstance(ori_module, nn.ReLU):
                    self.logger.info("enter block: {}".format(type(ori_module)))
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        assert False, "shallow model is None"

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        assert False, "shallow model is None"
                    block_count += 1

                    if block_count in self.settings.pivot_set:
                        self.ori_segments.append(net_origin)
                        self.pruned_segments.append(net_pruned)
                        net_origin = None
                        net_pruned = None
                elif isinstance(ori_module, (nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.AvgPool2d)):
                    if net_origin is not None:
                        net_origin.add_module(str(len(net_origin)), ori_module)
                    else:
                        net_origin = nn.Sequential(ori_module)

                    if net_pruned is not None:
                        net_pruned.add_module(str(len(net_pruned)), pruned_module)
                    else:
                        net_pruned = nn.Sequential(pruned_module)
        self.final_block_count = block_count

        self.ori_segments.append(net_origin)
        self.pruned_segments.append(net_pruned)
        self.logger.info("self.final_block_count = {:d}".format(self.final_block_count))
        self.logger.info("self.ori_segments:{}".format(self.ori_segments))
        self.logger.info("self.pruned_segments:{}".format(self.pruned_segments))

    # def replace_fc_with_conv1x1(self, fc_layer):
    #     temp_conv = nn.Conv2d(
    #         in_channels=fc_layer.in_features,
    #         out_channels=fc_layer.out_features,
    #         kernel_size=1,
    #         stride=1,
    #         padding=0,
    #         bias=(fc_layer.bias is not None))
    #     layer_weight = fc_layer.weight.data
    #     reshape_layer_weight = layer_weight.reshape(fc_layer.out_features, fc_layer.in_features, 1, 1)
    #     temp_conv.weight.data.copy_(reshape_layer_weight.data)
    #
    #     if fc_layer.bias is not None:
    #         temp_conv.bias.data.copy_(fc_layer.bias.data)
    #     return temp_conv

    def create_auxiliary_classifiers(self):
        # create auxiliary classifier
        num_classes = self.settings.n_classes
        in_channels = 0

        if self.settings.net_type not in ["LResnetxE-IR", "mobilefacenet_v1"]:
            for i in range(len(self.pruned_segments) - 1):
                if isinstance(self.pruned_segments[i][-1], (PreBasicBlock, BasicBlock)):
                    in_channels = self.pruned_segments[i][-1].conv2.out_channels
                elif isinstance(self.pruned_segments[i][-1], Bottleneck):
                    in_channels = self.pruned_segments[i][-1].conv3.out_channels
                elif isinstance(self.pruned_segments[i][-1], nn.ReLU) and self.settings.net_type in ["vgg"]:
                    if isinstance(self.pruned_segments[i][-2], nn.Conv2d):
                        in_channels = self.pruned_segments[i][-2].out_channels
                    elif isinstance(self.pruned_segments[i][-3], nn.Conv2d):
                        in_channels = self.pruned_segments[i][-3].out_channels
                else:
                    self.logger.error("Nonsupport layer type!")
                    assert False, "Nonsupport layer type!"
                assert in_channels != 0, "in_channels is zero"

                self.aux_fc.append(AuxClassifier(in_channels=in_channels, num_classes=num_classes))

            pruned_final_fc = None
            if self.settings.net_type == "preresnet":
                pruned_final_fc = nn.Sequential(*[
                    self.pruned_model.bn,
                    self.pruned_model.relu,
                    self.pruned_model.avg_pool,
                    View(),
                    self.pruned_model.fc])
            elif self.settings.net_type == "resnet":
                pruned_final_fc = nn.Sequential(*[
                    self.pruned_model.avgpool,
                    View(),
                    self.pruned_model.fc])
            elif self.settings.net_type in ["vgg"]:
                pruned_final_fc = nn.Sequential(*[
                    View(),
                    self.pruned_model.classifier])
            else:
                self.logger.error("Nonsupport net type: {}!".format(self.settings.net_type))
                assert False, "Nonsupport net type: {}!".format(self.settings.net_type)
            self.aux_fc.append(pruned_final_fc)

        elif self.settings.net_type == "LResnetxE-IR":
            for i in range(len(self.pruned_segments)):

                # for insightface_dcp
                if i != len(self.pruned_segments)-1:
                    if isinstance(self.pruned_segments[i][-1], IRBlock):
                        in_channels = self.pruned_segments[i][-1].conv2.out_channels
                        self.channel_array.append(in_channels)
                    else:
                        self.logger.error("Nonsupport layer type!")
                        assert False, "Nonsupport layer type!"
                    assert in_channels != 0, "in_channels is zero"

                else:
                    if isinstance(self.pruned_segments[i][-2], IRBlock):
                        in_channels = self.pruned_segments[i][-2].conv2.out_channels
                        self.channel_array.append(in_channels)
                    else:
                        self.logger.error("Nonsupport layer type!")
                        assert False, "Nonsupport layer type!"
                    assert in_channels != 0, "in_channels is zero"

                self.middle_layer.append(Middle_layer(in_channels=in_channels, feature_size=self.feature_size[i]))
                if i == len(self.pruned_segments)-1:
                    self.logger.info('i={:d} middle layer load self.model finished!'.format(i+1))
                    self.middle_layer[i].bn2 = self.pruned_model.bn2
                    self.middle_layer[i].dropout = self.pruned_model.dropout
                    self.middle_layer[i].fc = self.pruned_model.fc
                    self.middle_layer[i].bn3 = self.pruned_model.bn3

                self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))

        elif self.settings.net_type == "mobilefacenet_v1":
            for i in range(len(self.pruned_segments)):
                # for insightface_dcp
                if i != len(self.pruned_segments) - 1:
                    if isinstance(self.pruned_segments[i][-1], Bottleneck_mobilefacenet):
                        in_channels = self.pruned_segments[i][-1].conv3.out_channels
                        self.channel_array.append(in_channels)
                    else:
                        self.logger.error("Nonsupport layer type!")
                        assert False, "Nonsupport layer type!"
                    assert in_channels != 0, "in_channels is zero"

                else:
                    # if isinstance(self.pruned_segments[i][-2], Bottleneck_mobilefacenet):
                    #     in_channels = self.pruned_segments[i][-2].conv3.out_channels
                    #     self.channel_array.append(in_channels)
                    # else:
                    #     self.logger.error("Nonsupport layer type!")
                    #     assert False, "Nonsupport layer type!"
                    # assert in_channels != 0, "in_channels is zero"

                    if isinstance(self.pruned_segments[i][-1], Middle_layer_mobilefacenetv1):
                        in_channels = self.pruned_segments[i][-1].linear.out_features
                        self.channel_array.append(in_channels)
                    else:
                        self.logger.error("Nonsupport layer type!")
                        assert False, "Nonsupport layer type!"
                    assert in_channels != 0, "in_channels is zero"



                self.middle_layer.append(Middle_layer_mobilefacenetv1(in_channels=in_channels,
                                                                          feature_size=self.feature_size[i]))
                if i == len(self.pruned_segments) - 1:
                    self.logger.info('i={:d}'.format(i))
                    self.middle_layer[i].conv3 = self.pruned_model.conv3
                    self.middle_layer[i].bn3 = self.pruned_model.bn3
                    self.middle_layer[i].prelu3 = self.pruned_model.prelu3
                    self.middle_layer[i].conv4 = self.pruned_model.conv4
                    self.middle_layer[i].bn4 = self.pruned_model.bn4
                    self.middle_layer[i].linear = self.pruned_model.linear
                    self.middle_layer[i].bn5 = self.pruned_model.bn5

                self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))

        self.logger.info("self.middle_layer:{}".format(self.middle_layer))
        self.logger.info("self.aux_fc:{}".format(self.aux_fc))
        self.logger.info("self.channel_array:{}".format(self.channel_array))

    def model_parallelism(self):
        self.ori_segments = utils.data_parallel(model=self.ori_segments, n_gpus=self.settings.n_gpus)
        self.pruned_segments = utils.data_parallel(model=self.pruned_segments, n_gpus=self.settings.n_gpus)

        self.middle_layer = utils.data_parallel(model=self.middle_layer, n_gpus=self.settings.n_gpus)
        self.aux_fc = utils.data_parallel(model=self.aux_fc, n_gpus=1)

    def create_optimizers(self):
        # create optimizers
        for i in range(len(self.pruned_segments)):
            temp_optim = []
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            for j in range(i + 1):
                temp_optim.append({'params': self.pruned_segments[j].parameters(),
                                   'lr': self.settings.lr})

            # optimizer for segments and fc
            temp_seg_optim = torch.optim.SGD(
                temp_optim,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)

            temp_fc_optim = torch.optim.SGD(
                params=self.aux_fc[i].parameters(),
                lr=self.settings.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)

            self.seg_optimizer.append(temp_seg_optim)
            self.fc_optimizer.append(temp_fc_optim)

    def insert_additional_losses(self):
        """"
            1. Split the network into several segments with pre-define pivot set
            2. Create auxiliary classifiers
            3. Create optimizers for network segments and fcs
        """

        self.create_segments()
        self.create_auxiliary_classifiers()

        # parallel setting for segments and auxiliary classifiers
        self.model_parallelism()
        self.create_optimizers()

    @staticmethod
    def _convert_results(top1_error, top1_loss, top5_error):
        """
        Convert tensor list to float list
        :param top1_error: top1_error tensor list
        :param top1_loss:  top1_loss tensor list
        :param top5_error:  top5_error tensor list
        """

        assert isinstance(top1_error, list), "input should be a list"
        length = len(top1_error)
        top1_error_list = []
        top5_error_list = []
        top1_loss_list = []
        for i in range(length):
            top1_error_list.append(top1_error[i].avg)
            top5_error_list.append(top5_error[i].avg)
            top1_loss_list.append(top1_loss[i].avg)
        return top1_error_list, top1_loss_list, top5_error_list

    def update_model(self, original_model, pruned_model, aux_fc_state=None, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update model parameter and optimizer state
        :param original_model: baseline model
        :param pruned_model: pruned model
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        self.ori_segments = []
        self.pruned_segments = []
        self.seg_optimizer = []
        self.aux_fc = []
        self.fc_optimizer = []

        self.original_model = original_model
        self.pruned_model = pruned_model
        self.insert_additional_losses()
        if aux_fc_state is not None:
            self.update_aux_fc(aux_fc_state, aux_fc_opt_state, seg_opt_state)

    def update_aux_fc(self, middle_layer_state, aux_fc_state, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update auxiliary classifier parameter and optimizer state
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        if len(self.aux_fc) == 1:
            if isinstance(self.aux_fc[0], nn.DataParallel):
                self.aux_fc[0].module.load_state_dict(aux_fc_state[-1])
            else:
                self.aux_fc[0].load_state_dict(aux_fc_state[-1])
            if aux_fc_opt_state is not None:
                self.fc_optimizer[0].load_state_dict(aux_fc_opt_state[-1])
            if seg_opt_state is not None:
                self.seg_optimizer[0].load_state_dict(seg_opt_state[-1])

        elif len(self.aux_fc) == len(aux_fc_state):
            for i in range(len(aux_fc_state)):

                if isinstance(self.middle_layer[i], nn.DataParallel):

                    self.middle_layer[i].module.load_state_dict(middle_layer_state[i])
                else:
                    self.middle_layer[i].load_state_dict(middle_layer_state[i])
                self.logger.info("trainer: middle_layer load middle_layer_state!!!")

                if isinstance(self.aux_fc[i], nn.DataParallel):
                    self.aux_fc[i].module.load_state_dict(aux_fc_state[i])
                else:
                    self.aux_fc[i].load_state_dict(aux_fc_state[i])
                self.logger.info("trainer: aux_fc load aux_fc_state!!!")

                if aux_fc_opt_state is not None:
                    self.fc_optimizer[i].load_state_dict(aux_fc_opt_state[i])
                if seg_opt_state is not None:
                    self.seg_optimizer[i].load_state_dict(seg_opt_state[i])
        else:
            assert False, "size not match! len(self.aux_fc)={:d}, len(aux_fc_state)={:d}".format(
                len(self.aux_fc), len(aux_fc_state))

    def auxnet_forward(self, images, labels=None):
        """
        Forward propagation
        """

        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.pruned_segments)):
            # forward
            temp_output = self.pruned_segments[i](temp_input)
            fcs_output = self.aux_fc[i](temp_output)
            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    @staticmethod
    def _correct_nan(grad):
        """
        Fix nan
        :param grad: gradient input
        """

        grad.masked_fill_(grad.ne(grad), 0)
        return grad

    def auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.seg_optimizer[i].zero_grad()
        self.fc_optimizer[i].zero_grad()

        loss.backward(retain_graph=True)
        # set the gradient of unselected channel to zero
        for j in range(len(self.pruned_segments)):
            if isinstance(self.pruned_segments[i], nn.DataParallel):
                for module in self.pruned_segments[j].module.modules():
                    if isinstance(module, MaskConv2d) and module.pruned_weight.grad is not None:
                        module.pruned_weight.grad.data.mul_(
                            module.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(module.pruned_weight))
            else:
                for module in self.pruned_segments[j].modules():
                    if isinstance(module, MaskConv2d) and module.pruned_weight.grad is not None:
                        module.pruned_weight.grad.data.mul_(
                            module.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(module.pruned_weight))

        self.fc_optimizer[i].step()
        self.seg_optimizer[i].step()

    def update_lr(self, epoch):
        """
        Update learning rate of optimizers
        :param epoch: index of epoch
        """

        gamma = 0
        for step in self.settings.step:
            if epoch + 1.0 > int(step):
                gamma += 1
        lr = self.settings.lr * math.pow(0.1, gamma)
        self.lr = lr

        for i in range(len(self.seg_optimizer)):
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = lr

            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.pruned_segments)
        for i in range(num_segments):
            self.pruned_segments[i].eval()
            self.aux_fc[i].eval()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        iters = len(self.val_loader)

        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                start_time = time.time()
                data_time = start_time - end_time

                if self.settings.n_gpus == 1:
                    images = images.cuda()
                labels = labels.cuda()

                outputs, losses = self.auxnet_forward(images, labels)

                # compute loss and error rate
                single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                    outputs=outputs, labels=labels,
                    loss=losses, top5_flag=True)

                for j in range(num_segments):
                    top1_error[j].update(single_error[j], images.size(0))
                    top5_error[j].update(single5_error[j], images.size(0))
                    top1_loss[j].update(single_loss[j], images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

                utils.print_result(epoch, 1, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Validation",
                                   logger=self.logger)

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)

        if self.logger is not None:
            for i in range(num_segments):
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_top1_error_{:d}".format(i), top1_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_top5_error_{:d}".format(i), top5_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "segment_wise_fine_tune_val_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list


    ############################### insightface_dcp ###################################
    def val_evaluate(self, i, segments, middle_layer,  carray, issame, emb_size=512, batch_size=512, nrof_folds=5):
        idx = 0
        embeddings = np.zeros([len(carray), emb_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                tem_batch = batch
                for j in range(i+1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out

                # self.logger.info('tem_batch shape={}'.format(tem_batch.size()))
                # out = middle_layer[i](tem_batch)
                if i != len(self.pruned_segments)-1:
                    out = middle_layer[i](tem_batch)
                    # self.logger.info("i={}, across middle_layer".format(i))
                else:
                    out = tem_batch
                    # self.logger.info("i={}, don't across middle_layer".format(i))

                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
                idx += batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                tem_batch = batch
                for j in range(i+1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out

                # out = middle_layer[i](tem_batch)
                if i != len(self.pruned_segments)-1:
                    out = middle_layer[i](tem_batch)
                else:
                    out = tem_batch

                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm

        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        # buf = gen_plot(fpr, tpr)
        # roc_curve = Image.open(buf)
        # roc_curve_tensor = transforms.ToTensor()(roc_curve)
        return accuracy.mean()

    def face_val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        lfw_val_acc = []
        # cfp_val_acc = []
        # age_val_acc = []

        num_segments = len(self.pruned_segments)
        for i in range(num_segments):
            self.pruned_segments[i].eval()
            self.middle_layer[i].eval()
            # lfw_val_acc.append(utils.AverageMeter())

        with torch.no_grad():
            for i in range(num_segments):
                # self.logger.info("i={:d}".format(i))
                lfw_accuracy = self.val_evaluate(i,
                                                 self.pruned_segments,
                                                 self.middle_layer,
                                                 self.val_loader[-1]['lfw'],
                                                 self.val_loader[-1]['lfw_issame'],
                                                 emb_size=self.channel_array[i])
                # self.logger.info("lfw_accuracy={:3f}".format(lfw_accuracy))
                lfw_val_acc.append(lfw_accuracy)

                # cfp_accuracy = self.val_evaluate(i,
                #                                  self.pruned_segments,
                #                                  self.middle_layer,
                #                                  self.val_loader[-2]['cfp_fp'],
                #                                  self.val_loader[-2]['cfp_fp_issame'],
                #                                  emb_size=self.channel_array[i])
                # cfp_val_acc.append(cfp_accuracy)
                #
                # age_accuracy = self.val_evaluate(i,
                #                                  self.pruned_segments,
                #                                  self.middle_layer,
                #                                  self.val_loader[-3]['agedb_30'],
                #                                  self.val_loader[-3]['agedb_30_issame'],
                #                                  emb_size=self.channel_array[i])
                # age_val_acc.append(age_accuracy)

        if self.logger is not None:
            for i in range(num_segments):
                self.logger.info('{:d}_auxnet_val_lfw_acc ={:.4f}'.format(i, lfw_val_acc[i]))
                self.tensorboard_logger.scalar_summary("auxnet_val_lfw_acc_{:d}".format(i), lfw_val_acc[i], self.run_count)

                # self.logger.info('{:d}_auxnet_val_cfp_acc ={:.4f}'.format(i, cfp_val_acc[i]))
                # self.tensorboard_logger.scalar_summary("auxnet_val_cfp_acc_{:d}".format(i), cfp_val_acc[i],
                #                                        self.run_count)
                #
                # self.logger.info('{:d}_auxnet_val_age_acc ={:.4f}'.format(i, age_val_acc[i]))
                # self.tensorboard_logger.scalar_summary("auxnet_val_age_acc_{:d}".format(i), age_val_acc[i],
                #                                        self.run_count)

        self.run_count += 1
        self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_val_acc[-1]))
        # self.logger.info("|===>Validation cfp acc: {:4f}".format(cfp_val_acc[-1]))
        # self.logger.info("|===>Validation age acc: {:4f}".format(age_val_acc[-1]))
        return lfw_val_acc

    # def face_val_before_train(self, epoch):
    #     """
    #     Validation
    #     :param epoch: index of epoch
    #     """
    #     lfw_val_acc = []
    #     num_segments = len(self.pruned_segments)
    #     for i in range(num_segments):
    #         self.pruned_segments[i].eval()
    #         self.middle_layer[i].eval()
    #         # lfw_val_acc.append(utils.AverageMeter())
    #
    #     with torch.no_grad():
    #         for i in range(num_segments):
    #             # self.logger.info("i={:d}".format(i))
    #             lfw_accuracy = self.val_evaluate(i,
    #                                              self.pruned_segments,
    #                                              self.middle_layer,
    #                                              self.val_loader[-1]['lfw'],
    #                                              self.val_loader[-1]['lfw_issame'],
    #                                              emb_size=self.channel_array[i])
    #             # self.logger.info("lfw_accuracy={:3f}".format(lfw_accuracy))
    #             lfw_val_acc.append(lfw_accuracy)
    #
    #     if self.logger is not None:
    #         for i in range(num_segments):
    #             self.logger.info('{:d}_auxnet_val_lfw_acc ={:.4f}'.format(i, lfw_val_acc[i]))
    #
    #     self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_val_acc[-1]))
    #     return lfw_val_acc