import math
import time

import torch.autograd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator, background

import dcp.utils as utils
from dcp.utils.verifacation import evaluate, l2_norm
from dcp.aux_classifier import AuxClassifier
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck

from dcp.models.insightface_resnet import IRBlock
from dcp.middle_layer import Middle_layer
from dcp.arcface import Arcface

from dcp.models.mobilefacenet import Bottleneck_mobilefacenet
from dcp.middle_layer import Middle_layer_mobilefacenetv1
from dcp.middle_layer import Middle_layer_gnap

from dcp.models.zq_mobilefacenet import ZQBottleneck_mobilefacenet
from dcp.models.mobilefacenetv3_v4_width import Block as mobilenetv3_block

from dcp.utils.ver_data import count_roc, count_scores

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


class AuxTrainer(object):
    """
    Trainer for auxnet
    """

    def __init__(self, model, train_loader, val_loader, test_loader, settings, logger, tensorboard_logger, run_count=0):

        self.model = model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.lr = self.settings.lr
        self.segments = []
        self.aux_fc = []
        self.middle_layer = []

        self.seg_optimizer = []
        self.fc_optimizer = []
        self.run_count = run_count

        self.channel_array = []
        # run pre-processing
        self.set_loss_weight()

        # insightface_dcp
        if self.settings.net_type in ["LResnetxE-IR"] :
            if self.settings.n_losses == 2:
                self.feature_size = [28, 14, 7]
            else:
                self.logger.info("{}, self.settings.n_losses={}".format(self.settings.net_type, self.settings.n_losses))
                assert False

        elif self.settings.net_type in ["mobilefacenet_v1", "mobilenet_v3"]:
            if self.settings.n_losses == 2:
                self.feature_size = [14, 14, 7]
            else:
                self.logger.info("{}, self.settings.n_losses={}".format(self.settings.net_type, self.settings.n_losses))
                assert False

        elif self.settings.net_type in ["zq_mobilefacenet"]:
            if self.settings.n_losses == 2:
                self.feature_size = [28, 14, 7]
            else:
                self.logger.info("{}, self.settings.n_losses={}".format(self.settings.net_type, self.settings.n_losses))
                assert False

        self.insert_additional_losses()

    def insert_additional_losses(self):
        """"
        1. Split the network into several segments with pre-define pivot set
        2. Create auxiliary classifiers
        3. Create optimizers for network segments and fcs
        """

        self.create_segments()
        # self.logger.info(len(self.segments))
        # assert False
        self.create_auxiliary_classifiers()

        # parallel setting for segments and auxiliary classifiers
        self.model_parallelism()

        if self.settings.net_type not in ["LResnetxE-IR", "mobilefacenet_v1", "zq_mobilefacenet", "mobilenet_v3"]:
            self.create_optimizers()

        elif self.settings.net_type in ["LResnetxE-IR", "mobilefacenet_v1", "zq_mobilefacenet","mobilenet_v3"]:
            # self.face_part_create_optimizers()
            self.face_all_create_optimizers()

    def set_loss_weight(self):
        """
        The weight of the k-th auxiliary loss: gamma_k = \max(0.01, (\frac{L_k}{L_K})^2)
        More details can be found in Section 3.2 in "The Shallow End: Empowering Shallower Deep-Convolutional Networks
        through Auxiliary Outputs": https://arxiv.org/abs/1611.01773.
        """

        base_weight = 0
        self.lr_weight = torch.zeros(len(self.settings.pivot_set)).cuda()
        self.pivot_weight = self.lr_weight.clone()
        num_layers = 1
        if self.settings.net_type in ["preresnet", "LResnetxE-IR"] or \
                (self.settings.net_type == "resnet" and self.settings.depth < 50):
            num_layers = 2
        elif self.settings.net_type == "resnet" and self.settings.depth >= 50:
            num_layers = 3
        elif self.settings.net_type in ["zq_mobilefacenet", "mobilefacenet_v1"]:
            num_layers = 3
        elif self.settings.net_type in ["vgg"]:
            num_layers = 1
        for i in range(len(self.settings.pivot_set) - 1, -1, -1):  # 倒序赋值给i
            temp_weight = max(pow(float(self.settings.pivot_set[i] * num_layers + 1) /
                                  (self.settings.pivot_set[-1] * num_layers + 1), 2), 0.01)
            base_weight += temp_weight
            self.pivot_weight[i] = temp_weight
            self.lr_weight[i] = base_weight

    def create_segments(self):
        """
        Split the network into several segments with pre-define pivot set
        """

        shallow_model = None

        # for head
        if self.settings.net_type in ["preresnet", "resnet", "LResnetxE-IR", "mobilefacenet_v1",
                                      "zq_mobilefacenet","mobilenet_v3"]:
            if self.settings.net_type == "preresnet":
                shallow_model = nn.Sequential(self.model.conv)
            elif self.settings.net_type == "resnet":
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool)
                shallow_model = nn.Sequential(net_head)
            elif self.settings.net_type == "LResnetxE-IR":
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.prelu,
                    self.model.maxpool)
                shallow_model = nn.Sequential(net_head)
            elif self.settings.net_type == "mobilefacenet_v1":
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.prelu1,
                    self.model.conv2,
                    self.model.bn2,
                    self.model.prelu2)
                shallow_model = nn.Sequential(net_head)

            elif self.settings.net_type in ["zq_mobilefacenet", "mobilefacenet_v2"]:
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.prelu1)
                shallow_model = nn.Sequential(net_head)

            elif self.settings.net_type in ["mobilenet_v3"]:
                net_head = nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.hs1,
                    self.model.conv2,
                    self.model.bn2,
                    self.model.hs2)
                shallow_model = nn.Sequential(net_head)

        elif self.settings.net_type in ["vgg"]:
            shallow_model = None

        else:
            self.logger.info("unsupported net_type: {}".format(self.settings.net_type))
            assert False, "unsupported net_type: {}".format(self.settings.net_type)
        self.logger.info("init shallow head done!")

        # for middle block
        block_count = 0
        if self.settings.net_type in ["resnet", "preresnet", "LResnetxE-IR", "mobilefacenet_v1",
                                      "zq_mobilefacenet", "mobilenet_v3"]:
            for module in self.model.modules():
                if isinstance(module, (PreBasicBlock, Bottleneck, BasicBlock, IRBlock,
                                       Bottleneck_mobilefacenet, ZQBottleneck_mobilefacenet, mobilenetv3_block)):
                    self.logger.info("enter block: {}".format(type(module)))
                    if shallow_model is not None:
                        shallow_model.add_module(str(len(shallow_model)), module)
                    else:
                        shallow_model = nn.Sequential(module)
                    block_count += 1

                    # if block_count is equals to pivot_num, then create new segment
                    if block_count in self.settings.pivot_set:
                        self.segments.append(shallow_model)
                        shallow_model = None
        elif self.settings.net_type in ["vgg"]:
            for module in self.model.features.modules():
                if isinstance(module, nn.ReLU):
                    if shallow_model is not None:
                        shallow_model.add_module(str(len(shallow_model)), module)
                        block_count += 1
                    else:
                        assert False, "shallow model is None"
                    if block_count in self.settings.pivot_set:
                        self.segments.append(shallow_model)
                        shallow_model = None
                elif isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.AvgPool2d)):
                    if shallow_model is None:
                        shallow_model = nn.Sequential(module)
                    else:
                        shallow_model.add_module(str(len(shallow_model)), module)

        self.final_block_count = block_count
        self.segments.append(shallow_model)
        self.logger.info("block_count={:d}".format(block_count))
        self.logger.info("self.segments:{}".format(self.segments))

    def create_auxiliary_classifiers(self):
        """
        We insert the auxiliary classifiers after the convolutional layer.
        """

        num_classes = self.settings.n_classes
        in_channels = 0

        if self.settings.net_type not in ["LResnetxE-IR", "mobilefacenet_v1",
                                          "zq_mobilefacenet", "mobilefacenet_v2", "mobilenet_v3"]:
            for i in range(len(self.segments) - 1):
                if isinstance(self.segments[i][-1], (PreBasicBlock, BasicBlock)):
                    in_channels = self.segments[i][-1].conv2.out_channels
                elif isinstance(self.segments[i][-1], Bottleneck):
                    in_channels = self.segments[i][-1].conv3.out_channels
                elif isinstance(self.segments[i][-1], nn.ReLU) and self.settings.net_type in ["vgg"]:
                    if isinstance(self.segments[i][-2], nn.Conv2d):
                        in_channels = self.segments[i][-2].out_channels
                    elif isinstance(self.segments[i][-3], nn.Conv2d):
                        in_channels = self.segments[i][-3].out_channels
                else:
                    self.logger.error("Nonsupport layer type!")
                    assert False, "Nonsupport layer type!"
                assert in_channels != 0, "in_channels is zero"

                self.aux_fc.append(AuxClassifier(in_channels=in_channels, num_classes=num_classes))

            final_fc = None
            if self.settings.net_type == "preresnet":
                final_fc = nn.Sequential(*[
                    self.model.bn,
                    self.model.relu,
                    self.model.avg_pool,
                    View(),
                    self.model.fc])
            elif self.settings.net_type == "resnet":
                final_fc = nn.Sequential(*[
                    self.model.avgpool,
                    View(),
                    self.model.fc])
            elif self.settings.net_type in ["vgg"]:
                final_fc = nn.Sequential(*[
                    View(),
                    self.model.classifier])
            else:
                self.logger.error("Nonsupport net type: {}!".format(self.settings.net_type))
                assert False, "Nonsupport net type: {}!".format(self.settings.net_type)
            self.aux_fc.append(final_fc)

        elif self.settings.net_type == "LResnetxE-IR":
            for i in range(len(self.segments)):
                # for insightface_dcp
                if isinstance(self.segments[i][-1], IRBlock):
                    in_channels = self.segments[i][-1].conv2.out_channels
                    self.channel_array.append(in_channels)
                else:
                    self.logger.error("Nonsupport layer type!")
                    assert False, "Nonsupport layer type!"
                assert in_channels != 0, "in_channels is zero"
                self.middle_layer.append(Middle_layer(in_channels=in_channels, feature_size=self.feature_size[i]))

                if i == len(self.segments) - 1:
                    self.logger.info('i={:d} middle layer load self.model finished!'.format(i + 1))
                    self.middle_layer[i].bn2 = self.model.bn2
                    self.middle_layer[i].dropout = self.model.dropout
                    self.middle_layer[i].fc = self.model.fc
                    self.middle_layer[i].bn3 = self.model.bn3
                self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))

        elif self.settings.net_type in ["mobilefacenet_v1"]:
            for i in range(len(self.segments)):
                # for insightface_dcp
                if isinstance(self.segments[i][-1], (Bottleneck_mobilefacenet)):
                    in_channels = self.segments[i][-1].conv3.out_channels
                    self.channel_array.append(in_channels)
                else:
                    self.logger.error("Nonsupport layer type!")
                    assert False, "Nonsupport layer type!"
                assert in_channels != 0, "in_channels is zero"

                self.middle_layer.append(Middle_layer_mobilefacenetv1(in_channels=in_channels,
                                                                      feature_size=self.feature_size[i]))
                if i == len(self.segments) - 1:
                    if self.settings.net_type in ["mobilefacenet_v1"]:
                        self.logger.info('i={:d}'.format(i))
                        self.middle_layer[i].conv3 = self.model.conv3
                        self.middle_layer[i].bn3 = self.model.bn3
                        self.middle_layer[i].prelu3 = self.model.prelu3
                        self.middle_layer[i].conv4 = self.model.conv4
                        self.middle_layer[i].bn4 = self.model.bn4
                        self.middle_layer[i].linear = self.model.linear
                        self.middle_layer[i].bn5 = self.model.bn5


                if self.settings.loss_type == "softmax-arcface":
                    if i != len(self.segments) - 1:
                        self.aux_fc.append(nn.Linear(in_channels, num_classes))
                    elif i == len(self.segments) - 1:
                        self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))
                    else:
                        assert False
                elif self.settings.loss_type == "softmax":
                    self.aux_fc.append(nn.Linear(in_channels, num_classes))

                elif self.settings.loss_type == "arcface":
                    self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))
                else:
                    assert False,"Not support {}".format(self.settings.loss_type)

        elif self.settings.net_type in ["zq_mobilefacenet", "mobilefacenet_v2", "mobilenet_v3"]:

            for i in range(len(self.segments)):
                # for insightface_dcp
                if isinstance(self.segments[i][-1], (ZQBottleneck_mobilefacenet, mobilenetv3_block)):
                    in_channels = self.segments[i][-1].conv3.out_channels
                    self.channel_array.append(in_channels)
                else:
                    self.logger.error("Nonsupport layer type!")
                    assert False, "Nonsupport layer type!"
                assert in_channels != 0, "in_channels is zero"

                if self.settings.fc_type in ["gnap"]:
                    self.middle_layer.append(Middle_layer_gnap(in_channels=in_channels,
                                                               feature_size=self.feature_size[i],
                                                               width_mult=self.settings.width_mult))
                elif self.settings.fc_type in ["gdc"]:
                    self.middle_layer.append(Middle_layer_mobilefacenetv1(in_channels=in_channels, feature_size=self.feature_size[i]))
                else:
                    assert False, "not support fc_type:{}".format(self.settings.fc_type)

                if i == len(self.segments) - 1:
                    if self.settings.net_type in ["zq_mobilefacenet"]:
                        self.logger.info('i={:d}'.format(i))
                        self.middle_layer[i].conv3 = self.model.conv2
                        self.middle_layer[i].bn3 = self.model.bn2
                        self.middle_layer[i].prelu3 = self.model.prelu2

                    elif self.settings.net_type in ["mobilenet_v3"]:
                        self.logger.info('i={:d}'.format(i))
                        self.middle_layer[i].conv3 = self.model.conv3
                        self.middle_layer[i].bn3 = self.model.bn3
                        self.middle_layer[i].prelu3 = self.model.hs3

                    if self.settings.fc_type in ["gnap"]:
                        self.middle_layer[i].fc1 = self.model.fc1
                    elif self.settings.fc_type in ["gdc"]:
                        self.middle_layer[i].conv4 = self.model.conv3
                        self.middle_layer[i].bn4 = self.model.bn3
                        self.middle_layer[i].linear = self.model.linear
                        self.middle_layer[i].bn5 = self.model.bn4
                    else:
                        assert False, "not support fc_type:{}".format(self.settings.fc_type)

                if self.settings.net_type in ["mobilenet_v3"] and i == len(self.segments)-1:
                    in_channels = 512
                    self.channel_array[i] = in_channels

                if self.settings.loss_type == "softmax-arcface":
                    if i != len(self.segments) - 1:
                        self.aux_fc.append(nn.Linear(in_channels, num_classes))
                    elif i == len(self.segments) - 1:
                        self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))
                    else:
                        assert False
                elif self.settings.loss_type == "softmax":
                    self.aux_fc.append(nn.Linear(in_channels, num_classes))

                elif self.settings.loss_type == "arcface":
                    self.aux_fc.append(Arcface(in_channels=in_channels, num_classes=num_classes))
                else:
                    assert False,"Not support {}".format(self.settings.loss_type)

        self.logger.info("self.middle_layer:{}".format(self.middle_layer))
        self.logger.info("self.aux_fc:{}".format(self.aux_fc))
        self.logger.info("self.channel_array:{}".format(self.channel_array))

    def model_parallelism(self):
        self.segments = utils.data_parallel(model=self.segments, n_gpus=self.settings.n_gpus)
        self.middle_layer = utils.data_parallel(model=self.middle_layer, n_gpus=self.settings.n_gpus)
        self.aux_fc = utils.data_parallel(model=self.aux_fc, n_gpus=self.settings.n_gpus)

    def create_optimizers(self):
        """
        Create optimizers for network segments and fcs
        """
        for i in range(len(self.segments)):
            temp_optim = []
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            for j in range(i + 1):
                temp_optim.append({'params': self.segments[j].parameters(), 'lr': self.settings.lr})

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

    def update_model(self, model, aux_fc_state=None, aux_fc_opt_state=None, seg_opt_state=None):
        """
        Update model parameter and optimizer state
        :param model: model
        :param aux_fc_state: state dict of auxiliary fully-connected layer
        :param aux_fc_opt_state: optimizer's state dict of auxiliary fully-connected layer
        :param seg_opt_state: optimizer's state dict of segment
        """

        self.segments = []
        self.aux_fc = []
        self.seg_optimizer = []
        self.fc_optimizer = []

        self.model = model
        self.insert_additional_losses()
        if aux_fc_state is not None:
            self.update_aux_fc(aux_fc_state, aux_fc_opt_state, seg_opt_state)

    def update_aux_fc(self, aux_fc_state, aux_fc_opt_state=None, seg_opt_state=None):
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
                if isinstance(self.aux_fc[i], nn.DataParallel):
                    self.aux_fc[i].module.load_state_dict(aux_fc_state[i])
                else:
                    self.aux_fc[i].load_state_dict(aux_fc_state[i])

                if aux_fc_opt_state is not None:
                    self.fc_optimizer[i].load_state_dict(aux_fc_opt_state[i])
                if seg_opt_state is not None:
                    self.seg_optimizer[i].load_state_dict(seg_opt_state[i])
        else:
            assert False, "size not match! len(self.aux_fc)={:d}, len(aux_fc_state)={:d}".format(
                len(self.aux_fc), len(aux_fc_state))

    @staticmethod
    def adjustweight(lr_weight=1.0):
        """
        Adjust weight according to loss
        :param lr_weight: weight of the learning rate
        """

        return 1.0 / lr_weight

    def auxnet_forward(self, images, labels=None):
        """
        Forward propagation fot auxnet
        """

        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.segments)):
            # forward
            temp_output = self.segments[i](temp_input)
            fcs_output = self.aux_fc[i](temp_output)
            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    def auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.seg_optimizer[i].zero_grad()
        self.fc_optimizer[i].zero_grad()

        # lr = lr * (pivot_weight / lr_weight)
        if i < len(self.seg_optimizer) - 1:
            self.logger.info(i)
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = self.lr * self.adjustweight(self.lr_weight[i].item()) * self.pivot_weight[i]

            loss.backward(retain_graph=True)
            # for param_group in self.seg_optimizer[i].param_groups:
            #     for p in param_group['params']:
            #         if p.grad is None:
            #             continue
            #         p.grad.data.mul_(p.new([self.pivot_weight[i]]))
        else:
            loss.backward(retain_graph=True)

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

        for i in range(len(self.fc_optimizer)):
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = lr

            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    def train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        iters = len(self.train_loader)
        self.update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].train()
            self.aux_fc[i].train()
            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(self.train_loader):
            start_time = time.time()
            data_time = start_time - end_time

            if self.settings.n_gpus == 1:
                images = images.cuda()
            labels = labels.cuda()

            # forward
            outputs, losses = self.auxnet_forward(images, labels)
            # backward
            for j in range(len(self.seg_optimizer)):
                self.auxnet_backward_for_loss_i(losses[j], j)

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

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Train",
                                   logger=self.logger)

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            for i in range(num_segments):
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top1_error_{:d}".format(i), top1_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top5_error_{:d}".format(i), top5_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
            self.tensorboard_logger.scalar_summary("lr", self.lr, self.run_count)

        self.logger.info("|===>Training Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
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

                if i % self.settings.print_frequency == 0:
                    utils.print_result(epoch, self.settings.n_epochs, i + 1,
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
                    "auxnet_val_top1_error_{:d}".format(i), top1_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_val_top5_error_{:d}".format(i), top5_error[i].avg, self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_val_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
        self.run_count += 1

        self.logger.info("|===>Validation Error: {:4f}/{:4f}, Loss: {:4f}".format(
            top1_error[-1].avg, top5_error[-1].avg, top1_loss[-1].avg))
        return top1_error_list, top1_loss_list, top5_error_list

    ######################## for insightface_dcp: all layers update ###########################
    def face_all_create_optimizers(self):
        """
        Create optimizers for network segments and fcs
        """
        for i in range(len(self.segments)):
            temp_optim = []
            # add parameters in segmenets into optimizer
            # from the i-th optimizer contains [0:i] segments
            for j in range(i + 1):
                temp_optim.append({'params': self.segments[j].parameters(), 'lr': self.settings.lr})

            # optimizer for segments and fc
            temp_seg_optim = torch.optim.SGD(
                temp_optim,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay*0.1,
                nesterov=True)

            temp_fc_optim = torch.optim.SGD(
                [{'params': self.middle_layer[i].parameters()}, {'params': self.aux_fc[i].parameters()}],
                lr=self.settings.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)

            self.seg_optimizer.append(temp_seg_optim)
            self.fc_optimizer.append(temp_fc_optim)

    def face_auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.seg_optimizer[i].zero_grad()
        self.fc_optimizer[i].zero_grad()

        # self.logger.info('seg len: {}'.format(len(self.seg_optimizer)))
        # self.logger.info('fc len: {}'.format(len(self.fc_optimizer)))
        # self.logger.info('lr_weight len: {}'.format(len(self.lr_weight)))
        # self.logger.info('pivot len: {}'.format(len(self.pivot_weight)))

        # lr = lr * (pivot_weight / lr_weight)
        if i < len(self.seg_optimizer) - 1:
            # self.logger.info('{}-th'.format(i))
            for param_group in self.seg_optimizer[i].param_groups:
                param_group['lr'] = self.lr * self.adjustweight(self.lr_weight[i].item()) * self.pivot_weight[i]

            loss.backward(retain_graph=True)
            # for param_group in self.seg_optimizer[i].param_groups:
            #     for p in param_group['params']:
            #         if p.grad is None:
            #             continue
            #         p.grad.data.mul_(p.new([self.pivot_weight[i]]))
        else:
            loss.backward(retain_graph=True)

        self.fc_optimizer[i].step()
        self.seg_optimizer[i].step()

    def face_train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        iters = len(self.train_loader)
        self.update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].train()
            self.middle_layer[i].train()
            self.aux_fc[i].train()

            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time

        # use prefetch_generator and tqdm for iterating through data
        # pbar = enumerate(BackgroundGenerator(self.train_loader))

        for i, (images, labels) in enumerate(self.train_loader):
        # for i, (images, labels) in pbar:
            start_time = time.time()
            data_time = start_time - end_time

            # if self.settings.n_gpus == 1:
            images = images.cuda()
            labels = labels.cuda()

            # forward
            outputs, losses = self.face_auxnet_forward(images, labels)
            # backward
            for j in range(len(self.fc_optimizer)):
                self.face_auxnet_backward_for_loss_i(losses[j], j)

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

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Train",
                                   logger=self.logger)

            # break

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            for i in range(num_segments):
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top1_error_{:d}".format(i), top1_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top5_error_{:d}".format(i), top5_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)

                if i == num_segments-1:
                    self.tensorboard_logger.scalar_summary("Train_Top1_Accuracy",
                                                           (100-top1_error[i].avg)/100.0, self.run_count)
                    self.tensorboard_logger.scalar_summary("Train_Top5_Accuracy",
                                                           (100-top5_error[i].avg)/100.0, self.run_count)
                    self.tensorboard_logger.scalar_summary("Train_Loss", top1_loss[i].avg, self.run_count)

            self.tensorboard_logger.scalar_summary("Train_lr", self.lr, self.run_count)

        for i in range(num_segments):
            self.logger.info("|===>Training Error_{:d}: top1_error={:4f}/top5_error={:4f}, Loss: {:4f}".format(i,
                                                                                                               top1_error[
                                                                                                                   i].avg,
                                                                                                               top5_error[
                                                                                                                   i].avg,
                                                                                                               top1_loss[
                                                                                                                   i].avg))

        return top1_error_list, top1_loss_list, top5_error_list

    #############################################################################################

    ############### for insightface_dcp: additional_aux_fc update, others fixed ##################
    def fixed_update_lr(self, epoch):
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

        for i in range(len(self.fc_optimizer)):
            for param_group in self.fc_optimizer[i].param_groups:
                param_group['lr'] = lr

    def face_part_create_optimizers(self):
        """
        Create optimizers for network segments and fcs
        """
        # turn gradient off
        # avoid computing the gradient
        for params in self.model.parameters():
            params.requires_grad = False

        for i in range(len(self.segments)):
            for params in self.segments[i].parameters():
                params.requires_grad = False

            if i == len(self.segments) - 1:
                for params in self.middle_layer[i].parameters():
                    params.requires_grad = False
                for params in self.aux_fc[i].parameters():
                    params.requires_grad = False

        for i in range(len(self.segments) - 1):
            temp = [{'params': self.middle_layer[i].parameters()}, {'params': self.aux_fc[i].parameters()}]
            self.logger.info("i={:d}".format(i))
            temp_fc_optim = torch.optim.SGD(
                temp,
                lr=self.settings.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True)
            self.fc_optimizer.append(temp_fc_optim)
        self.logger.info("part update optimizer finished!!! \nself.fc_optimizer len={:d}".format(len(self.fc_optimizer)))

    def fixed_face_auxnet_backward_for_loss_i(self, loss, i):
        """
        Backward propagation for the i-th loss
        :param loss: the i-th loss
        :param i: which one to perform backward propagation
        """

        self.fc_optimizer[i].zero_grad()
        loss.backward()
        self.fc_optimizer[i].step()

    def fixed_face_auxnet_forward(self, images, labels=None):
        """
        Forward propagation fot auxnet
        """
        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.segments)-1):
            # forward
            with torch.no_grad():
                temp_output = self.segments[i](temp_input)
            middle_output = self.middle_layer[i](temp_output)
            fcs_output = self.aux_fc[i](middle_output, labels)

            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    def face_fixed_train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """
        iters = len(self.train_loader)
        self.fixed_update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments-1):
            self.segments[i].eval()
            self.middle_layer[i].train()
            self.aux_fc[i].train()

            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time

        # use prefetch_generator and tqdm for iterating through data
        pbar =enumerate(BackgroundGenerator(self.train_loader))

        # for i, (images, labels) in enumerate(self.train_loader):
        for i, (images, labels) in pbar:
            start_time = time.time()
            data_time = start_time - end_time

            # if self.settings.n_gpus == 1:
            images = images.cuda()
            labels = labels.cuda()

            # forward
            outputs, losses = self.fixed_face_auxnet_forward(images, labels)
            # backward
            for j in range(len(self.fc_optimizer)):
                self.fixed_face_auxnet_backward_for_loss_i(losses[j], j)

            # compute loss and error rate
            single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                outputs=outputs, labels=labels,
                loss=losses, top5_flag=True)

            for j in range(num_segments-1):
                top1_error[j].update(single_error[j], images.size(0))
                top5_error[j].update(single5_error[j], images.size(0))
                top1_loss[j].update(single_loss[j], images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Train",
                                   logger=self.logger)
            # break

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            for i in range(num_segments-1):
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top1_error_{:d}".format(i), top1_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_top5_error_{:d}".format(i), top5_error[i].avg,
                    self.run_count)
                self.tensorboard_logger.scalar_summary(
                    "auxnet_train_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
            self.tensorboard_logger.scalar_summary("lr", self.lr, self.run_count)

        for i in range(num_segments-1):
            self.logger.info("|===>Training Error_{:d}: top1_error={:4f}/top5_error={:4f}, Loss: {:4f}".format(i,
                                                                                                               top1_error[
                                                                                                                   i].avg,
                                                                                                               top5_error[
                                                                                                                   i].avg,
                                                                                                               top1_loss[
                                                                                                                   i].avg))

        return top1_error_list, top1_loss_list, top5_error_list

    ##############################################################################################
    def face_update_aux_fc(self, aux_fc_state, middle_layer_state, aux_fc_opt_state, count):
        self.logger.info("start load resume !!!")
        self.run_count = count
        if len(self.aux_fc) == len(aux_fc_state):
            for i in range(len(aux_fc_state)):
                if isinstance(self.aux_fc[i], nn.DataParallel):
                    self.aux_fc[i].module.load_state_dict(aux_fc_state[i])
                else:
                    self.aux_fc[i].load_state_dict(aux_fc_state[i])

                if isinstance(self.middle_layer[i], nn.DataParallel):
                    self.middle_layer[i].module.load_state_dict(middle_layer_state[i])
                else:
                    self.middle_layer[i].load_state_dict(middle_layer_state[i])

        if aux_fc_opt_state is not None:
            for i in range(len(aux_fc_opt_state)):
                self.fc_optimizer[i].load_state_dict(aux_fc_opt_state[i])
                self.logger.info("checkpoint load fc_optuimizer!!")

        else:
            assert False, "size not match! len(self.aux_fc)={:d}, len(aux_fc_state)={:d}".format(
                len(self.aux_fc), len(aux_fc_state))

    def face_finetune_softmax_to_arcface(self, aux_fc_state, middle_layer_state):
        self.logger.info("load fintune: aux_softmax + end softmax --> aux_softmax + end arcface:")
        if len(self.aux_fc) == len(aux_fc_state):
            for i in range(len(aux_fc_state)-1):
                self.logger.info("i={}".format(i))
                if isinstance(self.middle_layer[i], nn.DataParallel):
                    self.middle_layer[i].module.load_state_dict(middle_layer_state[i])
                else:
                    self.middle_layer[i].load_state_dict(middle_layer_state[i])
                self.logger.info("middle_layer load finished!!!")

                if isinstance(self.aux_fc[i], nn.DataParallel):
                    self.aux_fc[i].module.load_state_dict(aux_fc_state[i])
                else:
                    self.aux_fc[i].load_state_dict(aux_fc_state[i])
                self.logger.info("aux_fc load finished!!!")

    def face_update_final_aux_fc(self, final_aux_fc_state):
        if isinstance(self.aux_fc[-1], nn.DataParallel):
            self.aux_fc[-1].module.load_state_dict(final_aux_fc_state)
        else:
            self.aux_fc[-1].load_state_dict(final_aux_fc_state)
        self.logger.info("trainer load final aux_fc_state finish !!!")

    def face_auxnet_forward(self, images, labels=None):
        """
        Forward propagation fot auxnet
        """
        outputs = []
        temp_input = images
        losses = []
        for i in range(len(self.segments)):
            # forward
            temp_output = self.segments[i](temp_input)
            middle_output = self.middle_layer[i](temp_output)

            if self.settings.loss_type in ["softmax-arcface"]:
                if i != len(self.segments) - 1:
                    fcs_output = self.aux_fc[i](middle_output)
                elif i == len(self.segments) - 1:
                    fcs_output = self.aux_fc[i](middle_output, labels)

            elif self.settings.loss_type in ["softmax"]:
                fcs_output = self.aux_fc[i](middle_output)

            elif self.settings.loss_type in ["arcface"]:
                fcs_output = self.aux_fc[i](middle_output, labels)

            else:
                assert False, "Not support {}".format(self.settings.loss_type)

            outputs.append(fcs_output)
            if labels is not None:
                losses.append(self.criterion(fcs_output, labels))
            temp_input = temp_output
        return outputs, losses

    def val_evaluate(self, i, segments, middle_layer, carray, issame, emb_size=512, batch_size=512, nrof_folds=10):
        idx = 0
        # self.logger.info()
        embeddings = np.zeros([len(carray), emb_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                tem_batch = batch
                for j in range(i + 1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out
                # self.logger.info('tem_batch shape={}'.format(tem_batch.size()))
                out = middle_layer[i](tem_batch)
                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
                idx += batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                tem_batch = batch
                for j in range(i + 1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out
                out = middle_layer[i](tem_batch)
                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        return accuracy.mean()

    def val_tpr(self, i, segments, middle_layer, carray, issame, emb_size=512, batch_size=512, far=10 ** -3):
        # segments.eval()
        # middle_layer.eval()
        # 提取特征
        idx = 0
        embeddings = np.zeros([len(carray), emb_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                tem_batch = batch
                for j in range(i + 1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out
                out = middle_layer(tem_batch)
                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
                idx += batch_size

            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                tem_batch = batch
                for j in range(i + 1):
                    model_out = segments[j](tem_batch.cuda())
                    tem_batch = model_out
                out = middle_layer(tem_batch)
                embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm

        # 计算cos similarly
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        scores = count_scores(embeddings1, embeddings2)

        # count roc
        results = count_roc(issame, scores, far)
        return float(results[0])

    def face_val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        lfw_val_acc = []
        cfp_val_acc = []
        age_val_acc = []

        tpr_acc = 0
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
            self.middle_layer[i].eval()
            # lfw_val_acc.append(utils.AverageMeter())

        with torch.no_grad():
            for i in range(len(self.segments)):
                # self.logger.info("i={:d}".format(i))

                if i == len(self.segments)-1:
                    self.logger.info("i={}".format(i))
                    tpr_acc = self.val_tpr(i,
                                           self.segments,
                                           self.middle_layer[i],
                                           self.val_loader[-1]['verification'],
                                           self.val_loader[-1]['verification_issame'],
                                           emb_size=self.channel_array[i], far=10e-3)
                    self.logger.info("verification finished!!!")
                    lfw_accuracy = self.val_evaluate(i,
                                                     self.segments,
                                                     self.middle_layer,
                                                     self.val_loader[-2]['lfw'],
                                                     self.val_loader[-2]['lfw_issame'],
                                                     emb_size=self.channel_array[i])
                    lfw_val_acc.append(lfw_accuracy)
                    self.logger.info("lfw test finished!!!")

                    cfp_accuracy = self.val_evaluate(i,
                                                     self.segments,
                                                     self.middle_layer,
                                                     self.val_loader[-3]['cfp_fp'],
                                                     self.val_loader[-3]['cfp_fp_issame'],
                                                     emb_size=self.channel_array[i])
                    cfp_val_acc.append(cfp_accuracy)
                    self.logger.info("cfp_fp test finished!!!")

                    age_accuracy = self.val_evaluate(i,
                                                     self.segments,
                                                     self.middle_layer,
                                                     self.val_loader[-4]['agedb_30'],
                                                     self.val_loader[-4]['agedb_30_issame'],
                                                     emb_size=self.channel_array[i])
                    age_val_acc.append(age_accuracy)
                    self.logger.info("agedb_30 test finished!!!")

                else:
                    continue

        if self.logger is not None:
            # for i in range(num_segments):
            #     if i == num_segments - 1:
            # self.logger.info('{:d}_auxnet_val_lfw_acc ={:.4f}'.format(lfw_val_acc[i]))
            # self.tensorboard_logger.scalar_summary("auxnet_val_lfw_acc_{:d}".format(i), lfw_val_acc[i], self.run_count)
            #
            # self.logger.info('{:d}_auxnet_val_cfp_acc ={:.4f}'.format(i, cfp_val_acc[i]))
            # self.tensorboard_logger.scalar_summary("auxnet_val_cfp_acc_{:d}".format(i), cfp_val_acc[i],
            #                                        self.run_count)
            #
            # self.logger.info('{:d}_auxnet_val_age_acc ={:.4f}'.format(i, age_val_acc[i]))
            # self.tensorboard_logger.scalar_summary("auxnet_val_age_acc_{:d}".format(i), age_val_acc[i], self.run_count)

            # self.logger.info('val_lfw_accuracy'.format(lfw_val_acc[i]))
            self.tensorboard_logger.scalar_summary("val_lfw_accuracy", lfw_val_acc[-1]*100, self.run_count)

            # self.logger.info('{:d}_auxnet_val_cfp_acc ={:.4f}'.format(i, cfp_val_acc[i]))
            self.tensorboard_logger.scalar_summary("val_cfp_fp_accuracy", cfp_val_acc[-1]*100, self.run_count)

            # self.logger.info('{:d}_auxnet_val_age_acc ={:.4f}'.format(i, age_val_acc[i]))
            self.tensorboard_logger.scalar_summary("val_agedb_30_accuracy", age_val_acc[-1]*100, self.run_count)

                # if i == num_segments-1:
            self.logger.info('val_TPR_acc ={:.4f}'.format(tpr_acc))
            self.tensorboard_logger.scalar_summary("Test_TPR", tpr_acc * 100, self.run_count)

        self.run_count += 1
        self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_val_acc[-1]))
        self.logger.info("|===>Validation cfp acc: {:4f}".format(cfp_val_acc[-1]))
        self.logger.info("|===>Validation age acc: {:4f}".format(age_val_acc[-1]))
        self.logger.info("|===>Validation TPR acc: {:4f}".format(tpr_acc))
        return tpr_acc

    def face_test(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        iters = len(self.test_loader)
        # self.update_lr(epoch)

        top1_error = []
        top5_error = []
        top1_loss = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
            self.middle_layer[i].eval()
            self.aux_fc[i].eval()

            top1_error.append(utils.AverageMeter())
            top5_error.append(utils.AverageMeter())
            top1_loss.append(utils.AverageMeter())

        start_time = time.time()
        end_time = start_time

        # use prefetch_generator and tqdm for iterating through data
        pbar = enumerate(BackgroundGenerator(self.test_loader))

        # for i, (images, labels) in enumerate(self.train_loader):
        for i, (images, labels) in pbar:
            start_time = time.time()
            data_time = start_time - end_time

            # if self.settings.n_gpus == 1:
            images = images.cuda()
            labels = labels.cuda()

            # forward
            with torch.no_grad():
                outputs, losses = self.face_auxnet_forward(images, labels)

            # backward
            # for j in range(len(self.fc_optimizer)):
            #     self.face_auxnet_backward_for_loss_i(losses[j], j)

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

            if i % self.settings.print_frequency == 0:
                utils.print_result(epoch, self.settings.n_epochs, i + 1,
                                   iters, self.lr, data_time, iter_time,
                                   single_error,
                                   single_loss,
                                   mode="Test",
                                   logger=self.logger)

            # break

        top1_error_list, top1_loss_list, top5_error_list = self._convert_results(
            top1_error=top1_error, top1_loss=top1_loss, top5_error=top5_error)
        if self.logger is not None:
            for i in range(num_segments):
                # self.tensorboard_logger.scalar_summary(
                #     "auxnet_train_top1_error_{:d}".format(i), top1_error[i].avg,
                #     self.run_count)
                # self.tensorboard_logger.scalar_summary(
                #     "auxnet_train_top5_error_{:d}".format(i), top5_error[i].avg,
                #     self.run_count)
                # self.tensorboard_logger.scalar_summary(
                #     "auxnet_train_loss_{:d}".format(i), top1_loss[i].avg, self.run_count)
                if i == num_segments-1:
                    self.tensorboard_logger.scalar_summary("Test_Top1_Accuracy", (100-top1_error[i].avg)/100.0, self.run_count)
                    self.tensorboard_logger.scalar_summary("Test_Top5_Accuracy", (100-top5_error[i].avg)/100.0, self.run_count)
                    self.tensorboard_logger.scalar_summary("Test_Loss", top1_loss[i].avg, self.run_count)


            # self.tensorboard_logger.scalar_summary("lr", self.lr, self.run_count)

        for i in range(num_segments):
            if i == num_segments - 1:
                self.logger.info("|===>Testing Error_{:d}: top1_error={:4f}/top5_error={:4f}, Loss: {:4f}".format(i,
                                                                                                               top1_error[
                                                                                                                   i].avg,
                                                                                                               top5_error[
                                                                                                                   i].avg,
                                                                                                               top1_loss[
                                                                                                                   i].avg))

        return top1_error_list, top1_loss_list, top5_error_list


    def face_train_before_val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        lfw_val_acc = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            self.segments[i].eval()
            self.middle_layer[i].eval()
            # lfw_val_acc.append(utils.AverageMeter())

        with torch.no_grad():
            for i in range(len(self.segments)):
                # self.logger.info("i={:d}".format(i))
                lfw_accuracy = self.val_evaluate(i,
                                                 self.segments,
                                                 self.middle_layer,
                                                 self.val_loader[-1]['lfw'],
                                                 self.val_loader[-1]['lfw_issame'],
                                                 emb_size=self.channel_array[i])
                # self.logger.info("lfw_accuracy={:3f}".format(lfw_accuracy))
                lfw_val_acc.append(lfw_accuracy)

        if self.logger is not None:
            for i in range(num_segments):
                self.logger.info('{:d}_auxnet_val_lfw_acc ={:.4f}'.format(i, lfw_val_acc[i]))
                # self.tensorboard_logger.scalar_summary("auxnet_val_lfw_acc_{:d}".format(i), lfw_val_acc[i], self.run_count)

        # self.run_count += 1
        self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_val_acc[-1]))
        return lfw_val_acc
