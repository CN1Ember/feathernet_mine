import datetime
import math
import os
import time

import torch
import torch.nn as nn

import dcp.utils as utils
from dcp.mask_conv import MaskConv2d, MaskLinear
from dcp.utils.others import concat_gpu_data
from dcp.utils.write_log import write_log


class LayerChannelSelection(object):
    """
    Discrimination-aware channel selection
    """

    def __init__(self, trainer, train_loader, val_loader, settings, checkpoint, logger, tensorboard_logger):

        self.segment_wise_trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.settings = settings
        self.checkpoint = checkpoint

        self.logger = logger
        self.tensorboard_logger = tensorboard_logger

        self.feature_cache_original_input = {}
        self.feature_cache_original_output = {}
        self.feature_cache_pruned_input = {}
        self.feature_cache_pruned_output = {}

        self.criterion_mse = nn.MSELoss().cuda()
        self.criterion_softmax = nn.CrossEntropyLoss().cuda()

        self.logger_counter = 0

        self.record_time = utils.AverageMeter()
        self.record_selection_mse_loss = utils.AverageMeter()
        self.record_selection_softmax_loss = utils.AverageMeter()
        self.record_selection_loss = utils.AverageMeter()
        self.record_sub_problem_softmax_loss = utils.AverageMeter()
        self.record_sub_problem_mse_loss = utils.AverageMeter()
        self.record_sub_problem_loss = utils.AverageMeter()
        self.record_sub_problem_top1_error = utils.AverageMeter()
        self.record_sub_problem_top5_error = utils.AverageMeter()

    def split_segment_into_three_parts(self, original_segment, pruned_segment, block_count):
        """
        Split the segment into three parts:
            segment_before_pruned_module, pruned_module, segment_after_pruned_module.
        In this way, we can store the input of the pruned module.
        """

        original_segment_list = utils.model2list(original_segment)
        pruned_segment_list = utils.model2list(pruned_segment)

        original_segment_before_pruned_module = []
        pruned_segment_before_pruned_module = []
        pruned_segment_after_pruned_module = []
        for i in range(len(pruned_segment)):
            if i < block_count:
                original_segment_before_pruned_module.append(original_segment_list[i])
                pruned_segment_before_pruned_module.append(pruned_segment_list[i])
            if i > block_count:
                pruned_segment_after_pruned_module.append(pruned_segment_list[i])
        self.original_segment_before_pruned_module = nn.Sequential(*original_segment_before_pruned_module)
        self.pruned_segment_before_pruned_module = nn.Sequential(*pruned_segment_before_pruned_module)
        self.pruned_segment_after_pruned_module = nn.Sequential(*pruned_segment_after_pruned_module)

    @staticmethod
    def _replace_layer(net, layer, layer_index):
        assert isinstance(net, nn.Sequential), "only support nn.Sequential"
        new_net = None

        count = 0
        for origin_layer in net:
            if count == layer_index:
                if new_net is None:
                    new_net = nn.Sequential(layer)
                else:
                    new_net.add_module(str(len(new_net)), layer)
            else:
                if new_net is None:
                    new_net = nn.Sequential(origin_layer)
                else:
                    new_net.add_module(str(len(new_net)), origin_layer)
            count += 1
        return new_net

    def replace_layer_with_mask_conv(self, pruned_segment, module, layer_name, block_count):
        """
        Replace the pruned layer with mask convolution
        """
        if layer_name == "conv1":
            layer = module.conv1
        elif layer_name == "conv2":
            layer = module.conv2
        elif layer_name == "conv3":
            layer = module.conv3

        elif layer_name == "fc":
            if self.settings.net_type in ["LResnetxE-IR"]:
                layer = module.fc
            elif self.settings.net_type in ["mobilefacenet_v1"]:
                layer = module.linear
            else:
                assert False, "unsupport layer: {}, except LResnetxE-IR and mobilefacenet_v1".format(layer_name)

        elif layer_name == "conv":
            assert self.settings.net_type in ["vgg"], "only support vgg"
            layer = module
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, (MaskConv2d, MaskLinear)):
            if not isinstance(layer, nn.Linear):
                temp_conv = MaskConv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    bias=(layer.bias is not None))
                temp_conv.weight.data.copy_(layer.weight.data)

                if layer.bias is not None:
                    temp_conv.bias.data.copy_(layer.bias.data)
                temp_conv.pruned_weight.data.fill_(0)
                temp_conv.d.fill_(0)

            elif isinstance(layer, nn.Linear):
                temp_conv = MaskLinear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=(layer.bias is not None))
                temp_conv.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    temp_conv.bias.data.copy_(layer.bias.data)
                temp_conv.pruned_weight.data.fill_(0)
                temp_conv.d.fill_(0)

            else:
                assert False, "unsupport replace layer: {}".format(layer_name)

            if layer_name == "conv1":
                module.conv1 = temp_conv
            elif layer_name == "conv2":
                module.conv2 = temp_conv
            elif layer_name == "conv3":
                module.conv3 = temp_conv

            elif layer_name == "fc":
                if self.settings.net_type in ["LResnetxE-IR"]:
                    module.fc = temp_conv
                elif self.settings.net_type in ["mobilefacenet_v1"]:
                    module.linear = temp_conv

            elif layer_name == "conv":     # VGG net
                pruned_segment = self._replace_layer(net=pruned_segment,
                                                     layer=temp_conv,
                                                     layer_index=block_count)
            layer = temp_conv

        return pruned_segment, layer

    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_original_input[gpu_id] = input[0]
        self.feature_cache_original_output[gpu_id] = output

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_pruned_input[gpu_id] = input[0]
        self.feature_cache_pruned_output[gpu_id] = output

    def register_layer_hook(self, original_segment, pruned_segment, module, layer_name, block_count):
        """
        In order to get the input and the output of the intermediate layer, we register
        the forward hook for the pruned layer
        """

        if layer_name == "conv1":
            self.hook_origin = original_segment[block_count].conv1.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = module.conv1.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv2":
            self.hook_origin = original_segment[block_count].conv2.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = module.conv2.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv3":
            self.hook_origin = original_segment[block_count].conv3.register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = module.conv3.register_forward_hook(self._hook_pruned_feature)

        elif layer_name == "fc":
            if self.settings.net_type in ["LResnetxE-IR"]:
                self.hook_origin = original_segment[block_count].fc.register_forward_hook(self._hook_origin_feature)
                self.hook_pruned = module.fc.register_forward_hook(self._hook_pruned_feature)
            elif self.settings.net_type in ["mobilefacenet_v1"]:
                self.hook_origin = original_segment[block_count].linear.register_forward_hook(self._hook_origin_feature)
                self.hook_pruned = module.linear.register_forward_hook(self._hook_pruned_feature)

        elif layer_name == "conv":
            self.hook_origin = original_segment[block_count].register_forward_hook(self._hook_origin_feature)
            self.hook_pruned = pruned_segment[block_count].register_forward_hook(self._hook_pruned_feature)

    def prepare_feature(self):
        """
        Prepare input feature for the pruned module
        """

        self.logger.info("|===>Preparing feature")
        original_module_input_features = []
        pruned_module_input_features = []
        store_labels = []
        img_count = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.cuda()
                labels = labels.cuda()
                original_module_input_feature = self.original_segment_before_pruned_module(images)
                pruned_module_input_feature = self.pruned_segment_before_pruned_module(images)
                original_module_input_features.append(original_module_input_feature.detach().cpu())
                pruned_module_input_features.append(pruned_module_input_feature.detach().cpu())
                store_labels.append(labels.detach().cpu())

                img_count += images.size(0)
                if self.settings.max_samples != -1 and img_count >= self.settings.max_samples:
                    break

        self.original_module_input_features = torch.cat(original_module_input_features)
        self.pruned_module_input_features = torch.cat(pruned_module_input_features)
        self.store_labels = torch.cat(store_labels)

        # self.logger.info('input features size')
        # self.logger.info('Shape: {}'.format(self.original_module_input_features.shape))
        # self.logger.info('Element_size: {}'.format(self.original_module_input_features.element_size()))
        # self.logger.info('Num of element: {}'.format(self.original_module_input_features.nelement()))
        # self.logger.info('Memory: {}MB'.format(self.original_module_input_features.element_size() * \
        #                  self.original_module_input_features.nelement() / 1024 / 1024))

        self.num_samples = self.original_module_input_features.size(0)
        self.num_batch = int(self.num_samples / self.settings.batch_size)
        self.index_list = torch.arange(self.num_samples)
        self.logger.info("|===>Preparing feature completed!")

    def segment_parallelism(self, original_segment, pruned_segment):
        """
        Parallel setting for segment
        """

        self.original_segment_parallel = utils.data_parallel(original_segment, self.settings.n_gpus)
        self.pruned_segment_parallel = utils.data_parallel(pruned_segment, self.settings.n_gpus)

    def reset_average_meter(self):
        self.record_time.reset()
        self.record_selection_mse_loss.reset()
        self.record_selection_softmax_loss.reset()
        self.record_selection_loss.reset()
        self.record_sub_problem_softmax_loss.reset()
        self.record_sub_problem_mse_loss.reset()
        self.record_sub_problem_loss.reset()
        self.record_sub_problem_top1_error.reset()
        self.record_sub_problem_top5_error.reset()

    def prepare_channel_selection(self, original_segment, pruned_segment, module, aux_fc, layer_name, block_count):
        """
        Prepare for channel selection
        1. Split the segment into three parts.
        2. Replace the pruned layer with mask convolution.
        3. Store the input feature map of the pruned layer in advance to accelerate channel selection.
        """

        self.split_segment_into_three_parts(original_segment, pruned_segment, block_count)
        pruned_segment, layer = self.replace_layer_with_mask_conv(pruned_segment, module, layer_name, block_count)
        self.register_layer_hook(original_segment, pruned_segment, module, layer_name, block_count)

        # parallel setting
        self.segment_parallelism(original_segment, pruned_segment)

        # turn gradient off
        # avoid computing the gradient
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = False
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = False

        # freeze the Batch Normalization
        self.original_segment_parallel.eval()
        self.pruned_segment_parallel.eval()

        # store the input feature map of the pruned layer in advance
        if self.settings.prepare_features:
            self.prepare_feature()
        else:
            self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        layer.pruned_weight.requires_grad = True
        aux_fc.cuda()

        self.logger_counter = 0
        self.reset_average_meter()
        return pruned_segment, layer

    def get_prepare_batch_data(self, batch_index):
        """
        Get mini-batch data
        :param batch_index: the index of the batch
        """

        length = self.num_samples - batch_index * self.settings.batch_size \
            if batch_index * self.settings.batch_size + self.settings.batch_size > self.num_samples \
            else self.settings.batch_size
        index = torch.narrow(self.index_list, 0, batch_index * self.settings.batch_size, length)
        cuda0 = torch.device('cuda:0')
        # index = torch.narrow(self.index_list, 0, batch_index * self.settings.batch_size, length).cuda()
        original_module_input_feature = torch.index_select(self.original_module_input_features, 0, index).to(cuda0)
        pruned_module_input_feature = torch.index_select(self.pruned_module_input_features, 0, index).to(cuda0)
        labels = torch.index_select(self.store_labels, 0, index).to(cuda0)
        return original_module_input_feature, pruned_module_input_feature, labels

    def get_batch_data(self, train_dataloader_iter):
        images, labels = train_dataloader_iter.next()
        images = images.cuda()
        labels = labels.cuda()
        original_module_input_feature = self.original_segment_before_pruned_module(images)
        pruned_module_input_feature = self.pruned_segment_before_pruned_module(images)
        return original_module_input_feature, pruned_module_input_feature, labels

    def get_correlation_loss(self, pruned_output, origin_output):
        # get the feature and semantic correlation loss
        pruned_output_size = pruned_output.size()
        if len(pruned_output_size) != 4:
            assert 'get_correlation_loss: pruned_output dimension is not 4 !'
        bath_size = pruned_output_size[0]
        N = pruned_output_size[2] * pruned_output_size[3]
        M = pruned_output_size[1]
        pruned_output_reshape = pruned_output.view(pruned_output_size[0], pruned_output_size[1], -1)  # n*c*h*w -> n*c*(h*w)

        origin_output_size = origin_output.size()
        if len(origin_output_size) != 4:
            assert 'get_correlation_loss: origin_output dimension is not 4 !'
        origin_output_reshape = origin_output.view(origin_output_size[0], origin_output_size[1], -1)

        G_f = torch.bmm(origin_output_reshape, origin_output_reshape.permute(0, 2, 1))
        G_f_prun = torch.bmm(pruned_output_reshape, pruned_output_reshape.permute(0, 2, 1))
        G_f_minus = G_f - G_f_prun

        G_s = torch.bmm(origin_output_reshape.permute(0, 2, 1), origin_output_reshape)
        G_s_prun = torch.bmm(pruned_output_reshape.permute(0, 2, 1), pruned_output_reshape)
        G_s_minus = G_s - G_s_prun

        # correlation_loss = (torch.sum(G_f_minus.mul(G_f_minus).view(bath_size, -1), 1)
        #                     + torch.sum(G_s_minus.mul(G_s_minus).view(bath_size, -1), 1)) / (4 * N * N * M * M)

        correlation_loss = (torch.sum(G_f_minus.mul(G_f_minus).view(bath_size, -1), 1)
                            + torch.sum(G_s_minus.mul(G_s_minus).view(bath_size, -1), 1)) / (N * N * M * M)

        correlation_loss = torch.sum(correlation_loss) / bath_size
        return correlation_loss

    def compute_loss_error(self, original_segment, pruned_segment, block_count, aux_fc,
                           original_module_input_feature, pruned_module_input_feature, labels):
        """
        Compute the total loss, softmax_loss, mse_loss, top1_error and top5_error
        """

        # forward propagation
        original_segment[block_count](original_module_input_feature)
        pruned_output = pruned_segment[block_count](pruned_module_input_feature)
        fc_output = aux_fc(self.pruned_segment_after_pruned_module(pruned_output))

        # get the output feature of the pruned layer
        origin_output = concat_gpu_data(self.feature_cache_original_output)
        pruned_output = concat_gpu_data(self.feature_cache_pruned_output)

        # compute loss
        softmax_loss = self.criterion_softmax(fc_output, labels)
        mse_loss = self.criterion_mse(pruned_output, origin_output.detach())

        # original codes
        loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight

        # xz codes
        # correlation_loss = self.get_correlation_loss(pruned_output, origin_output.detach())
        # loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight + \
        #        correlation_loss * self.settings.correlation_weight

        top1_error, _, top5_error = utils.compute_singlecrop_error(
            outputs=fc_output, labels=labels,
            loss=softmax_loss, top5_flag=True)
        return loss, softmax_loss, mse_loss, top1_error, top5_error

    def find_maximum_grad_fnorm(self, grad_fnorm, layer):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """

        # grad_fnorm.data.mul_(1 - layer.d)
        # _, max_index = torch.topk(grad_fnorm, 1)
        # layer.d[max_index[0]] = 1
        # # warm-started from the pre-trained model
        # layer.pruned_weight.data[:, max_index[0], :, :] = layer.weight[:, max_index[0], :, :].data.clone()
        # self.logger.info(grad_fnorm)
        # self.logger.info(grad_fnorm.nonzero())
        # assert False
        while True:
            _, max_index = torch.topk(grad_fnorm, 1)
            if layer.d[max_index[0]] == 0:
                layer.d[max_index[0]] = 1
                layer.pruned_weight.data[:, max_index[0], :, :] = layer.weight[:, max_index[0], :, :].data.clone()
                break
            else:
                grad_fnorm[max_index[0]] = -1

    def find_most_violated(self, original_segment, pruned_segment, aux_fc, layer, block_count):
        """
        Find the channel with maximum gradient frobenius norm.
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        layer.pruned_weight.grad = None
        train_dataloader_iter = iter(self.train_loader)

        for j in range(self.num_batch):
            # get data
            if self.settings.prepare_features:
                original_module_input_feature, pruned_module_input_feature, labels = self.get_prepare_batch_data(j)
            else:
                original_module_input_feature, pruned_module_input_feature, labels = \
                    self.get_batch_data(train_dataloader_iter)

            loss, softmax_loss, mse_loss, top1_error, top5_error = \
                self.compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                        original_module_input_feature, pruned_module_input_feature, labels)

            loss.backward()

            self.record_selection_loss.update(loss.item(), labels.size(0))
            self.record_selection_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_selection_softmax_loss.update(softmax_loss.item(), labels.size(0))

        cum_grad = layer.pruned_weight.grad.data.clone()
        layer.pruned_weight.grad = None

        # calculate F norm of gradient
        grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)   # input-channel dimension

        # find grad_fnorm with maximum absolute gradient
        self.find_maximum_grad_fnorm(grad_fnorm, layer)

    def set_layer_wise_optimizer(self, layer):
        params_list = []
        params_list.append({"params": layer.pruned_weight, "lr": self.settings.layer_wise_lr})
        if layer.bias is not None:
            layer.bias.requires_grad = True
            params_list.append({"params": layer.bias, "lr": self.settings.layer_wise_lr})
        optimizer = torch.optim.SGD(params=params_list,
                                    weight_decay=self.settings.weight_decay,
                                    momentum=self.settings.momentum,
                                    nesterov=True)
        return optimizer

    def solve_sub_problem(self, original_segment, pruned_segment, aux_fc, layer, block_count):
        """
        We optimize W w.r.t. the selected channels by minimizing the problem (8)
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        optimizer = self.set_layer_wise_optimizer(layer)
        train_dataloader_iter = iter(self.train_loader)

        for j in range(self.num_batch):
            # get data
            if self.settings.prepare_features:
                original_module_input_feature, pruned_module_input_feature, labels = self.get_prepare_batch_data(j)
            else:
                original_module_input_feature, pruned_module_input_feature, labels = \
                    self.get_batch_data(train_dataloader_iter)

            loss, softmax_loss, mse_loss, top1_error, top5_error = \
                self.compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                        original_module_input_feature, pruned_module_input_feature, labels)

            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # we only optimize W with respect to the selected channel
            layer.pruned_weight.grad.data.mul_(layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
            optimizer.step()

            # update record info
            self.record_sub_problem_softmax_loss.update(softmax_loss.item(), labels.size(0))
            self.record_sub_problem_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_sub_problem_loss.update(loss.item(), labels.size(0))

            self.record_sub_problem_top1_error.update(top1_error, labels.size(0))
            self.record_sub_problem_top5_error.update(top5_error, labels.size(0))

        layer.pruned_weight.grad = None
        if layer.bias is not None:
            layer.bias.grad = None
        if layer.bias is not None:
            layer.bias.requires_grad = False

    def write_log(self, layer, block_count, layer_name):
        self.write_tensorboard_log(block_count, layer_name)
        self.write_log2file(layer, block_count, layer_name)

    def write_tensorboard_log(self, block_count, layer_name):
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_LossAll".format(block_count, layer_name),
            value=self.record_selection_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_MSELoss".format(block_count, layer_name),
            value=self.record_selection_mse_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Selection-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
            value=self.record_selection_softmax_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
            value=self.record_sub_problem_softmax_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Loss".format(block_count, layer_name),
            value=self.record_sub_problem_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_MSELoss".format(block_count, layer_name),
            value=self.record_sub_problem_mse_loss.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Top1Error".format(block_count, layer_name),
            value=self.record_sub_problem_top1_error.avg,
            step=self.logger_counter)
        self.tensorboard_logger.scalar_summary(
            tag="Sub-problem-block-{}_{}_Top5Error".format(block_count, layer_name),
            value=self.record_sub_problem_top5_error.avg,
            step=self.logger_counter)

        self.logger_counter += 1

    def write_log2file(self, layer, block_count, layer_name):
        write_log(
            dir_name=os.path.join(self.settings.save_path, "log"),
            file_name="log_block-{:0>2d}_{}.txt".format(block_count, layer_name),
            log_str="{:d}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t\n".format(
                int(layer.d.sum()),
                self.record_selection_loss.avg,
                self.record_selection_mse_loss.avg,
                self.record_selection_softmax_loss.avg,
                self.record_sub_problem_loss.avg,
                self.record_sub_problem_mse_loss.avg,
                self.record_sub_problem_softmax_loss.avg,
                self.record_sub_problem_top1_error.avg,
                self.record_sub_problem_top5_error.avg))
        log_str = "Block-{:0>2d}-{}\t#channels: [{:0>4d}|{:0>4d}]\t".format(
            block_count, layer_name,
            int(layer.d.sum()), layer.d.size(0))
        log_str += "[selection]loss: {:4f}\tmseloss: {:4f}\tsoftmaxloss: {:4f}\t".format(
            self.record_selection_loss.avg,
            self.record_selection_mse_loss.avg,
            self.record_selection_softmax_loss.avg)
        log_str += "[subproblem]loss: {:4f}\tmseloss: {:4f}\tsoftmaxloss: {:4f}\t".format(
            self.record_sub_problem_loss.avg,
            self.record_sub_problem_mse_loss.avg,
            self.record_sub_problem_softmax_loss.avg)
        log_str += "top1error: {:4f}\ttop5error: {:4f}".format(
            self.record_sub_problem_top1_error.avg,
            self.record_sub_problem_top5_error.avg)
        self.logger.info(log_str)

    def remove_layer_hook(self):
        self.hook_origin.remove()
        self.hook_pruned.remove()
        self.logger.info("|===>remove hook")

    # dcp
    def channel_selection_for_one_layer(self, original_segment, pruned_segment, aux_fc, module,
                                        block_count, layer_name="conv2"):
        """
        Conduct channel selection for one layer in a module
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param module: the module need to be pruned
        :param block_count: current block no.
        :param layer_name: the name of layer need to be pruned
        """

        # layer-wise channel selection
        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        pruned_segment, layer = self.prepare_channel_selection(original_segment, pruned_segment, module,
                                                               aux_fc, layer_name, block_count)

        for channel in range(layer.in_channels):
            if layer.d.eq(0).sum() <= math.floor(layer.in_channels * self.settings.pruning_rate):
                break

            time_start = time.time()
            # find the channel with the maximum gradient norm
            self.find_most_violated(original_segment, pruned_segment, aux_fc, layer, block_count)

            # solve problem (8) w.r.t. the selected channels
            self.solve_sub_problem(original_segment, pruned_segment, aux_fc, layer, block_count)
            time_interval = time.time() - time_start

            self.write_log(layer, block_count, layer_name)
            self.record_time.update(time_interval)

        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=self.record_time.sum)),
            str(datetime.timedelta(seconds=self.record_time.avg)))
        self.logger.info(log_str)

        # turn requires_grad on
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = True
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = True
        self.remove_layer_hook()
        return pruned_segment


    ########################################### insightface_dcp ###########################################
    def find_maximum_grad_fnorm_linear(self, grad_fnorm, layer):
        """
        Find the channel index with maximum gradient frobenius norm,
        and initialize the pruned_weight w.r.t. the selected channel.
        """

        # grad_fnorm.data.mul_(1 - layer.d)
        # _, max_index = torch.topk(grad_fnorm, 1)
        # layer.d[max_index[0]] = 1
        # # warm-started from the pre-trained model
        # layer.pruned_weight.data[:, max_index[0], :, :] = layer.weight[:, max_index[0], :, :].data.clone()
        # self.logger.info(grad_fnorm)
        # self.logger.info(grad_fnorm.nonzero())
        # assert False
        while True:
            _, max_index = torch.topk(grad_fnorm, 1)
            if layer.d[max_index[0]] == 0:
                layer.d[max_index[0]] = 1
                layer.pruned_weight.data[:, max_index[0]] = layer.weight[:, max_index[0]].data.clone()
                break
            else:
                grad_fnorm[max_index[0]] = -1

    def face_compute_loss_error(self, original_segment, pruned_segment, block_count, middle_layer, aux_fc,
                           original_module_input_feature, pruned_module_input_feature, labels):
        """
        Compute the total loss, softmax_loss, mse_loss, top1_error and top5_error
        """

        # forward propagation
        original_segment[block_count](original_module_input_feature)
        pruned_output = pruned_segment[block_count](pruned_module_input_feature)
        middle_layer_output = middle_layer(self.pruned_segment_after_pruned_module(pruned_output))
        fc_output = aux_fc(middle_layer_output, labels)

        # get the output feature of the pruned layer
        origin_output = concat_gpu_data(self.feature_cache_original_output)
        pruned_output = concat_gpu_data(self.feature_cache_pruned_output)

        # compute loss
        softmax_loss = self.criterion_softmax(fc_output, labels)
        mse_loss = self.criterion_mse(pruned_output, origin_output.detach())

        # original codes
        loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight

        # xz codes
        # correlation_loss = self.get_correlation_loss(pruned_output, origin_output.detach())
        # loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight + \
        #        correlation_loss * self.settings.correlation_weight

        top1_error, _, top5_error = utils.compute_singlecrop_error(
            outputs=fc_output, labels=labels,
            loss=softmax_loss, top5_flag=True)
        return loss, softmax_loss, mse_loss, top1_error, top5_error

    def face_fc_compute_loss_error(self, original_segment, pruned_segment, block_count, aux_fc,
                                   original_module_input_feature, pruned_module_input_feature, labels):
        """
        Compute the total loss, softmax_loss, mse_loss, top1_error and top5_error
        """

        # forward propagation
        original_segment[block_count](original_module_input_feature)
        pruned_output = pruned_segment[block_count](pruned_module_input_feature)
        # middle_layer_output = middle_layer(self.pruned_segment_after_pruned_module(pruned_output))
        fc_output = aux_fc(self.pruned_segment_after_pruned_module(pruned_output), labels)

        # get the output feature of the pruned layer
        origin_output = concat_gpu_data(self.feature_cache_original_output)
        pruned_output = concat_gpu_data(self.feature_cache_pruned_output)

        # compute loss
        softmax_loss = self.criterion_softmax(fc_output, labels)
        mse_loss = self.criterion_mse(pruned_output, origin_output.detach())

        # original codes
        loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight

        # xz codes
        # correlation_loss = self.get_correlation_loss(pruned_output, origin_output.detach())
        # loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight + \
        #        correlation_loss * self.settings.correlation_weight

        top1_error, _, top5_error = utils.compute_singlecrop_error(
            outputs=fc_output, labels=labels,
            loss=softmax_loss, top5_flag=True)
        return loss, softmax_loss, mse_loss, top1_error, top5_error


    def face_prepare_channel_selection(self, original_segment, pruned_segment, module, middle_layer, aux_fc,
                                       layer_name, block_count):
        """
        Prepare for channel selection
        1. Split the segment into three parts.
        2. Replace the pruned layer with mask convolution.
        3. Store the input feature map of the pruned layer in advance to accelerate channel selection.
        """

        self.split_segment_into_three_parts(original_segment, pruned_segment, block_count)
        pruned_segment, layer = self.replace_layer_with_mask_conv(pruned_segment, module, layer_name, block_count)
        self.register_layer_hook(original_segment, pruned_segment, module, layer_name, block_count)

        # parallel setting
        self.segment_parallelism(original_segment, pruned_segment)

        # turn gradient off
        # avoid computing the gradient
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = False
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = False

        # freeze the Batch Normalization
        self.original_segment_parallel.eval()
        self.pruned_segment_parallel.eval()

        middle_layer.eval()
        aux_fc.eval()

        # store the input feature map of the pruned layer in advance
        if self.settings.prepare_features:
            self.prepare_feature()
        else:
            self.num_batch = len(self.train_loader)

        # turn on the gradient with respect to the pruned layer
        layer.pruned_weight.requires_grad = True
        middle_layer.cuda()
        aux_fc.cuda()

        self.logger_counter = 0
        self.reset_average_meter()
        return pruned_segment, layer

    def face_find_most_violated(self, original_segment, pruned_segment, middle_layer, aux_fc, layer, block_count, layer_name):
        """
        Find the channel with maximum gradient frobenius norm.
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        layer.pruned_weight.grad = None
        train_dataloader_iter = iter(self.train_loader)

        for j in range(self.num_batch):
            # get data
            if self.settings.prepare_features:
                original_module_input_feature, pruned_module_input_feature, labels = self.get_prepare_batch_data(j)
            else:
                original_module_input_feature, pruned_module_input_feature, labels = \
                    self.get_batch_data(train_dataloader_iter)

            if layer_name == "fc":
                loss, softmax_loss, mse_loss, top1_error, top5_error = \
                    self.face_fc_compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                                 original_module_input_feature, pruned_module_input_feature, labels)
            else:
                loss, softmax_loss, mse_loss, top1_error, top5_error = \
                    self.face_compute_loss_error(original_segment, pruned_segment, block_count, middle_layer, aux_fc,
                                                 original_module_input_feature, pruned_module_input_feature, labels)

            loss.backward()

            self.record_selection_loss.update(loss.item(), labels.size(0))
            self.record_selection_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_selection_softmax_loss.update(softmax_loss.item(), labels.size(0))

        cum_grad = layer.pruned_weight.grad.data.clone()
        layer.pruned_weight.grad = None

        if layer_name != "fc":
            # calculate F norm of gradient
            grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)  # input-channel dimension
            # find grad_fnorm with maximum absolute gradient
            self.find_maximum_grad_fnorm(grad_fnorm, layer)

        else:
            # self.logger.info("find_maximum_grad_fnorm_linear !!!")
            grad_fnorm = cum_grad.mul(cum_grad).sqrt().sum(0)  # input-channel dimension
            self.find_maximum_grad_fnorm_linear(grad_fnorm, layer)

    def face_solve_sub_problem(self, original_segment, pruned_segment, middle_layer, aux_fc, layer, block_count, layer_name):
        """
        We optimize W w.r.t. the selected channels by minimizing the problem (8)
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param layer: the layer to be pruned
        :param block_count: current block no.
        """

        optimizer = self.set_layer_wise_optimizer(layer)
        train_dataloader_iter = iter(self.train_loader)

        for j in range(self.num_batch):
            # get data
            if self.settings.prepare_features:
                original_module_input_feature, pruned_module_input_feature, labels = self.get_prepare_batch_data(j)
            else:
                original_module_input_feature, pruned_module_input_feature, labels = \
                    self.get_batch_data(train_dataloader_iter)

            # loss, softmax_loss, mse_loss, top1_error, top5_error = \
            #     self.face_compute_loss_error(original_segment, pruned_segment, block_count, middle_layer, aux_fc,
            #                                  original_module_input_feature, pruned_module_input_feature, labels)

            if layer_name == "fc":
                loss, softmax_loss, mse_loss, top1_error, top5_error = \
                    self.face_fc_compute_loss_error(original_segment, pruned_segment, block_count, aux_fc,
                                                 original_module_input_feature, pruned_module_input_feature, labels)
            else:
                loss, softmax_loss, mse_loss, top1_error, top5_error = \
                    self.face_compute_loss_error(original_segment, pruned_segment, block_count, middle_layer, aux_fc,
                                                 original_module_input_feature, pruned_module_input_feature, labels)


            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # we only optimize W with respect to the selected channel

            if layer_name != "fc":
                layer.pruned_weight.grad.data.mul_(layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
            else:
                layer.pruned_weight.grad.data.mul_(layer.d.unsqueeze(0).expand_as(layer.pruned_weight))

            optimizer.step()

            # update record info
            self.record_sub_problem_softmax_loss.update(softmax_loss.item(), labels.size(0))
            self.record_sub_problem_mse_loss.update(mse_loss.item(), labels.size(0))
            self.record_sub_problem_loss.update(loss.item(), labels.size(0))

            self.record_sub_problem_top1_error.update(top1_error, labels.size(0))
            self.record_sub_problem_top5_error.update(top5_error, labels.size(0))

        layer.pruned_weight.grad = None
        if layer.bias is not None:
            layer.bias.grad = None
        if layer.bias is not None:
            layer.bias.requires_grad = False

    def face_channel_selection_for_one_layer(self, original_segment, pruned_segment, middle_layer, aux_fc, module,
                                             block_count, layer_name="conv2"):
        """
        Conduct channel selection for one layer in a module
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param middle_layer: net_tail
        :param aux_fc: auxiliary fully-connected layer
        :param module: the module need to be pruned
        :param block_count: current block no.
        :param layer_name: the name of layer need to be pruned
        """

        # layer-wise channel selection
        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        pruned_segment, layer = self.face_prepare_channel_selection(original_segment, pruned_segment, module,
                                                                    middle_layer, aux_fc, layer_name, block_count)

        if layer_name in ["conv2", "conv3"]:
            channel_num= layer.in_channels
        elif layer_name in ["fc"]:
            channel_num = layer.in_features
        else:
            assert False, "unsupported net_type: {}".format(layer_name)

        for channel in range(channel_num):
            # self.logger.info("|===>layer.d.eq(0).sum() = {}, expected_channel_num={}"
            #                  .format(layer.d.eq(0).sum(), math.floor(layer.in_channels * self.settings.pruning_rate)))
            if layer.d.eq(0).sum() <= math.floor(channel_num * self.settings.pruning_rate):
                break

            time_start = time.time()
            # find the channel with the maximum gradient norm
            self.face_find_most_violated(original_segment, pruned_segment, middle_layer, aux_fc, layer, block_count, layer_name)

            # solve problem (8) w.r.t. the selected channels
            self.face_solve_sub_problem(original_segment, pruned_segment, middle_layer, aux_fc, layer, block_count, layer_name)
            time_interval = time.time() - time_start

            self.write_log(layer, block_count, layer_name)
            self.record_time.update(time_interval)

        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=self.record_time.sum)),
            str(datetime.timedelta(seconds=self.record_time.avg)))
        self.logger.info(log_str)

        # turn requires_grad on
        for params in self.original_segment_parallel.parameters():
            params.requires_grad = True
        for params in self.pruned_segment_parallel.parameters():
            params.requires_grad = True
        self.remove_layer_hook()
        return pruned_segment

