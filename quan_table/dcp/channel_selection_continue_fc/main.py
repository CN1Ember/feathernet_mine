import argparse
import datetime
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from channel_selection import LayerChannelSelection
from dcp_checkpoint import DCPCheckPoint
from option import Option
from trainer import SegmentWiseTrainer

import dcp.utils as utils
from dcp.dataloader import *
from dcp.mask_conv import MaskConv2d, MaskLinear
from dcp.model_builder import get_model
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.logger import get_logger
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_settings

from dcp.models.insightface_resnet import IRBlock
from dcp.middle_layer import Middle_layer
from dcp.arcface import Arcface

from dcp.models.mobilefacenet import Bottleneck_mobilefacenet
from dcp.middle_layer import Middle_layer_mobilefacenetv1

block_num = {'vgg19': 16, 'preresnet56': 27, 'resnet18': 8, 'LResnetxE-IR34':16, "mobilefacenet_v1":15, 'resnet50': 16}


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.original_model = None
        self.pruned_model = None

        self.aux_fc_state = None
        self.middle_layer_state = None
        self.aux_fc_opt_state = None
        self.seg_opt_state = None
        self.current_pivot_index = None

        self.epoch = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu
        self.settings.set_save_path()
        write_settings(self.settings)
        self.logger = get_logger(self.settings.save_path, "dcp")
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)
        self.logger.info("|===>Result will be saved at {}".format(self.settings.save_path))

        self.prepare()

    def prepare(self):
        """
        Preparing experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._cal_pivot()
        self._set_checkpoint()
        self._set_trainier()
        self._set_channel_selection()

    def _set_gpu(self):
        """
        Initialize the seed of random number generator
        """

        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Create train loader and validation loader for channel pruning
        """

        if 'cifar' in self.settings.dataset:
            self.train_loader, self.val_loader = get_cifar_dataloader(self.settings.dataset,
                                                                      self.settings.batch_size,
                                                                      self.settings.n_threads,
                                                                      self.settings.data_path,
                                                                      self.logger)
        elif self.settings.dataset == 'imagenet':
            self.train_loader, self.val_loader = get_imagenet_dataloader(self.settings.dataset,
                                                                         self.settings.batch_size,
                                                                         self.settings.n_threads,
                                                                         self.settings.data_path,
                                                                         self.logger)


        elif self.settings.dataset in ['ms1m_v2', 'iccv_ms1m']:
            self.train_loader, self.val_loader, class_num = get_ms1m_dataloader(self.settings.dataset,
                                                                     self.settings.batch_size,
                                                                     self.settings.n_threads,
                                                                     self.settings.data_path,
                                                                     self.logger)
            self.settings.n_classes = class_num   # class number
            self.logger.info('self.settings.n_classes={:d}'.format(self.settings.n_classes))

    def _set_trainier(self):
        """
        Initialize segment-wise trainer trainer
        """

        # initialize segment-wise trainer
        self.segment_wise_trainer = SegmentWiseTrainer(original_model=self.original_model,
                                                       pruned_model=self.pruned_model,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       tensorboard_logger=self.tensorboard_logger)

        if self.aux_fc_state is not None:
            self.segment_wise_trainer.update_aux_fc(self.middle_layer_state, self.aux_fc_state, self.aux_fc_opt_state,
                                                    self.seg_opt_state)

    def _set_channel_selection(self):
        self.layer_channel_selection = LayerChannelSelection(self.segment_wise_trainer, self.train_loader,
                                                             self.val_loader, self.settings, self.checkpoint,
                                                             self.logger, self.tensorboard_logger)

    def _set_model(self):
        """
        Available model
        cifar:
            preresnet
            vgg
        imagenet:
            resnet
        """

        self.original_model = get_model(self.settings,
                                        self.settings.dataset,
                                        self.settings.net_type,
                                        self.settings.depth,
                                        self.settings.n_classes)
        self.pruned_model = get_model(self.settings,
                                      self.settings.dataset,
                                      self.settings.net_type,
                                      self.settings.depth,
                                      self.settings.n_classes)
        self.logger.info("self.original_model={}\n\nself.pruned_model={}".format(self.original_model, self.pruned_model))

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.original_model is not None and self.pruned_model is not None, "please create model first"

        self.checkpoint = DCPCheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        Load pre-trained model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)
            model_state = check_point_params['model']
            self.middle_layer_state = check_point_params['middle_layer']
            self.aux_fc_state = check_point_params['aux_fc']

            epoch = check_point_params['epoch']
            lfw_accuracy = check_point_params['lfw_accuracy']
            self.logger.info('epoch={}, lfw_acc={}'.format(epoch, lfw_accuracy))

            self.original_model = self.checkpoint.load_state(self.original_model, model_state)
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
            self.logger.info("|===>load restart file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        """
        # To do
        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            original_model_state = check_point_params["original_model"]
            pruned_model_state = check_point_params["pruned_model"]

            self.middle_layer_state = check_point_params["middle_layer"]
            self.aux_fc_state = check_point_params["aux_fc"]

            # path = "/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/" \
            #        "mobilefacenet/log_aux_mobilefacenetv2_baseline_512_arcface_iccv_emore_bs200_e12_lr0.100_" \
            #        "step[3, 6, 9]_auxnet_n2_ftfc_arcface_new_op_20190721/check_point/checkpoint_011_fix.pth"
            # check_param = torch.load(path)
            # last_aux_fc_state = check_param["aux_fc"]
            # self.aux_fc_state[-1] = last_aux_fc_state[-1]
            # self.logger.info("Add last aux_fc_state into model")

            # self.aux_fc_opt_state = check_point_params["aux_fc_opt"]
            # self.seg_opt_state = check_point_params["seg_opt"]
            self.current_pivot_index = check_point_params["current_pivot"]
            self.segment_num = check_point_params["segment_num"]
            self.current_block_count = check_point_params["block_num"]
            self.current_block_count = self.current_pivot_index

            self.logger.info("load resume: self.current_pivot_index={}, self.segment_num={}, self.current_block_count={}"
                             .format(self.current_pivot_index, self.segment_num, self.current_block_count))

            if self.current_pivot_index > self.settings.pivot_set[0]:
                self.replace_layer_mask_conv()
            self.logger.info("resume: after replace, self.pruned_model={}".format(self.pruned_model))
            self.original_model = self.checkpoint.load_state(self.original_model, original_model_state)
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    def _cal_pivot(self):
        """
        Calculate the inserted layer for additional loss
        """

        self.num_segments = self.settings.n_losses + 1
        if self.settings.net_type not in ["mobilefacenet_v1"]:
            num_block_per_segment = (block_num[self.settings.net_type + str(self.settings.depth)] // self.num_segments) + 1
        else:
            num_block_per_segment = (block_num[self.settings.net_type ] // self.num_segments) + 1

        pivot_set = []
        for i in range(self.num_segments - 1):
            pivot_set.append(num_block_per_segment * (i + 1))
        self.settings.pivot_set = pivot_set
        self.logger.info("pivot set: {}".format(pivot_set))

    def replace_layer_mask_conv(self):
        """
        Replace the convolutional layer with mask convolutional layer
        """

        block_count = 0
        if self.settings.net_type in ["preresnet", "resnet"]:
            for module in self.pruned_model.modules():
                if isinstance(module, (PreBasicBlock, BasicBlock, Bottleneck)):
                    block_count += 1
                    layer = module.conv2
                    if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                        temp_conv = MaskConv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None))
                        temp_conv.weight.data.copy_(layer.weight.data)

                        if layer.bias is not None:
                            temp_conv.bias.data.copy_(layer.bias.data)
                        module.conv2 = temp_conv

                    if isinstance(module, Bottleneck):
                        layer = module.conv3
                        if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                            temp_conv = MaskConv2d(
                                in_channels=layer.in_channels,
                                out_channels=layer.out_channels,
                                kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding,
                                bias=(layer.bias is not None))
                            temp_conv.weight.data.copy_(layer.weight.data)

                            if layer.bias is not None:
                                temp_conv.bias.data.copy_(layer.bias.data)
                            module.conv3 = temp_conv

        elif self.settings.net_type in ["LResnetxE-IR"]:
            for module in self.pruned_model.modules():
                if isinstance(module, IRBlock):
                    block_count += 1
                    layer = module.conv2
                    if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                        temp_conv = MaskConv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None))
                        temp_conv.weight.data.copy_(layer.weight.data)

                        if layer.bias is not None:
                            temp_conv.bias.data.copy_(layer.bias.data)
                        module.conv2 = temp_conv
                        self.logger.info("replace block_count={}".format(block_count))

        elif self.settings.net_type in ["mobilefacenet_v1"]:
            for module in self.pruned_model.modules():
                if isinstance(module, Bottleneck_mobilefacenet):
                    block_count += 1
                    layer = module.conv3
                    if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                        temp_conv = MaskConv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None))
                        temp_conv.weight.data.copy_(layer.weight.data)

                        if layer.bias is not None:
                            temp_conv.bias.data.copy_(layer.bias.data)
                        module.conv3 = temp_conv
                        self.logger.info("replace block_count={}".format(block_count))


    ############################### insightface_dcp ##################################
    def face_channel_selection_for_one_segment(self, original_segment, pruned_segment, middle_layer, aux_fc, pivot_index, index):
        """
        Conduct channel selection for one segment
        :param original_segment: original network segments
        :param pruned_segment: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param pivot_index: the layer index of the additional loss
        :param index: the index of segment
        """

        if self.settings.net_type in ["LResnetxE-IR"]:
            self.channel_selection_for_insightface_resnet_segment(original_segment, pruned_segment,
                                                                  middle_layer, aux_fc, pivot_index, index)
        elif self.settings.net_type in ["mobilefacenet_v1"]:
            self.channel_selection_for_mobilefacenet_segment(original_segment, pruned_segment,
                                                                  middle_layer, aux_fc, pivot_index, index)
        return pruned_segment


    def replace_linear_layer(self, masklinear):
        temp = MaskLinear(
            in_features=masklinear.in_features,
            out_features=masklinear.out_features,
            bias=(masklinear.bias is not None))

        temp.weight.data.copy_(masklinear.weight.data)
        if masklinear.bias is not None:
            temp.bias.data.copy_(masklinear.bias.data)

        temp.pruned_weight.data.copy_(masklinear.pruned_weight.data)
        temp.d.copy_(masklinear.d)
        return temp


    def channel_selection_for_insightface_resnet_segment(self, original_segment, pruned_segment,
                                                         middle_layer, aux_fc, pivot_index, index):
        """
        Conduct channel selection for one segment in resnet
        """

        block_count = 0
        for module in pruned_segment.modules():
            if isinstance(module, IRBlock):
                block_count += 1
                # We will not prune the pruned blocks again
                if not isinstance(module.conv2, MaskConv2d):
                    # dcp
                    self.layer_channel_selection.face_channel_selection_for_one_layer(
                        original_segment, pruned_segment, middle_layer, aux_fc, module, block_count, "conv2")

                    self.logger.info("|===>checking layer type: {}".format(type(module.conv2)))
                    self.checkpoint.face_save_dcp_model(self.original_model, self.pruned_model,
                                                        self.segment_wise_trainer.middle_layer,
                                                        self.segment_wise_trainer.aux_fc,
                                                        pivot_index, index=index, block_count=block_count)

                    self.checkpoint.face_save_dcp_checkpoint(self.original_model, self.pruned_model,
                                                             self.segment_wise_trainer.middle_layer,
                                                             self.segment_wise_trainer.aux_fc,
                                                             self.segment_wise_trainer.fc_optimizer,
                                                             self.segment_wise_trainer.seg_optimizer,
                                                             pivot_index, index=index, block_count=block_count)

            elif isinstance(module, Middle_layer):
                block_count += 1
                if not isinstance(module.fc, MaskLinear):
                    self.logger.info("{}: Start pruning last_fc layer !!!".format(self.settings.net_type))
                    self.layer_channel_selection.face_channel_selection_for_one_layer(
                        original_segment, pruned_segment, middle_layer, aux_fc, module, block_count, "fc")

                    self.logger.info("|===>checking layer type: {}".format(type(module.fc)))

                    if not isinstance(self.segment_wise_trainer.pruned_model.fc, MaskLinear):
                        self.logger.info("self.segment_wise_trainer.pruned_model.fc is not MaskLinear !!!")
                        self.logger.info("|===>checking self.segment_wise_trainer.pruned_model.fc layer type: {}, \n {}"
                                         .format(type(self.segment_wise_trainer.pruned_model.fc),
                                                 self.segment_wise_trainer.pruned_model.fc))

                        self.segment_wise_trainer.pruned_model.fc = \
                            self.replace_linear_layer(self.segment_wise_trainer.pruned_last_fc_module.fc)

                    self.logger.info("|===>checking self.segment_wise_trainer.pruned_model.fc layer type: {}, \n {}\n{}"
                                     .format(type(self.segment_wise_trainer.pruned_model.fc),
                                             self.segment_wise_trainer.pruned_model.fc,
                                             self.segment_wise_trainer.pruned_model.fc.d))

                    # self.checkpoint.face_save_dcp_model(self.segment_wise_trainer.original_model,
                    #                                     self.segment_wise_trainer.pruned_model,
                    #                                     self.segment_wise_trainer.middle_layer,
                    #                                     self.segment_wise_trainer.aux_fc,
                    #                                     pivot_index, index=index, block_count=block_count)

                    self.checkpoint.face_save_dcp_checkpoint(self.segment_wise_trainer.original_model,
                                                             self.segment_wise_trainer.pruned_model,
                                                             self.segment_wise_trainer.middle_layer,
                                                             self.segment_wise_trainer.aux_fc,
                                                             self.segment_wise_trainer.fc_optimizer,
                                                             self.segment_wise_trainer.seg_optimizer,
                                                             pivot_index, index=index, block_count=block_count)

    def channel_selection_for_mobilefacenet_segment(self, original_segment, pruned_segment,
                                                         middle_layer, aux_fc, pivot_index, index):
        """
        Conduct channel selection for one segment in resnet
        """

        block_count = 0
        for module in pruned_segment.modules():
            if isinstance(module, Bottleneck_mobilefacenet):
                block_count += 1
                # if not isinstance(module.conv1, MaskConv2d):
                #     self.layer_channel_selection.face_channel_selection_for_one_layer(
                #         original_segment, pruned_segment, middle_layer, aux_fc, module, block_count, "conv1")
                #
                #     self.logger.info("|===>checking layer type: {}".format(type(module.conv1)))
                #     self.checkpoint.face_save_dcp_model(self.original_model, self.pruned_model,
                #                                         self.segment_wise_trainer.middle_layer,
                #                                         self.segment_wise_trainer.aux_fc,
                #                                         pivot_index, index=index, block_count=block_count)
                #
                #     self.checkpoint.face_save_dcp_checkpoint(self.original_model, self.pruned_model,
                #                                              self.segment_wise_trainer.middle_layer,
                #                                              self.segment_wise_trainer.aux_fc,
                #                                              self.segment_wise_trainer.fc_optimizer,
                #                                              self.segment_wise_trainer.seg_optimizer,
                #                                              pivot_index, index=index, block_count=block_count)

                if not isinstance(module.conv3, MaskConv2d):
                    self.layer_channel_selection.face_channel_selection_for_one_layer(
                        original_segment, pruned_segment, middle_layer, aux_fc, module, block_count, "conv3")

                    self.logger.info("|===>checking layer type: {}".format(type(module.conv3)))
                    # self.checkpoint.face_save_dcp_model(self.original_model, self.pruned_model,
                    #                                     self.segment_wise_trainer.middle_layer,
                    #                                     self.segment_wise_trainer.aux_fc,
                    #                                     pivot_index, index=index, block_count=block_count)

                    self.checkpoint.face_save_dcp_checkpoint(self.original_model, self.pruned_model,
                                                             self.segment_wise_trainer.middle_layer,
                                                             self.segment_wise_trainer.aux_fc,
                                                             self.segment_wise_trainer.fc_optimizer,
                                                             self.segment_wise_trainer.seg_optimizer,
                                                             pivot_index, index=index, block_count=block_count)

            elif isinstance(module, Middle_layer_mobilefacenetv1):
                block_count += 1
                if not isinstance(module.linear, MaskLinear):
                    self.logger.info("{}: Start pruning last_fc layer !!!".format(self.settings.net_type))
                    self.layer_channel_selection.face_channel_selection_for_one_layer(
                        original_segment, pruned_segment, middle_layer, aux_fc, module, block_count, "fc")

                    self.logger.info("|===>checking layer type: {}".format(type(module.linear)))
                    if not isinstance(self.segment_wise_trainer.pruned_model.linear, MaskLinear):
                        self.logger.info("self.segment_wise_trainer.pruned_model.linear is not MaskLinear !!!")
                        self.logger.info("|===>checking self.segment_wise_trainer.pruned_model.linear layer type: {}"
                                         .format(type(self.segment_wise_trainer.pruned_model.linear)))

                        self.segment_wise_trainer.pruned_model.linear = \
                            self.replace_linear_layer(self.segment_wise_trainer.pruned_last_fc_module.linear)

                    self.logger.info("|===>checking self.segment_wise_trainer.pruned_model.fc layer type: {}"
                                     .format(type(self.segment_wise_trainer.pruned_model.linear)))

                    # self.checkpoint.face_save_dcp_model(self.segment_wise_trainer.original_model,
                    #                                     self.segment_wise_trainer.pruned_model,
                    #                                     self.segment_wise_trainer.middle_layer,
                    #                                     self.segment_wise_trainer.aux_fc,
                    #                                     pivot_index, index=index, block_count=block_count)

                    self.checkpoint.face_save_dcp_checkpoint(self.segment_wise_trainer.original_model,
                                                             self.segment_wise_trainer.pruned_model,
                                                             self.segment_wise_trainer.middle_layer,
                                                             self.segment_wise_trainer.aux_fc,
                                                             self.segment_wise_trainer.fc_optimizer,
                                                             self.segment_wise_trainer.seg_optimizer,
                                                             pivot_index, index=index, block_count=block_count)

    def face_channel_selection_for_network(self):
        """
        Conduct face channel selection
        """

        # get testing error
        self.segment_wise_trainer.face_val(0)
        time_start = time.time()

        restart_index = None
        # find restart segment index
        if self.current_pivot_index:
            if self.current_pivot_index in self.settings.pivot_set:
                restart_index = self.settings.pivot_set.index(self.current_pivot_index)
            else:
                restart_index = len(self.settings.pivot_set)

        for index in range(self.num_segments):
            self.logger.info("index = {}".format(index))
            if restart_index is not None:
                if index < restart_index:
                    continue
                # elif index == restart_index:
                #     if self.current_block_count == self.current_pivot_index:
                #         continue

            if index == self.num_segments - 1:
                self.current_pivot_index = self.segment_wise_trainer.final_block_count
            else:
                self.current_pivot_index = self.settings.pivot_set[index]

            self.logger.info("self.current_pivot_index = {}".format(self.current_pivot_index))

            # conduct channel selection
            # contains [0:index] segments
            original_segment_list = []
            pruned_segment_list = []
            for j in range(index + 1):
                original_segment_list += utils.model2list(self.segment_wise_trainer.ori_segments[j])
                pruned_segment_list += utils.model2list(self.segment_wise_trainer.pruned_segments[j])

            original_segment = nn.Sequential(*original_segment_list)
            pruned_segment = nn.Sequential(*pruned_segment_list)

            net_pruned = self.face_channel_selection_for_one_segment(original_segment, pruned_segment,
                                                                     self.segment_wise_trainer.middle_layer[index],
                                                                     self.segment_wise_trainer.aux_fc[index],
                                                                     self.current_pivot_index, index)

            # self.logger.info("self.segment_wise_trainer.pruned_segments:\n{}"
            #                  .format(self.segment_wise_trainer.pruned_segments))
            #
            # self.logger.info("net_pruned:\n{}".format(net_pruned))
            self.logger.info("self.original_model:\n{}".format(self.original_model))
            self.logger.info("self.pruned_model:\n{}".format(self.pruned_model))
            self.segment_wise_trainer.face_val(0)
            self.current_pivot_index = None

        self.checkpoint.face_save_dcp_model(self.segment_wise_trainer.original_model,
                                            self.segment_wise_trainer.pruned_model,
                                            self.segment_wise_trainer.middle_layer,
                                            self.segment_wise_trainer.aux_fc,
                                            self.segment_wise_trainer.final_block_count,
                                            index=self.num_segments)

        time_interval = time.time() - time_start
        log_str = "cost time: {}".format(str(datetime.timedelta(seconds=time_interval)))
        self.logger.info(log_str)


def main():
    parser = argparse.ArgumentParser(description="Discrimination-aware channel pruning")
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='configuration path')
    parser.add_argument('--model_path', type=str, metavar='model_path',
                        help='model path of the pruned model')
    args = parser.parse_args()

    option = Option(args.conf_path)
    if args.model_path:
        option.pretrained = args.model_path

    experiment = Experiment(option)

    # experiment.channel_selection_for_network()
    experiment.face_channel_selection_for_network()


if __name__ == '__main__':
    main()
