import argparse
from easydict import EasyDict as edict

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from option import Option
from torch import nn
from trainer import NetworkWiseTrainer

from dcp.checkpoint import CheckPoint
from dcp.dataloader import *
from dcp.mask_conv import MaskConv2d, MaskLinear
from dcp.model_builder import get_model
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.logger import get_logger
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_log, write_settings
from dcp.utils.model_analyse import ModelAnalyse

from dcp.models.insightface_resnet import IRBlock
from dcp.arcface import Arcface

from dcp.models.mobilefacenet import Bottleneck_mobilefacenet
from dcp.models.mobilefacenet_pruned import pruned_Mobilefacenet, pruned_Bottleneck_mobilefacenet
from dcp.models.mobilefacenetv2_width_wm import Bottleneck_mobilefacenetv2, Mobilefacenetv2_width_wm
from dcp.ncnn_quantization import Conv2d_Ncnn_int8, replace_layer_by_unique_name


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.pruned_model = None
        self.aux_fc = None
        self.network_wise_trainer = None
        self.optimizer_state = None

        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu

        self.settings.set_save_path()
        write_settings(self.settings)
        self.logger = get_logger(self.settings.save_path, "finetune")
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)
        self.logger.info("|===>Result will be saved at {}".format(self.settings.save_path))
        self.epoch = 0
        self.test_input = None
        self.activation_value_list = edict()

        self.prepare()

    def write_settings(self):
        """
        Save expriment settings to a file
        """

        with open(os.path.join(self.settings.save_path, "settings.log"), "w") as f:
            for k, v in self.settings.__dict__.items():
                f.write(str(k) + ": " + str(v) + "\n")

    def prepare(self):
        """
        Preparing experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()

        self._get_activation_value()
        self._replace()

        self._set_trainier()
        # self.network_wise_trainer.face_before_val(0)
        # assert False

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
            self.settings.n_classes = class_num  # class number
            self.logger.info('self.settings.n_classes={:d}'.format(self.settings.n_classes))

    def _set_model(self):
        """
        Get model
        """

        self.pruned_model = get_model(self.settings,
                                      self.settings.dataset,
                                      self.settings.net_type,
                                      self.settings.depth,
                                      self.settings.n_classes)
        self.logger.info("before replace: {}".format(self.pruned_model))
        # self.logger.info(self.pruned_model)
        self.aux_fc = Arcface(self.settings.embed_size, self.settings.n_classes)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.pruned_model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        Load pre-trained model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)

            model_state = check_point_params["model"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
            self.logger.info("load pruned_model state finished!!")

            # xz codes
            aux_fc_state = check_point_params["aux_fc"]
            self.aux_fc.load_state_dict(aux_fc_state[-1])
            # self.aux_fc.load_state_dict(aux_fc_state)
            self.logger.info("load aux_fc state finished!!")

            # self.logger.info("val_acc={}".format(check_point_params['val_acc']))
            # self.logger.info(
            #     "epoch={}, val_acc={}".format(check_point_params['epoch'], check_point_params['lfw_accuracy']))
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        """
        # To do
        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)

            pruned_model_state = check_point_params["model"]
            aux_fc_state = check_point_params["aux_fc"]
            self.optimizer_state = check_point_params["optimizer"]
            self.epoch = check_point_params["epoch"] + 1
            val_acc = check_point_params["val_acc"]

            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)

            self.logger.info("|===>load resume file: {}".format(self.settings.resume))


    def _get_activation_value(self):
        for line in open(self.settings.table_path):
            line = line.strip()
            key, value = line.split(' ')

            # remove 'net.'
            if "net" in key:
                key = key.replace('net.', '')
            self.activation_value_list[key] = value

        for key in self.activation_value_list:
            self.logger.info('{} {}'.format(key, self.activation_value_list[key]))

    # old
    # def _replace(self):
    #     if self.settings.net_type == "mobilefacenet_v1_p0.5":
    #         for module in self.pruned_model.modules():
    #             if isinstance(module, (pruned_Mobilefacenet, Mobilefacenetv2_width_wm)):
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv1.in_channels,
    #                     out_channels=module.conv1.out_channels,
    #                     kernel_size=module.conv1.kernel_size,
    #                     stride=module.conv1.stride,
    #                     padding=module.conv1.padding,
    #                     groups=module.conv1.groups,
    #                     bias=(module.conv1.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv1.weight.data)
    #                 if module.conv1.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv1.bias.data)
    #                 module.conv1 = temp_conv
    #
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv2.in_channels,
    #                     out_channels=module.conv2.out_channels,
    #                     kernel_size=module.conv2.kernel_size,
    #                     stride=module.conv2.stride,
    #                     padding=module.conv2.padding,
    #                     groups=module.conv2.groups,
    #                     bias=(module.conv2.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv2.weight.data)
    #                 if module.conv2.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv2.bias.data)
    #                 module.conv2 = temp_conv
    #
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv3.in_channels,
    #                     out_channels=module.conv3.out_channels,
    #                     kernel_size=module.conv3.kernel_size,
    #                     stride=module.conv3.stride,
    #                     padding=module.conv3.padding,
    #                     groups=module.conv3.groups,
    #                     bias=(module.conv3.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv3.weight.data)
    #                 if module.conv3.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv3.bias.data)
    #                 module.conv3 = temp_conv
    #
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv4.in_channels,
    #                     out_channels=module.conv4.out_channels,
    #                     kernel_size=module.conv4.kernel_size,
    #                     stride=module.conv4.stride,
    #                     padding=module.conv4.padding,
    #                     groups=module.conv4.groups,
    #                     bias=(module.conv4.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv4.weight.data)
    #                 if module.conv4.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv4.bias.data)
    #                 module.conv4 = temp_conv
    #
    #             elif isinstance(module, (pruned_Bottleneck_mobilefacenet, Bottleneck_mobilefacenetv2)):
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv1.in_channels,
    #                     out_channels=module.conv1.out_channels,
    #                     kernel_size=module.conv1.kernel_size,
    #                     stride=module.conv1.stride,
    #                     padding=module.conv1.padding,
    #                     groups=module.conv1.groups,
    #                     bias=(module.conv1.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv1.weight.data)
    #                 if module.conv1.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv1.bias.data)
    #                 module.conv1 = temp_conv
    #
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv2.in_channels,
    #                     out_channels=module.conv2.out_channels,
    #                     kernel_size=module.conv2.kernel_size,
    #                     stride=module.conv2.stride,
    #                     padding=module.conv2.padding,
    #                     groups=module.conv2.groups,
    #                     bias=(module.conv2.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv2.weight.data)
    #                 if module.conv2.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv2.bias.data)
    #                 module.conv2 = temp_conv
    #
    #                 temp_conv = Conv2d_Ncnn_int8(
    #                     in_channels=module.conv3.in_channels,
    #                     out_channels=module.conv3.out_channels,
    #                     kernel_size=module.conv3.kernel_size,
    #                     stride=module.conv3.stride,
    #                     padding=module.conv3.padding,
    #                     groups=module.conv3.groups,
    #                     bias=(module.conv3.bias is not None))
    #                 temp_conv.weight.data.copy_(module.conv3.weight.data)
    #                 if module.conv3.bias is not None:
    #                     temp_conv.bias.data.copy_(module.conv3.bias.data)
    #                 module.conv3 = temp_conv
    #
    #     else:
    #         assert False, "unsupported net_type: {}".format(self.settings.netType)
    #
    #     self.logger.info("after replace: {}".format(self.pruned_model))

    # new
    def _replace(self):
        for name, module in self.pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                temp_conv = Conv2d_Ncnn_int8(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    groups=module.groups,
                    bias=(module.bias is not None),
                    activation_value=float(self.activation_value_list[name]))
                temp_conv.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    temp_conv.bias.data.copy_(module.bias.data)
                replace_layer_by_unique_name(self.pruned_model, name, temp_conv)

        self.logger.info("after replace: {}".format(self.pruned_model))


    def _set_trainier(self):
        """
        Initialize network-wise trainer
        """
        self.network_wise_trainer = NetworkWiseTrainer(pruned_model=self.pruned_model,
                                                       aux_fc=self.aux_fc,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       tensorboard_logger=self.tensorboard_logger,
                                                       run_count=self.epoch)


    def fine_tuning(self):
        """
        Conduct network-wise fine-tuning after channel selection
        """

        # best_top1_acc = 0

        start_epoch = 0
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        self.network_wise_trainer.face_before_val(0)
        for epoch in range(start_epoch, self.settings.n_epochs):
            train_error, train_loss, train5_error = self.network_wise_trainer.face_train(epoch)
            val_acc = self.network_wise_trainer.face_val(epoch)

            # write and print result
            log_str = "{:d}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_error, train_loss, val_acc)
            write_log(self.settings.save_path, 'log.txt', log_str)

            # self.logger.info("|===>Best Result is: Top1 acc: {:f}\n".format(best_top1_acc))
            self.checkpoint.save_face_checkpoint(self.network_wise_trainer.pruned_model,
                                                 self.network_wise_trainer.aux_fc,
                                                 self.network_wise_trainer.optimizer,
                                                 epoch, val_acc,
                                                 self.network_wise_trainer.scheduler)


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model_path', type=str, metavar='model_path',
                        help='model path of the pruned model')
    args = parser.parse_args()

    option = Option(args.conf_path)
    if args.model_path:
        option.pretrained = args.model_path

    experiment = Experiment(option)
    experiment.fine_tuning()


if __name__ == '__main__':
    main()
