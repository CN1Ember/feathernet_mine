import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from option import Option
from pruning import ResModelPrune, SeqModelPrune
from torch import nn
from trainer import NetworkWiseTrainer

from dcp.checkpoint import CheckPoint
from dcp.dataloader import *
from dcp.mask_conv import MaskConv2d
from dcp.model_builder import get_model
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.logger import get_logger
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_log, write_settings
from dcp.utils.model_analyse import ModelAnalyse

from dcp.models.insightface_resnet import IRBlock
from dcp.arcface import Arcface

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

        self.pruning_rate = 0.5
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
        self._set_trainier()

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

        elif self.settings.dataset == 'ms1m_v2':
            self.train_loader, self.val_loader, class_num = get_ms1m_dataloader(self.settings.dataset,
                                                                     self.settings.batch_size,
                                                                     self.settings.n_threads,
                                                                     self.settings.data_path,
                                                                     self.logger)
            self.settings.n_classes = class_num   # class number
            self.logger.info('self.settings.n_classes={:d}'.format(self.settings.n_classes))

    def replace_layer_with_mask_conv_resnet(self):
        """
        Replace the conv layer in resnet with mask_conv for ResNet
        """

        for module in self.pruned_model.modules():
            if isinstance(module, (PreBasicBlock, BasicBlock, Bottleneck, IRBlock)):
                # replace conv2
                temp_conv = MaskConv2d(
                    in_channels=module.conv2.in_channels,
                    out_channels=module.conv2.out_channels,
                    kernel_size=module.conv2.kernel_size,
                    stride=module.conv2.stride,
                    padding=module.conv2.padding,
                    bias=(module.conv2.bias is not None))

                temp_conv.weight.data.copy_(module.conv2.weight.data)
                if module.conv2.bias is not None:
                    temp_conv.bias.data.copy_(module.conv2.bias.data)
                module.conv2 = temp_conv

                if isinstance(module, Bottleneck):
                    # replace conv3
                    temp_conv = MaskConv2d(
                        in_channels=module.conv3.in_channels,
                        out_channels=module.conv3.out_channels,
                        kernel_size=module.conv3.kernel_size,
                        stride=module.conv3.stride,
                        padding=module.conv3.padding,
                        bias=(module.conv3.bias is not None))

                    temp_conv.weight.data.copy_(module.conv3.weight.data)
                    if module.conv3.bias is not None:
                        temp_conv.bias.data.copy_(module.conv3.bias.data)
                    module.conv3 = temp_conv

    def replace_layer_with_mask_conv_vgg(self):
        """
        Replace the conv layer in resnet with mask_conv for VGG
        """
        new_net = None
        for layer in self.pruned_model.features.modules():
            if isinstance(layer, nn.Conv2d):
                if new_net is None:
                    new_net = nn.Sequential(layer)
                    continue
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

                new_net.add_module(str(len(new_net)), temp_conv)

            elif isinstance(layer, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
                if new_net is None:
                    new_net = nn.Sequential(layer)
                else:
                    new_net.add_module(str(len(new_net)), layer)
        # print self.model.features
        # print new_net
        self.pruned_model.features = new_net

    def replace_layer_with_mask_conv(self):
        """
        Replace the conv layer in resnet with mask_conv
        """

        if self.settings.net_type in ["preresnet", "resnet", "LResnetxE-IR"]:
            self.replace_layer_with_mask_conv_resnet()
        elif self.settings.net_type in ["vgg"]:
            self.replace_layer_with_mask_conv_vgg()

    def _set_model(self):
        """
        Get model
        """

        self.pruned_model = get_model(self.settings.dataset,
                                      self.settings.net_type,
                                      self.settings.depth,
                                      self.settings.n_classes,
                                      self.pruning_rate)
        # self.replace_layer_with_mask_conv()
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
            aux_fc_state = check_point_params["aux_fc"]
            self.aux_fc.load_state_dict(aux_fc_state)
            self.logger.info("load aux_fc state finished!!")

            self.optimizer_state = check_point_params["optimizer"]
            self.epoch = check_point_params["epoch"]
            val_acc = check_point_params["val_acc"]
            self.logger.info("load resume: epoch={:.1f}, val_lfw_acc={:.4f}".format(self.epoch, val_acc))

            # self.network_wise_trainer.update_optimizer(self.optimizer_state)
            # self.network_wise_trainer.run_count = self.epoch
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))
            # self.network_wise_trainer.face_before_val(0)

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
            self.epoch = check_point_params["epoch"]
            val_acc = check_point_params["val_acc"]

            self.logger.info("load resume: epoch={:.1f}, val_lfw_acc={:.4f}".format(self.epoch, val_acc))

            # self.network_wise_trainer.pruned_model.load_state_dict(pruned_model_state)
            if isinstance(self.network_wise_trainer.pruned_model, nn.DataParallel):
                self.network_wise_trainer.pruned_model.module.load_state_dict(pruned_model_state)
            else:
                self.network_wise_trainer.pruned_model.load_state_dict(pruned_model_state)

            # self.network_wise_trainer.aux_fc.load_state_dict(aux_fc_state)
            if isinstance(self.network_wise_trainer.aux_fc, nn.DataParallel):
                self.network_wise_trainer.aux_fc.module.load_state_dict(aux_fc_state)
            else:
                self.network_wise_trainer.aux_fc.load_state_dict(aux_fc_state)

            self.network_wise_trainer.update_optimizer(self.optimizer_state)
            self.network_wise_trainer.run_count = self.epoch
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))
            self.network_wise_trainer.face_before_val(0)

    def _set_trainier(self):
        """
        Initialize network-wise trainer
        """
        self.network_wise_trainer = NetworkWiseTrainer(pruned_model=self.pruned_model,
                                                       aux_fc = self.aux_fc,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       tensorboard_logger=self.tensorboard_logger,
                                                       run_count=self.epoch)
        self.network_wise_trainer.update_optimizer(self.optimizer_state)
        self.network_wise_trainer.run_count = self.epoch
        self.network_wise_trainer.face_before_val(0)

    def pruning(self):
        """
        Prune channels
        """
        self.logger.info("Before pruning:")
        self.logger.info(self.pruned_model)

        self.network_wise_trainer.face_before_val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))

        if self.settings.net_type in ["preresnet", "resnet", "LResnetxE-IR"]:
            model_prune = ResModelPrune(model=self.pruned_model,
                                        net_type=self.settings.net_type,
                                        depth=self.settings.depth)
        elif self.settings.net_type == "vgg":
            model_prune = SeqModelPrune(model=self.pruned_model, net_type=self.settings.net_type)
        else:
            assert False, "unsupport net_type: {}".format(self.settings.net_type)

        model_prune.run()
        # After channel pruning
        self.network_wise_trainer.update_model(model_prune.model, self.optimizer_state)

        self.logger.info("After pruning:")
        self.logger.info(self.pruned_model)

        self.network_wise_trainer.face_before_val(0)
        model_analyse = ModelAnalyse(self.pruned_model, self.logger)
        params_num = model_analyse.params_count()
        zero_num = model_analyse.zero_count()
        zero_rate = zero_num * 1.0 / params_num
        self.logger.info("zero rate is: {}".format(zero_rate))

        # xz codes
        # save dcp_model with aux_fc_state
        # self.checkpoint.save_aux_model(self.pruned_model)

    def fine_tuning(self):
        """
        Conduct network-wise fine-tuning after channel selection
        """

        best_top1_acc = 0
        start_epoch = 0
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        for epoch in range(start_epoch, self.settings.n_epochs):
            train_error, train_loss, train5_error = self.network_wise_trainer.face_train(epoch)
            val_acc = self.network_wise_trainer.face_val(epoch)

            # write and print result
            log_str = "{:d}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_error, train_loss, val_acc)
            write_log(self.settings.save_path, 'log.txt', log_str)

            best_flag = False
            if best_top1_acc <= val_acc:
                best_top1_acc = val_acc
                best_flag = True

            if best_flag:
                self.checkpoint.save_face_model(self.network_wise_trainer.pruned_model, best_flag)

            self.logger.info("|===>Best Result is: Top1 acc: {:f}\n".format(best_top1_acc))
            if epoch <= self.settings.n_epochs/2:
                if epoch%2 ==0:
                    self.checkpoint.save_face_checkpoint(self.network_wise_trainer.pruned_model, self.network_wise_trainer.aux_fc,
                                                         self.network_wise_trainer.optimizer, epoch, val_acc,
                                                         self.network_wise_trainer.scheduler)
            else:
                self.checkpoint.save_face_checkpoint(self.network_wise_trainer.pruned_model, self.network_wise_trainer.aux_fc,
                                                     self.network_wise_trainer.optimizer, epoch, val_acc,
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
    # experiment.pruning()
    # experiment._load_resume()
    experiment.fine_tuning()


if __name__ == '__main__':
    main()
