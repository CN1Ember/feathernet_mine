import numpy as np
import torch
from torch import nn
from utils.attentionTransfer_util.util import AverageMeter
from prefetch_generator import BackgroundGenerator
from utils.attentionTransfer_util.util import get_learning_rate, accuracy, record_epoch_learn_alpha, get_fc_name
from utils.attentionTransfer_util.regularizer import reg_classifier, reg_fea_map, reg_att_fea_map, reg_l2sp, \
    reg_pixel_att_fea_map_learn, reg_channel_att_fea_map_learn, reg_channel_pixel_att_fea_map_learn



class TransferFramework:

    def __init__(self, args, train_loader, target_class_num, base_model_name, model_source, source_metric_fc,
                 model_target, target_metric_fc, feature_criterions, reg_type, loss_fn, channel_weights, num_epochs, alpha,
                 beta, optimizer, lr_scheduler, writer, logger, print_freq=10):

        self.setting = args
        self.train_loader = train_loader
        self.target_class_num = target_class_num

        self.base_model_name = base_model_name
        self.model_source = model_source
        self.source_metric_fc = source_metric_fc
        self.model_target = model_target
        self.target_metric_fc = target_metric_fc

        self.loss_fn = loss_fn

        self.model_source_weights = {}
        self.model_target_weights = {}

        self.reg_type = reg_type
        self.feature_criterions = feature_criterions
        self.alpha = alpha
        self.beta = beta
        self.channel_weights = channel_weights

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.lr = 0.0
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.logger = logger
        self.print_freq = print_freq

        # framework init
        # self.fc_name = get_fc_name(self.base_model_name, self.logger)
        self.hook_layers = []
        self.layer_outputs_source = []
        self.layer_outputs_target = []
        self.framework_init()


    def framework_init(self):
        if 'fea_map' in self.reg_type:
            self.hook_setting()

        elif self.reg_type in ['l2sp']:
            for name, param in self.model_source.named_parameters():
                self.model_source_weights[name] = param.detach()
                # print('name={}'.format(name))
        self.logger.info('self.model_source_weights len = {} !'.format(len(self.model_source_weights)))

    # hook
    def _for_hook_source(self, module, input, output):
        self.layer_outputs_source.append(output)

    def _for_hook_target(self, module, input, output):
        self.layer_outputs_target.append(output)

    def register_hook(self, model, func):
        for name, layer in model.named_modules():
            if name in self.hook_layers:
                layer.register_forward_hook(func)

    def get_hook_layers(self):
        if self.base_model_name == 'LResNet34E_IR':
            self.hook_layers = ['layer1.2.conv2', 'layer2.3.conv2', 'layer3.5.conv2', 'layer4.2.conv2']

        elif self.base_model_name == 'mobilefacenet_p0.5':
            self.hook_layers = ['layers.4.conv3', 'layers.8.conv3', 'layers.11.conv3', 'layers.14.conv3']

        else:
            assert False, self.logger.info("invalid base_model_name={}".format(self.base_model_name))

    def hook_setting(self):
        # hook
        self.get_hook_layers()
        self.register_hook(self.model_source, self._for_hook_source)
        self.register_hook(self.model_target, self._for_hook_target)
        self.logger.info("self.hook_layers={}".format(self.hook_layers))


    def train(self, epoch):
        # train mode
        self.model_target.train()
        self.target_metric_fc.train()
        self.model_source.eval()

        clc_losses = AverageMeter()
        classifier_losses = AverageMeter()
        feature_losses = AverageMeter()
        # attention_losses = AverageMeter()
        total_losses = AverageMeter()
        train_top1_accs = AverageMeter()

        self.lr_scheduler.step(epoch)
        self.lr = get_learning_rate(self.optimizer)
        self.logger.info('self.optimizer={}'.format(self.optimizer))
        self.logger.info('feature_loss alpha={}, beta={}'.format(self.alpha, self.beta))
        self.logger.info('self.reg_type={}'.format(self.reg_type))

        # for i, (imgs, labels) in enumerate(target_par):
        for i, (imgs, labels) in enumerate(self.train_loader):
            # target_data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            # taget forward and loss
            outputs = self.model_target(imgs)
            outputs = self.target_metric_fc(outputs, labels)
            clc_loss = self.loss_fn(outputs, labels)

            classifier_loss = 0
            feature_loss = 0
            # attention_loss = 0

            # source_model forward for hook
            if self.reg_type not in ['l2', 'l2fe', 'l2sp']:
                with torch.no_grad():
                    # self.logger.info("model_source forward!")
                    _ = self.model_source(imgs)


            if not self.reg_type in ['l2', 'l2fe']:
                classifier_loss = reg_classifier(self.target_metric_fc)

            if self.reg_type == 'pixel_att_fea_map_learn':
                feature_loss = reg_pixel_att_fea_map_learn(self.layer_outputs_source,
                                                           self.layer_outputs_target, self.feature_criterions)

            elif self.reg_type == 'channel_att_fea_map_learn':
                feature_loss = reg_channel_att_fea_map_learn(self.layer_outputs_source,
                                                             self.layer_outputs_target, self.feature_criterions)

            total_loss = clc_loss + self.alpha * feature_loss + self.beta * classifier_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # batch update
            self.layer_outputs_source.clear()
            self.layer_outputs_target.clear()

            clc_losses.update(clc_loss.item(), imgs.size(0))
            if classifier_loss == 0:
                classifier_losses.update(classifier_loss, imgs.size(0))
            else:
                classifier_losses.update(classifier_loss.item(), imgs.size(0))

            if feature_loss == 0:
                feature_losses.update(feature_loss, imgs.size(0))
            else:
                feature_losses.update(feature_loss.item(), imgs.size(0))

            total_losses.update(total_loss.item(), imgs.size(0))

            # compute accuracy
            top1_accuracy = accuracy(outputs, labels, 1)
            train_top1_accs.update(top1_accuracy, imgs.size(0))

            # Print status
            if i % self.print_freq == 0:
                self.logger.info(
                    'Train Epoch: [{:d}/{:d}][{:d}/{:d}]\tlr={:.6f}\tclc_loss={:.4f}\t\tclassifier_loss={:.4f}'
                    '\t\tfeature_loss={:.6f}\t\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                        .format(epoch, self.num_epochs, i, len(self.train_loader), self.lr, clc_losses.avg,
                                classifier_losses.avg, feature_losses.avg, total_losses.avg, train_top1_accs.avg))

            # break

        # save tensorboard
        self.writer.add_scalar('lr', self.lr, epoch)
        self.writer.add_scalar('Train_classification_loss', clc_losses.avg, epoch)
        self.writer.add_scalar('Train_classifier_loss', classifier_losses.avg, epoch)
        self.writer.add_scalar('Train_feature_loss', feature_losses.avg, epoch)
        # self.writer.add_scalar('Train_attention_loss', attention_losses.avg, epoch)
        self.writer.add_scalar('Train_total_loss', total_losses.avg, epoch)
        self.writer.add_scalar('Train_top1_accuracy', train_top1_accs.avg, epoch)

        self.logger.info(
            '||==> Train Epoch: [{:d}/{:d}]\tTrain: lr={:.6f}\tclc_loss={:.4f}\t\tclassifier_loss={:.4f}'
            '\t\tfeature_loss={:.6f}\t\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                .format(epoch, self.num_epochs, self.lr, clc_losses.avg, classifier_losses.avg,
                        feature_losses.avg,  total_losses.avg, train_top1_accs.avg))

        return clc_losses.avg, classifier_losses.avg, feature_losses.avg, \
               total_losses.avg, train_top1_accs.avg


    # def val(self, epoch):
    #     # test mode
    #     self.model_target.eval()
    #
    #     val_losses = AverageMeter()
    #     val_top1_accs = AverageMeter()
    #
    #     # Batches
    #     for i, (imgs, labels) in enumerate(self.val_loader):
    #         # Move to GPU, if available
    #         if torch.cuda.is_available():
    #             imgs = imgs.cuda()
    #             labels = labels.cuda()
    #
    #         if self.data_aug == 'improved':
    #             bs, ncrops, c, h, w = imgs.size()
    #             imgs = imgs.view(-1, c, h, w)
    #
    #         # forward and loss
    #         with torch.no_grad():
    #             outputs = self.model_target(imgs)
    #             if self.data_aug == 'improved':
    #                 outputs = outputs.view(bs, ncrops, -1).mean(1)
    #
    #             val_loss = self.loss_fn(outputs, labels)
    #
    #         val_losses.update(val_loss.item(), imgs.size(0))
    #         # compute accuracy
    #         top1_accuracy = accuracy(outputs, labels, 1)
    #         val_top1_accs.update(top1_accuracy, imgs.size(0))
    #
    #         # batch update
    #         self.layer_outputs_source.clear()
    #         self.layer_outputs_target.clear()
    #
    #         # Print status
    #         if i % self.print_freq == 0:
    #             self.logger.info('Val Epoch: [{:d}/{:d}][{:d}/{:d}]\tval_loss={:.4f}\t\ttop1_accuracy={:.4f}\t'
    #                         .format(epoch, self.num_epochs, i, len(self.val_loader), val_losses.avg, val_top1_accs.avg))
    #         # break
    #
    #     self.writer.add_scalar('Val_loss', val_losses.avg, epoch)
    #     self.writer.add_scalar('Val_top1_accuracy', val_top1_accs.avg, epoch)
    #
    #     self.logger.info('||==> Val Epoch: [{:d}/{:d}]\tval_oss={:.4f}\t\ttop1_accuracy={:.4f}'
    #                      .format(epoch, self.num_epochs, val_losses.avg, val_top1_accs.avg))
    #
    #     return val_losses.avg, val_top1_accs.avg















