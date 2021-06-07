import math
import time
import numpy as np

import torch.autograd
import torch.nn as nn
from torch import optim

from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import dcp.utils as utils
from dcp.utils.verifacation import val_evaluate

class NetworkWiseTrainer(object):
    """
        Network-wise trainer for fine tuning after channel selection
    """

    def __init__(self, pruned_model, aux_fc, train_loader, val_loader, settings, logger, tensorboard_logger, run_count=0):
        self.pruned_model = utils.data_parallel(pruned_model, settings.n_gpus)
        self.aux_fc = utils.data_parallel(aux_fc, settings.n_gpus)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.settings = settings
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.run_count = run_count

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            # params=self.pruned_model.parameters(),
            [{'params': self.pruned_model.parameters()},
             {'params': self.aux_fc.parameters()}],
            lr=self.settings.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay,
            nesterov=True)
        self.lr = self.settings.lr
        self.scalar_info = {}

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.n_epochs)

    def update_model(self, pruned_model, optimizer_state=None):
        """
        Update pruned model parameter
        :param pruned_model: pruned model
        """

        self.optimizer = None
        self.pruned_model = utils.data_parallel(pruned_model, self.settings.n_gpus)
        self.optimizer = torch.optim.SGD(
            # params=self.pruned_model.parameters(),
            [{'params': self.pruned_model.parameters()},
             {'params': self.aux_fc.parameters()}],
            lr=self.settings.lr,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay,
            nesterov=True)
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.n_epochs)

    def update_optimizer(self, optimizer_state):
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

    def update_lr(self, epoch):
        """
        Update learning rate of optimizers
        :param epoch: current training epoch
        """

        if epoch < self.settings.warmup_n_epochs:
            self.lr = self.settings.warmup_lr
            lr = self.settings.warmup_lr
        else:
            gamma = 0
            for step in self.settings.step:
                if epoch + 1.0 > int(step):
                    gamma += 1
            lr = self.settings.lr * math.pow(0.1, gamma)
            self.lr = lr
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group['lr']]
        return lr[len(lr) - 1]

    def forward(self, images, labels=None):
        """
        Forward propagation
        """

        # forward and backward and optimize
        output = self.pruned_model(images)

        if labels is not None:
            loss = self.criterion(output, labels)
            return output, loss
        else:
            return output, None

    def backward(self, loss):
        """
        Backward propagation
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    ######################### insightface_dcp ##############################
    def face_forward(self, images, labels=None):
        """
        Forward propagation
        """

        # forward and backward and optimize
        tmp_output = self.pruned_model(images)
        output = self.aux_fc(tmp_output, labels)

        loss = self.criterion(output, labels)
        return output, loss

    def face_train(self, epoch):
        """
        Training
        """

        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        iters = len(self.train_loader)

        # switch to train mode
        self.pruned_model.train()
        self.aux_fc.train()

        start_time = time.time()
        end_time = start_time

        if self.settings.cos_lr:
            self.scheduler.step(epoch)
            self.lr = self.get_learning_rate()
            self.logger.info("CosineAnnealingLR update !!!")
        else:
            self.update_lr(epoch)

        # use prefetch_generator and tqdm for iterating through data
        pbar = enumerate(BackgroundGenerator(self.train_loader))

        # for i, (images, labels) in enumerate(self.train_loader):
        for i, (images, labels) in pbar:
            start_time = time.time()
            data_time = start_time - end_time

            # if self.settings.n_gpus == 1:
            images = images.cuda()
            labels = labels.cuda()

            output, loss = self.face_forward(images, labels)
            self.backward(loss)

            single_error, single_loss, single5_error = utils.compute_singlecrop_error(
                outputs=output, labels=labels,
                loss=loss, top5_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()
            iter_time = end_time - start_time

            if i % self.settings.print_frequency == 0:
                utils.print_result(
                    epoch, self.settings.n_epochs, i + 1,
                    iters, self.lr, data_time, iter_time,
                    single_error,
                    single_loss, top5error=single5_error,
                    mode="Train",
                    logger=self.logger)
            # break

        self.scalar_info['network_wise_fine_tune_train_top1_error'] = top1_error.avg
        self.scalar_info['network_wise_fine_tune_train_top5_error'] = top5_error.avg
        self.scalar_info['network_wise_fine_tune_train_loss'] = top1_loss.avg
        self.scalar_info['network_wise_fine_tune_lr'] = self.lr

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        self.logger.info(
            "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(top1_error.avg, top1_loss.avg,
                                                                                  top5_error.avg))

        # update learning_rate
        # if self.settings.cos_lr:
        #     self.scheduler.step()
        #     self.lr = self.get_learning_rate()
        #     self.logger.info("CosineAnnealingLR update !!!")
        # else:
        #     self.update_lr(epoch)

        return top1_error.avg, top1_loss.avg, top5_error.avg

    # def val_evaluate(self, model, carray, issame, emb_size=512, batch_size=512, nrof_folds=10):
    #     idx = 0
    #     embeddings = np.zeros([len(carray), emb_size])
    #     with torch.no_grad():
    #         while idx + batch_size <= len(carray):
    #             batch = torch.tensor(carray[idx:idx + batch_size])
    #             out = model(batch.cuda())
    #             embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
    #             idx += batch_size
    #
    #         if idx < len(carray):
    #             batch = torch.tensor(carray[idx:])
    #             out = model(batch.cuda())
    #             embeddings[idx:idx + batch_size] = l2_norm(out.cpu())  # xz: add l2_norm
    #
    #     tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    #     return accuracy.mean()


    def face_val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        self.pruned_model.eval()
        # self.aux_fc.eval()

        with torch.no_grad():
            lfw_accuracy, _ = val_evaluate(self.pruned_model, self.val_loader[-1]['lfw'],
                                             self.val_loader[-1]['lfw_issame'], emb_size=self.settings.embed_size)
            cfp_fp_accuracy, _ = val_evaluate(self.pruned_model, self.val_loader[-2]['cfp_fp'],
                                                self.val_loader[-2]['cfp_fp_issame'], emb_size=self.settings.embed_size)
            agedb_30_accuracy, _ = val_evaluate(self.pruned_model, self.val_loader[-3]['agedb_30'],
                                                  self.val_loader[-3]['agedb_30_issame'], emb_size=self.settings.embed_size)

        self.tensorboard_logger.scalar_summary("val_lfw_acc", lfw_accuracy, self.run_count)
        self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_accuracy))

        self.tensorboard_logger.scalar_summary("cfp_fp_accuracy", cfp_fp_accuracy, self.run_count)
        self.logger.info("|===>Validation cfp_fp acc: {:4f}".format(cfp_fp_accuracy))

        self.tensorboard_logger.scalar_summary("agedb_30_accuracy", agedb_30_accuracy, self.run_count)
        self.logger.info("|===>Validation agedb_30 acc: {:4f}".format(agedb_30_accuracy))

        self.run_count += 1
        return lfw_accuracy

    def face_before_val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """
        self.pruned_model.eval()
        # self.aux_fc.eval()

        with torch.no_grad():
            lfw_accuracy, _ = val_evaluate(self.pruned_model, self.val_loader[-1]['lfw'],
                                             self.val_loader[-1]['lfw_issame'], emb_size=self.settings.embed_size)

        self.logger.info("|===>Validation lfw acc: {:4f}".format(lfw_accuracy))
        return lfw_accuracy
