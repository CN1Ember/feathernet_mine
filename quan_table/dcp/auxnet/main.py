import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from aux_checkpoint import AuxCheckPoint
from aux_trainer import AuxTrainer
from option import Option

from dcp.dataloader import *
from dcp.model_builder import get_model
from dcp.utils.logger import get_logger
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_log, write_settings

block_num = {'vgg19': 16, 'preresnet56': 27, 'resnet18': 8,'LResnetxE-IR34':16, "mobilefacenet_v1":15,
              "mobilefacenet_v2":35, "mobilenet_v3": 15, 'zq_mobilefacenet':35, 'resnet50': 16}


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None

        self.trainer = None
        self.seg_opt_state = None
        self.fc_opt_state = None
        self.aux_fc_state = None
        self.middle_layer_state = None
        self.final_aux_fc_state = None

        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu
        self.settings.set_save_path()
        write_settings(self.settings)
        self.logger = get_logger(self.settings.save_path, "auxnet")
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)
        self.logger.info("|===>Result will be saved at {}".format(self.settings.save_path))

        self.epoch = 0
        self.prepare()

    def prepare(self):
        """
        Prepare experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._cal_pivot()
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
        Create train loader and validation loader for auxnet
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
        elif self.settings.dataset in ['ms1m_v2', 'iccv_ms1m', 'sub_iccv_ms1m']:
            self.train_loader, self.val_loader, class_num = get_ms1m_dataloader(self.settings.dataset,
                                                                     self.settings.batch_size,
                                                                     self.settings.n_threads,
                                                                     self.settings.data_path,
                                                                     self.logger)

        elif self.settings.dataset in ['sub_webface_0.1']:
            self.train_loader, self.val_loader, class_num = get_webface_dataloader(self.settings.dataset,
                                                                                self.settings.batch_size,
                                                                                self.settings.n_threads,
                                                                                self.settings.data_path,
                                                                                self.logger)

            self.test_loader = get_webface_testing_dataloader(self.settings.dataset,
                                                              self.settings.batch_size,
                                                              self.settings.n_threads,
                                                              self.settings.data_path,
                                                              self.logger)

        self.settings.n_classes = class_num   # class number
        self.logger.info('self.settings.n_classes={:d}'.format(self.settings.n_classes))

    def _set_model(self):
        """
        Available model
        cifar:
            preresnet
            vgg
        imagenet:
            resnet
        """

        self.model = get_model(self.settings,
                               self.settings.dataset,
                               self.settings.net_type,
                               self.settings.depth,
                               self.settings.n_classes)
        self.logger.info("set model:{}".format(self.model))

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.model is not None, "please create model first"

        self.checkpoint = AuxCheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        Load pretrained model
        the pretrained model contains state_dicts of model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)
            model_state = check_point_params['model']

            # self.final_aux_fc_state = check_point_params['arcface']
            self.aux_fc_state = check_point_params["aux_fc"]
            self.middle_layer_state = check_point_params["middle_layer"]

            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        the checkpoint contains state_dicts of model and optimizer, as well as training epoch
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            model_state = check_point_params["model"]
            self.middle_layer_state = check_point_params["middle_layer"]
            self.aux_fc_state = check_point_params["aux_fc"]
            self.seg_opt_state = check_point_params["seg_opt"]
            self.fc_opt_state = check_point_params["fc_opt"]
            self.epoch = check_point_params["epoch"]
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    def _set_trainier(self):
        """
        Initialize trainer for AuxNet
        """

        self.trainer = AuxTrainer(model=self.model,
                                  train_loader=self.train_loader,
                                  val_loader=self.val_loader,
                                  test_loader= self.test_loader,
                                  settings=self.settings,
                                  logger=self.logger,
                                  tensorboard_logger=self.tensorboard_logger)

        # if self.settings.net_type not in ["LResnetxE-IR", "mobilefacenet_v1"]:
        #     if self.seg_opt_state is not None:
        #         self.trainer.update_aux_fc(aux_fc_state=self.aux_fc_state,
        #                                    aux_fc_opt_state=self.fc_opt_state,
        #                                    seg_opt_state=self.seg_opt_state)
        #     if self.aux_fc_state is not None:
        #         self.trainer.update_aux_fc(aux_fc_state=self.aux_fc_state,
        #                                    aux_fc_opt_state=self.fc_opt_state,
        #                                    seg_opt_state=self.seg_opt_state)
        #
        # elif self.settings.net_type in ["LResnetxE-IR", "mobilefacenet_v1"]:
        #
        #     if self.aux_fc_state is not None:
        #         self.trainer.face_update_aux_fc(aux_fc_state=self.aux_fc_state,
        #                                         middle_layer_state=self.middle_layer_state,
        #                                         aux_fc_opt_state=self.fc_opt_state,
        #                                         count= self.epoch+1)
        #
        #     if self.final_aux_fc_state is not None:
        #         self.trainer.face_update_final_aux_fc(self.final_aux_fc_state)

        if self.settings.net_type in ["mobilefacenet_v1", "mobilefacenet_v2"]:
            if self.aux_fc_state is not None and self.middle_layer_state is not None:
                self.logger.info("Start load first two auxnet layer!!!")
                self.trainer.face_finetune_softmax_to_arcface(aux_fc_state=self.aux_fc_state,
                                                              middle_layer_state=self.middle_layer_state)

    def _cal_pivot(self):
        """
        Calculate the block index for additional loss
        """

        self.num_segments = self.settings.n_losses + 1
        if self.settings.net_type not in ["mobilefacenet_v1", "zq_mobilefacenet", "mobilefacenet_v2", "mobilenet_v3"]:
            num_block_per_segment = (block_num[self.settings.net_type + str(self.settings.depth)] // self.num_segments) + 1
        else:
            num_block_per_segment = (block_num[self.settings.net_type ] // self.num_segments) + 1
        pivot_set = []
        for i in range(self.num_segments - 1):
            pivot_set.append(num_block_per_segment * (i + 1))

        # insightface_dcp
        # if self.settings.net_type == "LResnetxE-IR" and self.settings.depth == 34 and self.settings.n_losses == 3:
        #     pivot_set = [4, 8, 12]

        self.settings.pivot_set = pivot_set
        self.logger.info("pivot set: {}".format(pivot_set))

    def run(self):
        """
        Learn the parameters of the additional classifier and
        fine tune model with the additional losses and the final loss
        """

        best_top1 = 100
        best_top5 = 100

        start_epoch = 0
        # if load resume checkpoint
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        self.trainer.val(0)

        for epoch in range(start_epoch, self.settings.n_epochs):
            train_error, train_loss, train5_error = self.trainer.train(epoch)
            val_error, val_loss, val5_error = self.trainer.val(epoch)

            # write log
            log_str = "{:d}\t".format(epoch)
            for i in range(len(train_error)):
                log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
                    train_error[i], train_loss[i], val_error[i],
                    val_loss[i], train5_error[i], val5_error[i])
            write_log(self.settings.save_path, 'log.txt', log_str)

            # save model and checkpoint
            best_flag = False
            if best_top1 >= val_error[-1]:
                best_top1 = val_error[-1]
                best_top5 = val5_error[-1]
                best_flag = True

            if best_flag:
                self.checkpoint.save_aux_model(self.trainer.model, self.trainer.aux_fc)

            self.logger.info("|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
            self.logger.info("|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                                     100 - best_top5))

            if self.settings.dataset in ["imagenet"]:
                self.checkpoint.save_aux_checkpoint(self.trainer.model, self.trainer.seg_optimizer,
                                                    self.trainer.fc_optimizer, self.trainer.aux_fc, epoch, epoch + 1)
            else:
                self.checkpoint.save_aux_checkpoint(self.trainer.model, self.trainer.seg_optimizer,
                                                    self.trainer.fc_optimizer, self.trainer.aux_fc, epoch)


    ############################ for insightface_dcp  ##################################
    def face_run(self):
        """
        Learn the parameters of the additional classifier and
        fine tune model with the additional losses and the final loss
        """

        best_top1_acc = 0
        start_epoch = 0
        # if load resume checkpoint
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        # self.trainer.face_train_before_val(0)
        for epoch in range(start_epoch, self.settings.n_epochs):

            train_error, train_loss, train5_error = self.trainer.face_train(epoch)
            # train_error, train_loss, train5_error = self.trainer.face_fixed_train(epoch)
            self.trainer.face_test(epoch)

            tpr_acc = self.trainer.face_val(epoch)

            # write log
            log_str = "{:d}\t".format(epoch)
            for i in range(len(train_error)):
                log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(train_error[i], train_loss[i], train5_error[i], tpr_acc)
            write_log(self.settings.save_path, 'log.txt', log_str)

            # save model and checkpoint
            best_flag = False
            if best_top1_acc <= tpr_acc:
                best_top1_acc = tpr_acc
                best_flag = True

            if best_flag:
                self.checkpoint.face_save_aux_model(self.trainer.model, self.trainer.middle_layer, self.trainer.aux_fc, best_top1_acc)

            self.logger.info("|==>Best Result is: Top1 Accuracy: {:f}\n".format(best_top1_acc))
            self.checkpoint.face_save_aux_checkpoint(self.trainer.model, self.trainer.middle_layer, self.trainer.aux_fc,
                                                     self.trainer.seg_optimizer, self.trainer.fc_optimizer, epoch)


def main():
    parser = argparse.ArgumentParser(description='AuxNet')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='the path of config file for training (default: 64)')
    args = parser.parse_args()

    option = Option(args.conf_path)
    experiment = Experiment(option)

    # experiment.run()
    experiment.face_run()


if __name__ == '__main__':
    main()
