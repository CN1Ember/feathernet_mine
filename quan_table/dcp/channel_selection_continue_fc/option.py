import os
import shutil

from pyhocon import ConfigFactory


class Option(object):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)

        # ------------- general options ----------------------------------------
        self.save_path = self.conf['save_path']  # log path
        self.data_path = self.conf['data_path']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar10
        self.seed = self.conf['seed']  # manually set RNG seed
        self.gpu = self.conf['gpu']  # GPU id to use, e.g. "0,1,2,3"
        self.n_gpus = len(self.gpu.split(','))  # number of GPUs to use by default

        # ------------- data options -------------------------------------------
        self.n_threads = self.conf['n_threads']  # number of threads used for data loading
        self.n_classes = self.conf['n_classes']  # number of classes in the dataset
        self.prepare_features = self.conf['prepare_features']  # whether to prepare feature

        # ------------- discrimination-aware options ---------------------------
        self.n_losses = self.conf['n_losses']  # number of additional losses
        self.pruning_rate = self.conf['pruning_rate']  # pruning rate
        self.softmax_weight = self.conf['softmax_weight']  # weight of the softmax loss
        self.mse_weight = self.conf['mse_weight']  # weight of the mean square loss
        # self.correlation_weight = self.conf['correlation_weight']  # weight of the feature and semantic correlation loss
        self.max_samples = self.conf['max_samples']  # maximum sample size used for channel selection, -1 means using whole data set

        # ------------- common optimization options ----------------------------
        self.batch_size = self.conf['batch_size']  # mini-batch size
        self.momentum = self.conf['momentum']  # momentum
        self.weight_decay = float(self.conf['weight_decay'])  # weight decay
        # self.lr = self.conf['lr']  # initial learning rate
        self.lr = 0.1
        self.layer_wise_lr = self.conf['layer_wise_lr']  # initial learning rate

        # ------------- model options ------------------------------------------
        self.net_type = self.conf['net_type']  # options: resnet | preresnet | vgg
        self.experiment_id = self.conf['experiment_id']  # identifier for experiment
        self.depth = self.conf['depth']  # resnet depth: (n-2)%6==0
        self.embed_size = self.conf['embed_size']  # resnet depth: (n-2)%6==0

        # ---------- resume or pretrained options ---------------------------------
        # path to model to pretrained with, load model state_dict only
        self.pretrained = None if len(self.conf['pretrained']) == 0 else self.conf['pretrained']
        # path to directory containing checkpoint, load state_dicts of model and optimizer, as well as training epoch
        self.resume = None if len(self.conf['resume']) == 0 else self.conf['resume']

    def params_check(self):
        if self.dataset in ["cifar10"]:
            self.n_classes = 10
        elif self.dataset == "imagenet":
            self.n_classes = 1000

    def set_save_path(self):
        self.params_check()

        if self.net_type in ["preresnet", "resnet"]:
            self.save_path = self.save_path + \
                             "log_cs_{}_d{:d}_{}_bs{:d}_n{:d}_p{:.3f}_lr{:.3f}_{}/"\
                                 .format(self.net_type, self.depth, self.dataset, self.batch_size, self.n_losses,
                                         self.pruning_rate, self.layer_wise_lr,  self.experiment_id)
        else:
            self.save_path = self.save_path + \
                             "log_cs_{}_d{:d}_{}_bs{:d}_n{:d}_p{:.3f}_lr{:.3f}_{}/" \
                                 .format(self.net_type, self.depth, self.dataset, self.batch_size, self.n_losses,
                                         self.pruning_rate, self.layer_wise_lr,  self.experiment_id)

        if os.path.exists(self.save_path):
            print("{} file exist!".format(self.save_path))
            action = input("Select Action: d (delete) / q (quit):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(self.save_path)
            else:
                raise OSError("Directory {} exits!".format(self.save_path))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
