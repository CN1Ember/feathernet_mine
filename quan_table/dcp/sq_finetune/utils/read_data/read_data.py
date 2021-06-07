from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import math
import cv2
import torchvision
import torch
import random

class CASIA(Dataset):
    def __init__(self,data_flag = None, transform=None):

        self.transform = transform


        #for train
        train_file = '/home/lidaiyuan/feathernet2020/FeatherNet/data/train_file_list/%s_train_quan_f1.txt'%data_flag
        label_train_file = '/home/lidaiyuan/feathernet2020/FeatherNet/data/train_file_list/%s_train_quan_f1_label.txt'%data_flag
        
        # for test
        try:
            with open(train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):

            return len(self.depth_dir_train)


    def __getitem__(self, idx):

        depth_dir = self.depth_dir_train
        label_dir = self.label_dir_train
        label = int(label_dir[idx])
        label = np.array(label)

        depth = Image.open(depth_dir[idx])
        depth = depth.convert('L')
        

        '''transform'''
        if self.transform:
            depth = self.transform(depth)

        return depth,label

