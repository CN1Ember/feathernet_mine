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
    def __init__(self,data_flag = None, transform=None, phase_train=True, data_dir=None,phase_test=False,add_mask = True,feat_map_shape = (14,14)):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.add_mask = add_mask
        self.mask_np = None
        self.shape = feat_map_shape

        #for train
        train_file = os.getcwd() +'/data/train_file_list/%s_train.txt'%data_flag
        label_train_file = os.getcwd() + '/data/train_file_list/%s_train_label.txt'%data_flag
        
        # for val
        val_file = os.getcwd() +'/data/train_file_list/%s_val.txt'%data_flag
        label_val_file = os.getcwd() + '/data/train_file_list/%s_val_label.txt'%data_flag
        
        # for test
        test_file = os.getcwd() +'/data/train_file_list/%s_val.txt'%data_flag
        label_test_file = os.getcwd() + '/data/train_file_list/%s_val_label.txt'%data_flag

        self.mask_np = np.fromfile('./data/mask_file/mask_for_nir.bin', np.uint8).reshape((112,112))
        self.mask_np = cv2.resize(self.mask_np,self.shape,cv2.INTER_NEAREST)

        try:
            with open(train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
                
            with open(val_file, 'r') as f:
                 self.depth_dir_val = f.read().splitlines()
            with open(label_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_test_file, 'r') as f:
                    self.label_dir_test = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test
                label_dir = self.label_dir_test
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)

        depth = Image.open(depth_dir[idx])
        depth = depth.convert('L')
        fmap_mask = np.zeros((112,112),np.uint8)
        fmap_mask[51 - 6:51 + 6,38 - 15:38 + 15] = abs(label - 1)
        fmap_mask[51 - 6:51 + 6,73 - 15:73 + 15] = abs(label - 1)
        fmap_mask = cv2.resize(fmap_mask,(14,14),cv2.INTER_NEAREST)

        fmap_mask2 = np.zeros(self.shape, np.uint8)
        fmap_mask2[1:self.shape[0] - 1,1:self.shape[0] - 1] = label
        # fmap_mask2 = np.bitwise_and(fmap_mask2,self.mask_np)

        # if label == 1:
            # cv2.imwrite('./test.jpg',fmap_mask2 * 255)
        

        '''filp left and right randonly and add mask'''
        if random.randint(0,9) < 5:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转
            fmap_mask = np.fliplr(fmap_mask).copy()
            # fmap_mask2 = np.fliplr(fmap_mask2).copy()



        '''transform'''
        if self.transform:
            depth = self.transform(depth)

        if self.phase_train:
            return depth,label,fmap_mask2
        else:
            return depth,label,depth_dir[idx]

