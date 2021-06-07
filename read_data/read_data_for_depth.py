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
    def __init__(self,data_flag = None, transform=None, phase_train=True, data_dir=None,phase_test=False,add_mask = True):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.add_mask = add_mask
        self.mask_np = None

        #for train
        train_file = os.getcwd() +'/data/train_file_list/depth_list/depth_train_minmax_mask_1021_reverse.txt'
        label_train_file = os.getcwd() + '/data/train_file_list/depth_list/label_train_minmax_mask_1021_reverse.txt'
        
        # for val
        val_file = os.getcwd() +'/data/train_file_list/depth_list/depth_test_minmax_mask_reverse.txt'
        label_val_file = os.getcwd() + '/data/train_file_list/depth_list/label_test_minmax_mask_reverse.txt'
        
        # for test
        test_file = os.getcwd() +'/data/train_file_list/depth_list/depth_test_minmax_mask_reverse.txt'
        label_test_file = os.getcwd() + '/data/train_file_list/depth_list/label_test_minmax_mask_reverse.txt'

        print("geting data from ",train_file)

        self.mask_np = np.fromfile('./data/mask_file/mask_for_nir.bin', np.uint8).reshape((112,112))
        

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
        # depth = depth.convert('RGB')
        depth = depth.convert('L')

        
        depth = depth.point(lambda x : x+40 if x>0 else 0) #在线把亮度增强40
        # depth = depth.point(lambda x : 255-x if x>0 else 0)  #已经离线做好了像素反转
        
        print(depth_dir[idx])
        depth_np = np.array(depth)
        cv2.imwrite('./train.jpg',depth_np.astype(np.uint8))


        # print("line 87,depth test")
        # print(depth.format)
        # print(depth.mode)
        # print(depth.size)
        # print(depth.show)
        # print(depth_dir)
        

        # '''filp left and right randonly and add mask'''
        # if random.randint(0,9) < 5:
        #     depth = depth.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转

        '''transform'''
        if self.transform:
            depth = self.transform(depth)

        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]

if __name__ == "__main__":
    print(os.getcwd())
    os.chdir('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/')
    print(os.getcwd())
    print("test")
    train_dataset = CASIA(
         phase_train=True,phase_test=False, add_mask=False)
    train_dataset.__getitem__(0)
    # train_dataset[5]
