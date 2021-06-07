from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import math
import cv2
import torchvision
import torch


# source train set
source_train_img_file = os.getcwd() +'/data/nir_adapt_0423/nir_train_indoor.txt'
source_train_label_file = os.getcwd() + '/data/nir_adapt_0423/label_train_indoor.txt'

# source valid set
source_val_img_file = os.getcwd() +'/data/nir_adapt_0423/nir_val_indoor.txt'
source_val_label_file = os.getcwd() +'/data/nir_adapt_0423/label_val_indoor.txt' #val-label 100%

# target train set
target_train_img_file = os.getcwd() +'/data/nir_adapt_0423/nir_train_company.txt'
target_train_label_file = os.getcwd() + '/data/nir_adapt_0423/label_train_company.txt'

# target valid set
target_val_img_file = os.getcwd() +'/data/nir_adapt_0423/nir_val_company.txt'
target_val_label_file = os.getcwd() +'/data/nir_adapt_0423/label_val_company.txt' #val-label 100%

target_test_img_file = os.getcwd() +'/data/nir_adapt_0423/nir_val_company.txt'
target_test_label_file = os.getcwd() +'/data/nir_adapt_0423/label_val_company.txt'


class SourceData(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None,phase_test=False):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform

        try:
            with open(source_train_img_file, 'r') as f:
                self.source_train_img = f.read().splitlines()
            with open(source_train_label_file, 'r') as f:
                self.source_train_label = f.read().splitlines()
                
            with open(source_val_img_file, 'r') as f:
                 self.source_val_img = f.read().splitlines()
            with open(source_val_label_file, 'r') as f:
                self.source_val_label = f.read().splitlines()
            if self.phase_test:
                with open(target_test_img_file, 'r') as f:
                    self.target_test_img = f.read().splitlines()
                with open(target_test_label_file, 'r') as f:
                    self.target_test_label = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.source_train_img)
        else:
            if self.phase_test:
                return len(self.target_test_img)
            else:
                return len(self.source_val_img)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.source_train_img
            label_dir = self.source_train_label
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                img_dir = self.target_test_img
                label_dir = self.target_test_label
                # label = int(label_dir[idx])
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                img_dir = self.source_val_img
                label_dir = self.source_val_label
                label = int(label_dir[idx])
                label = np.array(label)

        img = Image.open(img_dir[idx])
        # img = img.convert('RGB')
        img = img.convert('L')

        if self.transform:
            img = self.transform(img)
        if self.phase_train:
            return img,label
        else:
            return img,label,img_dir[idx]

class TargetData(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None,phase_test=False):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform

        try:
            with open(target_train_img_file, 'r') as f:
                self.target_train_img = f.read().splitlines()
            with open(target_train_label_file, 'r') as f:
                self.target_train_label = f.read().splitlines()
                
            with open(target_val_img_file, 'r') as f:
                 self.target_val_img = f.read().splitlines()
            with open(target_val_label_file, 'r') as f:
                self.target_val_label = f.read().splitlines()
            if self.phase_test:
                with open(target_test_img_file, 'r') as f:
                    self.target_test_img = f.read().splitlines()
                with open(target_test_label_file, 'r') as f:
                    self.target_test_label = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.target_train_img)
        else:
            if self.phase_test:
                return len(self.target_test_img)
            else:
                return len(self.target_val_img)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.target_train_img
            label_dir = self.target_train_label
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                img_dir = self.target_test_img
                label_dir = self.target_test_label
                # label = int(label_dir[idx])
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                img_dir = self.target_val_img
                label_dir = self.target_val_label
                label = int(label_dir[idx])
                label = np.array(label)

        img = Image.open(img_dir[idx])
        # img = img.convert('RGB')
        img = img.convert('L')

        if self.transform:
            img = self.transform(img)
        if self.phase_train:
            return img,label
        else:
            return img,label,img_dir[idx]

