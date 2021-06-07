#加入了新数据增强方法,在五点增强的基础上对对齐后的图像进行(随机地)轻微地旋转
#by lidaiyuan 2020-09-10

from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import random

import math
import cv2
import torchvision
import torch

from skimage.transform import estimate_transform, warp
import random
import math

def estimate_rotation_trans(is_aug = True,angle_range = [-5,5],landmarks = None):
    
    DST_PTS = np.float32([[(30.2946 + 8.0) , 51.6963],
	  [(65.5318 + 8.0) , 51.5014],
	  [(48.0252 + 8.0) , 71.7366],
	  [(33.5493 + 8.0) , 92.3655],
	  [(62.7299 + 8.0) , 92.2041]])
    
    assert(landmarks is not None)
    
    is_aug = (random.randint(0,10) > 5) and is_aug #随机数据增强
        
    """ 如果使用数据增强 """
    if is_aug:
       theta = 1 + random.randint(angle_range[0], angle_range[1])
       cos_a = math.cos((math.pi / 180) * theta)
       sin_a = math.sin((math.pi / 180) * theta)
       rscale = 1 + random.random() / 20
       rotate_matrix = np.array([[cos_a * rscale, -sin_a * rscale, 0],[sin_a * rscale, cos_a * rscale, 0],[0,0,1]])
       SRC_PTS = np.reshape(landmarks,[5,2])
       tform = estimate_transform('affine', SRC_PTS, DST_PTS)
       rotated_matrix = np.matmul(rotate_matrix,tform.params)
       return np.linalg.inv(rotated_matrix)
    else:
       SRC_PTS = np.reshape(landmarks,[5,2])
       tform = estimate_transform('affine', SRC_PTS, DST_PTS)
       return tform.inverse



class CASIA(Dataset):
    def __init__(self,data_flag = None, transform=None, phase_train=True, data_dir=None,phase_test=False):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.train_landmarks = []        
        self.val_landmarks = []
        self.test_landmarks = []
        
        #for train
        #for train
        train_file = os.getcwd() +'/data/train_file_list/%s_train.txt'%data_flag
        label_train_file = os.getcwd() + '/data/train_file_list/%s_train_label.txt'%data_flag
        
        # for val
        val_file = os.getcwd() +'/data/train_file_list/%s_val.txt'%data_flag
        label_val_file = os.getcwd() + '/data/train_file_list/%s_val_label.txt'%data_flag
                
        # for test
        test_file = os.getcwd() +'/data/train_file_list/%s_val.txt'%data_flag
        label_test_file = os.getcwd() + '/data/train_file_list/%s_val_label.txt'%data_flag
        count = 0
        try:
            with open(train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
                # print(self.depth_dir_train )
                # print(len(self.depth_dir_train))
            with open(label_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            #load landmarks
            for l in self.depth_dir_train:
              l_path = l.replace('jpg','npy')
              if os.path.exists(l_path):
                #   print(l_path)
                  self.train_landmarks.append(np.load(l_path))
            print(len(self.depth_dir_train),len(self.train_landmarks))
                 
            
            print(count)
            with open(val_file, 'r') as f:
                 self.depth_dir_val = f.read().splitlines()
            with open(label_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            #load landmarks
            for l in self.depth_dir_val:
              l_path = l.replace('jpg','npy')
              if os.path.exists(l_path):
                  self.val_landmarks.append(np.load(l_path))
                
            if self.phase_test:
                with open(test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_test_file, 'r') as f:
                    self.label_dir_test = f.read().splitlines()
            #load landmarks
            for l in self.depth_dir_val:
              l_path = l.replace('jpg','npy')
              if os.path.exists(l_path):
                  self.test_landmarks.append(np.load(l_path))
                    
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
        
        # print(depth_dir[idx])
        depth = Image.open(depth_dir[idx])
        # depth = depth.convert('RGB')
        depth = depth.convert('L')
        
        
        #data augment
        ''' 随机旋转缩放 '''
        
        if self.phase_train:
            landmarks = self.train_landmarks[idx]
            t = estimate_rotation_trans(landmarks = landmarks)
            depth = warp(np.array(depth), t, output_shape=(112, 112))
        else:     
            landmarks = self.val_landmarks[idx]        
            t = estimate_rotation_trans(is_aug = False,landmarks = landmarks)
            depth = warp(np.array(depth), t, output_shape=(112, 112))
        
        #TODO:文件加MASK
        
        depth = Image.fromarray(np.uint8(depth * 255))
        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]

