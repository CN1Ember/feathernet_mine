from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import math
import cv2
import torchvision
import torch
import random
import tqdm

class CASIA(Dataset):
    def __init__(self,data_flag = None, transform=None, phase_train=True, data_dir=None,phase_test=False,add_mask = True):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.add_mask = add_mask
        self.mask_np = None

        #for train
        train_file = os.getcwd() +'/data/train_file_list/depth_list/depth_train_minmax_mask_1021.txt'
        label_train_file = os.getcwd() + '/data/train_file_list/depth_list/label_train_minmax_mask_1021.txt'
        
        # for val
        val_file = os.getcwd() +'/data/train_file_list/depth_list/depth_test_minmax_mask.txt'
        label_val_file = os.getcwd() + '/data/train_file_list/depth_list/label_test_minmax_mask.txt'
        
        # for test
        test_file = os.getcwd() +'/data/train_file_list/depth_list/depth_test_minmax_mask.txt'
        label_test_file = os.getcwd() + '/data/train_file_list/depth_list/label_test_minmax_mask.txt'

        # print("geting data from ",train_file)

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
        
        
        #get path for new img
        # print(depth_dir[idx])
        # rindex = depth_dir[idx].rfind('/',0,len(depth_dir[idx]))
        # new_head = depth_dir[idx][:62] + '_reverse'

        print(len('/mnt/cephfs/dataset/face_anti_spoofing_lock/fas_depth_datasets'))

        new_head = '/mnt/cephfs/dataset/FAS_depth'
        new_depth_dir = new_head + depth_dir[idx][62:]
        rindex = new_depth_dir.rfind('/',0,len(new_depth_dir))
        new_depth_dir_reverse = new_depth_dir[:rindex] +'_reverse' + new_depth_dir[rindex:]

        print("path for img to save is ",new_depth_dir_reverse)


        if not os.path.exists(new_depth_dir[:rindex] +'_reverse/'):
            print("path not exist, now mkdir")
            os.makedirs(new_depth_dir[:rindex] +'_reverse/')

        #保存深度值反转后的深度图
        depth = depth.point(lambda x : 255-x if x>0 else 0)
        
        uniq,count = np.unique(depth,return_counts = True)
        # print(uniq)
        # print(count)
        count[0]=0
        max_index = np.argmax(count)
        max_num_intensity = uniq[max_index]
        print("the intensity with max num of pixels is ",max_num_intensity)

        depth = depth.point(lambda x : x if x<max_num_intensity+15 else 0) #把最高值右边15以上的截断


        depth_np = np.array(depth)
        cv2.imwrite(new_depth_dir_reverse,depth_np.astype(np.uint8))

        '''transform'''
        if self.transform:
            depth = self.transform(depth)

        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]

if __name__ == "__main__":


    #1 在getitem里修改，保存地址
        # 1 修改保存路径 
        # 2 if os.exist
    #2 遍历每张图过一个item
    #3 直接修改txt用替换

    train_dataset = CASIA(
         phase_train=True,phase_test=False, add_mask=False)

    train_dataset.__getitem__(55)
    # print("length of train = ",len(train_dataset))
    for i in  range(len(train_dataset)) :
        train_dataset.__getitem__(i)

    test_dataset = CASIA(
         phase_train=False,phase_test=True, add_mask=False)

    # print("length of test = ",len(test_dataset))
    
    for i in range(len(test_dataset)) :
        test_dataset.__getitem__(i)





    # check if transfer ok
    # check_test_file = os.getcwd() +'/data/train_file_list/depth_list/depth_test_minmax_mask_reverse.txt'
    # # depth_dir = './img_process/depth_data/1/align/00000062.png'

    # with open(check_test_file, 'r') as f:
    #      depth_dir = f.read().splitlines()
    # print(depth_dir[20])
    # depth = Image.open(depth_dir[20])
    # print(np.unique(depth))
    # depth,count = np.unique(depth,return_counts = True)
    # print(count)
