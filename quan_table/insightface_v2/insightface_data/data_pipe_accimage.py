from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from torchpie.experiment import distributed

import torchvision
from torch.utils.data import DataLoader, DistributedSampler
torchvision.set_image_backend('accimage')


def get_train_accimage_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if args.data_mode in ['emore', 'iccv_emore']:
        imgs_folder = os.path.join(args.emore_folder, 'imgs')
    elif args.data_mode in ['nir_face']:
        imgs_folder = os.path.join(args.emore_folder, 'train_dataset_v3_aligned')
    else:
        assert False

    train_dataset = ImageFolder(imgs_folder, train_transform)
    class_num = train_dataset[-1][1] + 1

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              pin_memory=True, sampler=train_sampler)

    return train_loader, class_num



# if __name__ == '__main__':
#     # load_bin(path='../lfw.bin', rootdir=os.path.join('./', 'lfw'))
#     # print("load bin...")
#     lfw, lfw_issame = get_one_val_data('./', 'lfw')
#     print("4\n", lfw.shape, type(lfw[0]), lfw[0])