from pathlib import Path
# from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
# from torchvision.datasets import ImageFolder
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


def load_bin(path, rootdir, image_size=[112, 112]):
    test_transform = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ]) # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()   # imdecode: three channel color output + RGB formatted output
        # plt.subplot(121)
        # plt.imshow(img)
        # print("1\n", img.shape, type(img),  img.transpose(2, 0, 1))
        # # print("2\n", img.astype(np.uint8))
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = test_transform(img)
        # print("3\n", data[i, ...].shape, type(data[i, ...]), data[i, ...])
        # plt.show()
        # break
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(rootdir + '_list', np.array(issame_list))
    return data, issame_list


def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    train_indx_path = os.path.join(rec_path, 'train.idx')
    train_rec_path = os.path.join(rec_path, 'train.rec')
    imgrec = mx.recordio.MXIndexedRecordIO(train_indx_path, train_rec_path, 'r')

    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    print("max_idx", str(max_idx))

    for idx in tqdm(range(1, max_idx)):
        # xz codes
        img_info = imgrec.read_idx(idx)
        header, s = mx.recordio.unpack(img_info)
        # print("header={}".format(header))
        img = mx.image.imdecode(s).asnumpy()      # imdecode: three channel color output + RGB formatted output
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = int(header.label)
        label_path = os.path.join(save_path, str(label))
        if not os.path.isdir(label_path):
            os.mkdir(label_path)
        cv2.imwrite(os.path.join(label_path, '{}.jpg'.format(idx)), img)
        # if idx == 2000:
        #     assert False


# if __name__ == '__main__':
#     # load_bin(path='../lfw.bin', rootdir=os.path.join('./', 'lfw'))
#     # print("load bin...")
#     lfw, lfw_issame = get_one_val_data('./', 'lfw')
#     print("4\n", lfw.shape, type(lfw[0]), lfw[0])