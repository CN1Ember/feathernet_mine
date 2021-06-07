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


# origin
def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs_mxnet')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    train_indx_path = os.path.join(rec_path, 'train.idx')
    train_rec_path = os.path.join(rec_path, 'train.rec')
    imgrec = mx.recordio.MXIndexedRecordIO(train_indx_path, train_rec_path, 'r')

    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    # print("max_idx", str(max_idx))

    for idx in tqdm(range(1, max_idx)):
        # xz codes
        img_info = imgrec.read_idx(idx)
        header, s = mx.recordio.unpack(img_info)
        # print("header={}".format(header))

        label = int(header.label[0])
        label_path = os.path.join(save_path, str(label))
        if not os.path.isdir(label_path):
            os.mkdir(label_path)

        img_path = os.path.join(label_path, '{}.jpg'.format(idx))
        with open(img_path, 'wb') as f:
            f.write(s)
        f.close()

        # if idx == 100:
        #     assert False


# def load_mx_rec(rec_path):
#     save_path = os.path.join(rec_path, 'imgs_test')
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
#     train_indx_path = os.path.join(rec_path, 'train.idx')
#     train_rec_path = os.path.join(rec_path, 'train.rec')
#     imgrec = mx.recordio.MXIndexedRecordIO(train_indx_path, train_rec_path, 'r')
#
#     img_info = imgrec.read_idx(0)
#     header, _ = mx.recordio.unpack(img_info)
#     max_idx = int(header.label[0])
#     print("max_idx", str(max_idx))
#     txt_path = './log.txt'
#
#     for idx in range(1, max_idx):
#         # xz codes
#         img_info = imgrec.read_idx(idx)
#         header, s = mx.recordio.unpack(img_info)
#         print("header={}".format(header))
#         label = int(header.label[0])
#
#         # img = mx.image.imdecode(s).asnumpy()  # imdecode: three channel color output + RGB formatted output
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#         label_path = os.path.join(save_path, str(label))
#         origin_img_path = os.path.join(label_path, '{}.jpg'.format(idx))
#         print("origin_img_path={}".format(origin_img_path))
#
#         img1 = mx.image.imdecode(s).asnumpy()
#         # print(img1)
#         # print("img1 type={}, {}".format(type(img1), type(img1[0])))
#         img1 = img1.astype(np.float32)
#         # print(img1.shape)
#
#         img_path = os.path.join("/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/imgs",
#                                 str(label), '{}.jpg'.format(idx))
#         print("img_path={}".format(img_path))
#         img2 = cv2.imread(img_path)
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#         # print("img2 type={}, {}".format(type(img2), type(img2[0])))
#         img2 = img2.astype(np.float32)
#
#         delta = img1 - img2
#         sum_data = np.sum(np.abs(delta))
#         mean_data = np.mean(np.abs(delta))
#
#         # if sum_data > max_sample:
#         #     max_sample = sum_data
#         #
#         # if sum_data < min_sample:
#         #     min_sample = sum_data
#         #
#         # if mean_data > max_mean_sample:
#         #     max_mean_sample = mean_data
#         #
#         # if mean_data < min_mean_sample:
#         #     min_mean_sample = mean_data
#
#         log_str = "img_path={}, sum_data={}, mean_data={}\n".format(img_path, sum_data, mean_data)
#         with open(txt_path, 'a+') as f:
#             f.write(log_str)
#         f.close()
#
#         print(log_str)
#
#         if idx == 100000:
#             assert False

# if __name__ == '__main__':
#     # load_bin(path='../lfw.bin', rootdir=os.path.join('./', 'lfw'))
#     # print("load bin...")
#     lfw, lfw_issame = get_one_val_data('./', 'lfw')
#     print("4\n", lfw.shape, type(lfw[0]), lfw[0])