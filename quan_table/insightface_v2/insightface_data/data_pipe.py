import torchvision
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
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

torchvision.set_image_backend('accimage')
from torch.utils.data import Dataset, DataLoader
import os
# from mxnet import recordio
from PIL import Image
from io import BytesIO
import imghdr

# import accimage
from torch.utils.data import DataLoader, DistributedSampler


# def opencv_loader(path):
#     with open(path, 'rb') as f:
#         img = cv2.imread(f.name, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#
# # def default_loader(path):
# #     return opencv_loader(path)
#
# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']
#
# class ICCVImageFolder(DatasetFolder):
#     def __init__(self, root, transform=None, target_transform=None, loader=opencv_loader):
#         super(ICCVImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
#                                               transform=transform,
#                                               target_transform=target_transform)
#         self.imgs = self.samples


def get_train_dataset(imgs_folder):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0
    ds = ImageFolder(imgs_folder, train_transform)
    # ds = ICCVImageFolder(imgs_folder, train_transform)

    # class_num = ds[-1][1] + 1
    class_num = len(ds.classes)
    return ds, class_num


def get_train_loader(args):
    if args.data_mode in ['emore', 'iccv_emore']:
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, 'imgs_mxnet'))

    elif args.data_mode in ['sub_webface_0.1']:
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, '0.1_imgs_train_data'))
    elif args.data_mode in ['sub_webface_0.3']:
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, '0.3_imgs_train_data'))

    elif args.data_mode in ['sub_iccv_emore']:
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, 'sample_imgs'))
    elif args.data_mode in ['sub_iccv_1000']:
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, 'sample_imgs_1000_new'))
    elif args.data_mode == 'nir_face':
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, 'train_dataset_v3_aligned'))
    elif args.data_mode == 'nir_face_1224':
        train_dataset, class_num = get_train_dataset(os.path.join(args.emore_folder, 'nir_jidian_train_1224_190727'))
    else:
        assert False, 'not support {}'.format(args.data_mode)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, class_num


def get_soft_finetune_train_loader(args):

    rgb_train_dataset, rgb_class_num = get_train_dataset(os.path.join(args.rgb_emore_folder, 'imgs_mxnet'))

    if args.nir_data_mode in ['nir_face_1224']:
        nir_train_dataset, nir_class_num = get_train_dataset(os.path.join(args.nir_emore_folder, 'jidian_nir_1224'))
    elif args.nir_data_mode in ['nir_face_1321_align']:
        nir_train_dataset, nir_class_num = get_train_dataset(os.path.join(args.nir_emore_folder, 'jidian_nir_1321_align'))
    elif args.nir_data_mode in ['nir_face_1420_align']:
        nir_train_dataset, nir_class_num = get_train_dataset(os.path.join(args.nir_emore_folder, 'jidian_nir_1420_align'))
    else:
        assert False, 'not support {}'.format(args.data_mode)


    rgb_train_loader = DataLoader(dataset=rgb_train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    nir_train_loader = DataLoader(dataset=nir_train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    return rgb_train_loader, rgb_class_num, nir_train_loader, nir_class_num


def get_test_loader(args):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if args.data_mode in ['sub_webface_0.1']:
        test_imgs_folder = os.path.join(args.emore_folder, '0.1_imgs_test_data')
    elif args.data_mode in ['sub_webface_0.3']:
        test_imgs_folder = os.path.join(args.emore_folder, '0.3_imgs_test_data')
    else:
        assert False, "no support {}".format(args.data_mode)

    test_dataset = ImageFolder(test_imgs_folder, test_transform)
    # test_dataset = ICCVImageFolder(test_imgs_folder, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return test_loader


def load_bin(path, rootdir, image_size=[112, 112]):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if not os.path.isdir(rootdir):
        os.mkdir(rootdir)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()  # imdecode: three channel color output + RGB formatted output
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


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return carray, issame


def get_all_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    # verfication, verfication_issame = get_val_pair(data_path, '160000_imgs_verfication_data')
    # return agedb_30, cfp_fp, lfw, verfication, agedb_30_issame, cfp_fp_issame, lfw_issame, verfication_issame
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


def get_one_val_data(data_path, data_name):
    data, data_issame = get_val_pair(data_path, data_name)
    return data, data_issame


# def load_mx_rec(rec_path):
#     save_path = os.path.join(rec_path, 'imgs')
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
#
#     for idx in tqdm(range(1, max_idx)):
#         # xz codes
#         img_info = imgrec.read_idx(idx)
#         header, s = mx.recordio.unpack(img_info)
#         # print("header={}".format(header))
#
#         img = mx.image.imdecode(s).asnumpy()  # imdecode: three channel color output + RGB formatted output
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         label = int(header.label)
#         label_path = os.path.join(save_path, str(label))
#         if not os.path.isdir(label_path):
#             os.mkdir(label_path)
#         cv2.imwrite(os.path.join(label_path, '{}.jpg'.format(idx)), img)
#         # if idx == 10:
#         #     assert False


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
    # print("max_idx", str(max_idx))

    for idx in tqdm(range(1, max_idx)):
        # xz codes
        img_info = imgrec.read_idx(idx)
        header, s = mx.recordio.unpack(img_info)
        # print("header={}".format(header))

        label = int(header.label)
        label_path = os.path.join(save_path, str(label))
        if not os.path.isdir(label_path):
            os.mkdir(label_path)

        img_path = os.path.join(label_path, '{}.jpg'.format(idx))
        with open(img_path, 'wb') as f:
            f.write(s)
        f.close()


# if __name__ == "__main__":
#     from torch import nn
#     import time
#
#     # dataset = RecordDataset('/mnt/ssd/Datasets/faces/ms1m-retinaface-t1/train.rec')
#
#     loader = DataLoader(dataset, batch_size=128,
#                         num_workers=8, shuffle=True, pin_memory=True)
#     conv = nn.Conv2d(3, 1, 1)
#
#     start = time.perf_counter()
#
#     for image, label in loader:
#         # print(label)
#
#         end = time.perf_counter()
#         duration = end - start
#         print(
#             f'data time: {duration}, images: {label.shape[0] / duration} images/s')
#
#         out = conv(image)
#
#         start = time.perf_counter()
#         # pass
