import os

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import bcolz
import numpy as np

torchvision.set_image_backend('accimage')

def get_cifar_dataloader(dataset, batch_size, n_threads=4, data_path='/home/dataset/', logger=None):
    """
    Get dataloader for cifar10/cifar100
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    if dataset == 'cifar10':
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        norm_mean = [0.50705882, 0.48666667, 0.44078431]
        norm_std = [0.26745098, 0.25568627, 0.27607843]
    data_root = os.path.join(data_path, 'cifar')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_root,
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=data_root,
                                       train=False,
                                       transform=val_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_root,
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=data_root,
                                        train=False,
                                        transform=val_transform)
    else:
        logger.info("invalid data set")
        assert False, "invalid data set"

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=n_threads)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=n_threads)
    return train_loader, val_loader


def get_imagenet_dataloader(dataset, batch_size, n_threads=4, data_path='/home/dataset/', logger=None):
    """
    Get dataloader for imagenet
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    dataset_path = os.path.join(data_path, "imagenet")
    traindir = os.path.join(dataset_path, "train")
    valdir = os.path.join(dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True)
    return train_loader, val_loader


def get_ms1m_dataloader(dataset, batch_size, n_threads=4,
                        data_path='/home/dataset/Face/xz_FaceRecognition/faces_ms1m-refine-v2_112x112/faces_emore/',
                        logger=None):
    """
    Get dataloader for ms1m_v2
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)
    # train_loader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if dataset in ['ms1m_v2', 'iccv_ms1m']:

        if os.path.exists(os.path.join(data_path, 'imgs_mxnet')):
            ds = ImageFolder(os.path.join(data_path, 'imgs_mxnet'), train_transform)
        else:
            ds = ImageFolder(os.path.join(data_path, 'imgs'), train_transform)

    elif dataset in ['sub_iccv_ms1m']:
        ds = ImageFolder(os.path.join(data_path, 'sample_imgs'), train_transform)
    else:
        assert False,"not support {}".format(dataset)

    # class_num = ds[-1][1] + 1
    class_num = len(ds.classes)
    train_loader = DataLoader(dataset=ds, batch_size=batch_size,
                              shuffle=True, num_workers=n_threads, pin_memory=True)

    # test_loader
    test_loader = []
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    test_loader.append({'agedb_30':agedb_30, 'agedb_30_issame':agedb_30_issame})
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    test_loader.append({'cfp_fp': cfp_fp, 'cfp_fp_issame': cfp_fp_issame})
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    test_loader.append({'lfw': lfw, 'lfw_issame': lfw_issame})

    return train_loader, test_loader, class_num



def get_webface_dataloader(dataset, batch_size, n_threads=4, data_path='', logger=None):
    """
    Get dataloader for ms1m_v2
    :param dataset: the name of the dataset
    :param batch_size: how many samples per batch to load
    :param n_threads:  how many subprocesses to use for data loading.
    :param data_path: the path of dataset
    :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)
    # train_loader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if dataset in ['sub_webface_0.1']:
        ds = ImageFolder(os.path.join(data_path, '0.1_imgs_train_data'), train_transform)
    elif dataset in ['sub_webface_0.3']:
        ds = ImageFolder(os.path.join(data_path, '0.3_imgs_train_data'), train_transform)
    else:
        assert False,"not support {}".format(dataset)

    # class_num = ds[-1][1] + 1
    class_num = len(ds.classes)
    train_loader = DataLoader(dataset=ds, batch_size=batch_size,
                              shuffle=True, num_workers=n_threads, pin_memory=True)

    # test_loader
    val_loader = []
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    val_loader.append({'agedb_30':agedb_30, 'agedb_30_issame':agedb_30_issame})
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    val_loader.append({'cfp_fp': cfp_fp, 'cfp_fp_issame': cfp_fp_issame})
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    val_loader.append({'lfw': lfw, 'lfw_issame': lfw_issame})

    verification, verification_issame = get_val_pair(data_path, '160000_imgs_verfication_data')
    val_loader.append({'verification': verification, 'verification_issame': verification_issame})

    return train_loader, val_loader, class_num


def get_webface_testing_dataloader(dataset, batch_size, n_threads=4, data_path='', logger=None):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    if dataset in ['sub_webface_0.1']:
        test_imgs_folder = os.path.join(data_path, '0.1_imgs_test_data')
    elif dataset in ['sub_webface_0.3']:
        test_imgs_folder = os.path.join(data_path, '0.3_imgs_test_data')
    else:
        assert False,"no support {}".format(dataset)

    test_dataset = ImageFolder(test_imgs_folder, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_threads, pin_memory=True)
    return test_loader


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
    return carray, issame


def get_quantization_lfw_dataloader(batch_size, n_threads=4,
                                    data_path='/home/dataset/xz_datasets/LFW/lfw_112x112_13233', logger=None):

    print("|===>Get datalaoder for quantization lfw calibration")
    # train_loader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0

    ds = ImageFolder(data_path, train_transform)
    # class_num = ds[-1][1] + 1
    class_num = len(ds.classes)
    train_loader = DataLoader(dataset=ds, batch_size=batch_size,
                              shuffle=False, num_workers=n_threads, pin_memory=True)

    return train_loader, class_num



# def get_nir_face_dataloader(dataset, batch_size, n_threads=4,
#                             data_path='/home/dataset/Face/xz_FaceRecognition/faces_ms1m-refine-v2_112x112/faces_emore/',
#                             logger=None):
#     """
#     Get dataloader for ms1m_v2
#     :param dataset: the name of the dataset
#     :param batch_size: how many samples per batch to load
#     :param n_threads:  how many subprocesses to use for data loading.
#     :param data_path: the path of dataset
#     :param logger: logger for logging
#     """
#
#     logger.info("|===>Get datalaoder for " + dataset)
#     # train_loader
#     train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
#     ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0
#
#     if dataset in ['nir_face_1224']:
#         ds = ImageFolder(os.path.join(data_path, 'nir_jidian_train_1224_190727'), train_transform)
#     else:
#         assert False,"not support {}".format(dataset)
#
#     # class_num = ds[-1][1] + 1
#     class_num = len(ds.classes)
#     train_loader = DataLoader(dataset=ds, batch_size=batch_size,
#                               shuffle=True, num_workers=n_threads, pin_memory=True)
#
#     # test_loader
#     test_loader = []
#     agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
#     test_loader.append({'agedb_30':agedb_30, 'agedb_30_issame':agedb_30_issame})
#     cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
#     test_loader.append({'cfp_fp': cfp_fp, 'cfp_fp_issame': cfp_fp_issame})
#     lfw, lfw_issame = get_val_pair(data_path, 'lfw')
#     test_loader.append({'lfw': lfw, 'lfw_issame': lfw_issame})
#
#     return train_loader, test_loader, class_num
