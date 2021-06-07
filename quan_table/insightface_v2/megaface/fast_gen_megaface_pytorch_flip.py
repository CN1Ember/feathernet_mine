from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import struct
import sys
import numpy as np
import cv2
# from easydict import EasyDict as edict
import torch
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA

# from models import MobileFaceNet
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
# import face_preprocess
# import facenet
# import lfw
# from caffe.proto import caffe_pb2
import torch.backends.cudnn as cudnn
from torchvision import transforms
import pickle

from insightface_v2.model.models import MobileFaceNet, resnet34
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v3_width import Mobilefacenetv2_v3_width
from insightface_v2.model.mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_v2.model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm

from insightface_v2.model.insightface_resnet import LResNet34E_IR
from insightface_v2.model.insightface_resnet_pruned import pruned_LResNet34E_IR


def de_preprocess(tensor):
    return tensor * 0.501960784 + 0.5

hflip = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.functional.hflip,
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

def parse_lst_line(line):
    vec = line.strip().split("\t")
    assert len(vec) >= 3
    aligned = int(vec[0])  # is or not aligned
    image_path = vec[1]    # aligned image path
    label = int(vec[2])    # new define image label(preson+ID-->ID) in align_facescrub.py
    bbox = None
    landmark = None
    # print(vec)
    if len(vec) > 3:
        bbox = np.zeros((4,), dtype=np.int32)
        for i in range(3, 7):
            bbox[i - 3] = int(vec[i])
        landmark = None
        if len(vec) > 7:
            _l = []
            for i in range(7, 17):
                _l.append(float(vec[i]))
            landmark = np.array(_l).reshape((2, 5)).T
    # print(aligned)
    return image_path, label, bbox, landmark, aligned

def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_feature(image_path, model, image_shape, use_flip=True):
    img = read_image(image_path)
    # print(img.shape)
    if img is None:
        print('parse image', image_path, 'error')
        return None
    assert img.shape == (image_shape[1], image_shape[2], image_shape[0])

    v_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape((1, 1, 3))
    img = img.astype(np.float32) - v_mean
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    img = torch.tensor(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
    # print('图片大小：', img.size())
    F = model(img.cuda())

    print('use_flip={}'.format(use_flip))
    if use_flip:
        fliped_img = hflip_batch(img)
        fliped_F = model(fliped_img.cuda())
        F = F + fliped_F

    F = F.data.cpu().numpy().flatten()
    _norm = np.linalg.norm(F)
    F /= _norm
    # print(F.shape)
    return F

def get_image_path(image_path, dataser_path, split_str):
    _, path = image_path.split(split_str)
    return os.path.join(dataser_path, path)

def get_batch_feature(img_path_list, model, image_shape, use_flip=True):
    batch_img = np.zeros([len(img_path_list), image_shape[0], image_shape[1], image_shape[2]])
    # print("batch_img shape={}".format(batch_img.shape))
    for i in range(len(img_path_list)):
        img = read_image(img_path_list[i])
        # print(img.shape)
        if img is None:
            print('parse image', img_path_list[i], 'error')
            return None
        assert img.shape == (image_shape[1], image_shape[2], image_shape[0])
        v_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape((1, 1, 3))
        img = img.astype(np.float32) - v_mean
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        batch_img[i, ...] = img

    batch_img = torch.tensor(batch_img).float()
    F = model(batch_img.cuda())

    print('use_flip={}'.format(use_flip))
    if use_flip:
        fliped_batch_img = hflip_batch(batch_img)
        fliped_F = model(fliped_batch_img.cuda())
        F = F + fliped_F

    F = F.data.cpu().numpy()
    F = sklearn.preprocessing.normalize(F)
    # print(F.shape)
    return F

def process_batch_feature(args, image_path_list, megaface_out, dataset_path, model, image_shape, use_flip):
    img_path_list = []
    out_dir_list = []
    b_list = []
    count = 0
    for i in range(len(image_path_list)):
        _path = image_path_list[i].split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        out_dir_list.append(out_dir)
        b_list.append(b)

        image_path = get_image_path(image_path_list[i], dataset_path, "MegaFace/")
        img_path_list.append(image_path)

    feature = get_batch_feature(img_path_list, model, image_shape, use_flip)
    # print("img_path_list len={}, out_dir_list len={}, b_list len={}".format(len(img_path_list), len(out_dir_list), len(b_list)))

    for i in range(len(img_path_list)):
        # print("i={}, {}".format(i, img_path_list[i]))
        if not os.path.exists(out_dir_list[i]):
            os.makedirs(out_dir_list[i])
        out_path = os.path.join(out_dir_list[i], b_list[i] + "_%s_%dx%d.bin" % (args.algo, image_shape[1], image_shape[2]))
        # print(out_path)
        write_bin(out_path, feature[i].flatten())
        count += 1
    return count


def main(args):
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    image_shape = [int(x) for x in args.image_size.split(',')]

    # 186
    megaface_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/megaface_112x112/lst"
    facescrub_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2/small_lst"
    dataset_path = "/home/dataset/xz_datasets/Megaface/Mageface_aligned"

    # ms1mv2: old_mobilefacenet_baseline_flip
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_v1_model/' \
    #                'pytorch_mobilefacenet_epoch34_feature-result_fliped/fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_v1_model/' \
    #                 'pytorch_mobilefacenet_epoch34_feature-result_fliped/fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/'
    #                                 'pytorch_mobilefacenet_v1_model/checkpoint_34.pth')

    # ms1mv2: old_mobilefacenet_p0.25_flip
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/' \
    #                'p0.25_mobilefacenet_v1_lr0.01_checkpoint22_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/' \
    #                 'p0.25_mobilefacenet_v1_lr0.01_checkpoint22_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/'
    #                                 'pytorch_pruned0.25_mobilefacenet_v1_better/'
    #                                 'p0.25_mobilefacenet_v1_without_fc_checkpoint_022.pth')

    # # ms1mv2: old_mobilefacenet_p0.5_flip
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.5_mobilefacenet_v1/' \
    #                'p0.5_mobilefacenet_v1_lr0.01_checkpoint25_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.5_mobilefacenet_v1/' \
    #                 'p0.5_mobilefacenet_v1_lr0.01_checkpoint25_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/'
    #                                 'pytorch_pruned0.5_mobilefacenet_v1/checkpoint_025.pth')

    # ms1mv2: old_r34_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/pytorch_resnet34_epoch48_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/pytorch_resnet34_epoch48_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/'
    #                                 'insightface_r34/insightface_r34_with_arcface_v2_epoch48.pth')

    # ms1mv2: old_r34_p0.3_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                'pytorch_pruned0.3_r34_lr0.01_checkpoint25_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                 'pytorch_pruned0.3_r34_lr0.01_checkpoint25_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/'
    #                                 'pytorch_pruned0.3_r34_lr0.01_checkpoint25_feature-result/checkpoint_025.pth')
    #
    # # # ms1mv2: old_r34_p0.5_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                'pytorch_pruned0.5_r34_lr0.01_checkpoint27_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                 'pytorch_pruned0.5_r34_lr0.01_checkpoint27_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/'
    #                                 'pytorch_pruned0.5_r34_lr0.01_checkpoint27_feature-result/checkpoint_027.pth')

    # ms1mv2: old_r34_p0.25_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                'pytorch_pruned0.25_r34_lr0.01_checkpoint26_feature-result_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/' \
    #                 'pytorch_pruned0.25_r34_lr0.01_checkpoint26_feature-result_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/'
    #                                 'pytorch_pruned0.25_r34_lr0.01_checkpoint26_feature-result_fliped/checkpoint_026.pth')

    # iccv_ms1m_mobilefacenet_baseline_two_stage_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage_checkpoint31/feature_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage_checkpoint31/feature_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                                 '2stage/dim128/stage2/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs200_e36_lr0.100_'
    #                                 'step[15, 25, 31]_kaiming_init_auxnet_arcface_new_op_stage2_20190726/check_point/checkpoint_031.pth')

    # iccv_emore: new_mobilefacenet_p0.5_without_fc_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.5_without_p_fc_checkpoint35/feature_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.5_without_p_fc_checkpoint35/feature_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                                 '2stage/dim128/finetune/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs512_e36_lr0.010_'
    #                                 'step[15, 25, 31]_p0.5_bs512_cosine_finetune_without_fc_20190809/check_point/checkpoint_035.pth')

    # iccv: new_mobilefacenet_p0.25_without_fc_fliped
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.25_without_p_fc_checkpoint34/feature_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.25_without_p_fc_checkpoint34/feature_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/'
    #                                 'finetune/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs200_e36_lr0.010_step[15, 25, 31]'
    #                                 '_p0.25_bs512_cosine_finetune_20190820/check_point/checkpoint_034.pth')


    # guoyong: mobilefacenet_2stage_noauxnet
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/guoyong/pytorch_mobilefacenet_noauxnet_checkpoint35_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/guoyong/pytorch_mobilefacenet_noauxnet_checkpoint35_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/auxnet_non_auxnet/stage2/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs200_e36_lr0.100_step[15, 25, 31]_kaiming_init_arcface_new_op_nowarmup_nolabelsmooth_resume2_20190831/check_point/checkpoint_035.pth')

    # guoyong: mobilefacenet_2stage_auxnet
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/guoyong/pytorch_mobilefacenet_auxnet_checkpoint35_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/guoyong/pytorch_mobilefacenet_auxnet_checkpoint35_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/auxnet_non_auxnet/stage2/log_aux_mobilefacenetv2_baseline_128_softmax-arcface_iccv_emore_bs384_e36_lr0.100_step[15, 25, 31]_kaiming_init_auxnet_n2_arcface_new_op_nowarmup_nolabelsmooth_stage2_resume_20190829/check_point/checkpoint_035.pth')


    # iccv_emore: new_mobilefacenet_p0.5_without_fc_fliped + s=128_finetune
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/' \
    #                'pytorch_mobilefacenet_p0.5_without_p_fc_checkpoint35_s128_second_finetune_fliped/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/' \
    #                 'pytorch_mobilefacenet_p0.5_without_p_fc_checkpoint35_s128_second_finetune_fliped/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/finetune/s128_second_finetune/log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_iccv_emore_bs512_e24_lr0.001_step[]_bs512_cosine_88.26_second_s128_finetune_20190829/check_point/checkpoint_023.pth')

    # ms1mv2: mobilefacenet_prun_adapt
    megaface_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_prun_adapt/' \
                   'pytorch_mobilefacenet_without_p_fc_adapt_checkpoint22_s128_fliped/' \
                   'fp_megaface_112x112_norm_features'
    facescrub_out = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_prun_adapt/' \
                   'pytorch_mobilefacenet_without_p_fc_adapt_checkpoint22_s128_fliped/' \
                    'fp_facescrub_112x112_norm_features'
    check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_prun_adapt/checkpoint_022.pth', map_location=torch.device('cpu'))

    print("megaface_out = ", megaface_out)
    print("facescrub_out = ", facescrub_out)

    # model = MobileFaceNet()    # old: mobilefacenet_v1
    # model = resnet34()         # r34
    # model = pruned_Mobilefacenet(pruning_rate=0.25)  # old
    # model = pruned_Mobilefacenet(pruning_rate=0.5)  # old
    # mobilefacenet_adapt
    f = open('/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_mobilefacenet_prun_adapt/model.txt', 'rb')
    model = pickle.load(f)

    # model = LResNet34E_IR()
    # model = pruned_LResNet34E_IR(pruning_rate=0.3)
    # model = pruned_LResNet34E_IR(pruning_rate=0.5)
    # model = pruned_LResNet34E_IR(pruning_rate=0.25)

    # new_model
    # model = Mobilefacenetv2_v3_width(embedding_size=512, width_mult=1.315)
    # model = Mobilefacenetv2_v3_width(embedding_size=128, width_mult=1.0)
    # model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)  # liujing
    # model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.25)  # liujing

    model_state = check_point_params['model']
    print(model)
    model.load_state_dict(model_state)
    print("model load success !!!")
    # torch.save(model_state, '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/'
    #                         'pytorch_pruned0.25_r34_lr0.01_checkpoint26_feature-result_fliped/'
    #                         'p0.25_r34_lr0.01_checkpoint26.pth')
    model = model.cuda()
    model.eval()

    if args.skip == 0:
        i = 0
        succ = 0
        print("get facescrub features start!")
        for line in open(facescrub_lst, 'r'):
            if i % 1000 == 0:
                print("writing facescrub", i, succ)

            print('i=', str(i))
            image_path, label, bbox, landmark, aligned = parse_lst_line(line)
            _path = image_path.split('/')
            a, b = _path[-2], _path[-1]

            out_dir = os.path.join(facescrub_out, a)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            image_path = get_image_path(image_path, dataset_path, "MegaFace/")
            print(image_path)
            feature = get_feature(image_path, model, image_shape, args.use_flip)

            out_path = os.path.join(out_dir, b + "_%s_%dx%d.bin" % (args.algo, image_shape[1], image_shape[2]))
            write_bin(out_path, feature)
            succ += 1
            i += 1
        print('facescrub finish!', i, succ)

    # return
    if args.mf == 0:
        return
    i = 0
    succ = 0
    batch_count = 0
    image_path_list = []
    print("get megaface features start!")
    for line in open(megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing megaface", i, succ)
        print('i={}, succ={}'.format(i, succ))

        image_path, label, bbox, landmark, aligned = parse_lst_line(line)
        image_path_list.append(image_path)

        if (i + 1) % args.batch_size == 0:
            batch_count += 1
            print("batch count={}".format(batch_count))
            count = process_batch_feature(args, image_path_list, megaface_out,
                                          dataset_path, model, image_shape, args.use_flip)
            succ += count
            image_path_list = []
            # continue
            # break
        # succ += 1
        i += 1
        # if batch_count == 2:
        #     break

    # for last batch
    if len(image_path_list) != 0:
        count = process_batch_feature(args, image_path_list, megaface_out, dataset_path,
                                      model, image_shape, args.use_flip)
        succ += count
    print('mf stat', i, succ)
    print(args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=64)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--skip', type=int, help='', default=0)
    parser.add_argument('--mf', type=int, help='', default=1)

    # parser.add_argument('--algo', type=str, help='', default='mobilefacenetNcnn')
    parser.add_argument('--algo', type=str, help='', default='mobilefacenetPytorch')
    # parser.add_argument('--algo', type=str, help='', default='r34Pytorch')

    parser.add_argument('--gpu', type=str, help='gpu', default='1')
    parser.add_argument('--use_flip', type=bool, help='use_flip', default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
