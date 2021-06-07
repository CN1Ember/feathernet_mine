from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
sys.path.append('/home/xiezheng/lidaiyuan/insightface_dcp')
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

from insightface_v2.model.models import MobileFaceNet, resnet34
# from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v3_width import Mobilefacenetv2_v3_width
# from insightface_v2.model.mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_v2.model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])

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

def get_feature(image_path, model, image_shape):
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
    F = F.data.cpu().numpy().flatten()
    _norm = np.linalg.norm(F)
    F /= _norm
    # print(F.shape)
    return F

def get_image_path(image_path, dataser_path, split_str):
    _, path = image_path.split(split_str)
    return os.path.join(dataser_path, path)

def get_batch_feature(img_path_list, model, image_shape):
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
    F = F.data.cpu().numpy()
    F = sklearn.preprocessing.normalize(F)
    # print(F.shape)
    return F

def process_batch_feature(args, image_path_list, megaface_out, dataset_path, model, image_shape):
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

    feature = get_batch_feature(img_path_list, model, image_shape)
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

    megaface_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/megaface_112x112/lst"

    # 186
    facescrub_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2/small_lst"
    dataset_path = "/home/dataset/xz_datasets/Megaface/Mageface_aligned"


    # mobilefacenet_width
    # megaface_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet_width/pytorch_mobilefacenet_width_with_auxnet/fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet_width/pytorch_mobilefacenet_width_with_auxnet/fp_facescrub_112x112_norm_features'
    # without_auxnet
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/auxnet/mobilefacenet/'
    #                                 'mobilenfacenetv2_width_4e-5_stagetwo_loadfc_magin64/check_point/checkpoint_34.pth')
    # with_auxnet
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/auxnet/mobilefacenet/'
    #                                 'log_aux_mobilefacenetv2_widen_512_softmax-arcface_iccv_emore_bs200_e36_lr0.100_step[15, 25, 31]_'
    #                                 'auxnet_softmax_arcface_label_nosmooth_warmup_20190706/check_point/checkpoint_034.pth')


    # iccv_ms1m_mobilefacenet_baseline_two_stage
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                                 '2stage/dim128/stage2/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore_bs200_e36_lr0.100_'
    #                                 'step[15, 25, 31]_kaiming_init_auxnet_arcface_new_op_stage2_20190726/check_point/checkpoint_031.pth')

    # ms1m_v2_mobilefacenet_p0.25_with_fc
    # megaface_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_with_fc/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_with_fc/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/'
    #                                 'pytorch_pruned0.25_mobilefacenet_v1_with_fc/checkpoint_024.pth')

    # ms1m_v2_mobilefacenet_p0.25_without_fc
    # megaface_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/'
    #                                 'pytorch_pruned0.25_mobilefacenet_v1_better/'
    #                                 'pytorch_pruned0.25_mobilefacenet_v1_without_fc_checkpoint_022.pth')

    # iccv_emore: new_mobilefacenet_p0.5_with_fc
    # megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.5_with_p_fc/' \
    #                'fp_megaface_112x112_norm_features'
    # facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0.5_with_p_fc/' \
    #                 'fp_facescrub_112x112_norm_features'
    # check_point_params = torch.load('/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                                 '2stage/dim128/finetune/log_aux_mobilefacenetv2_baseline_128_arcface_iccv_emore'
    #                                 '_bs512_e36_lr0.010_step[15, 25, 31]_bs512_cosine_finetune_20190804/check_point/'
    #                                 'checkpoint_033.pth')

    # iccv_emore: new_mobilefacenet_p0.5_without_fc
    megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0_without_p_fc/' \
                   'fp_megaface_112x112_norm_features'
    facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0_without_p_fc/' \
                    'fp_facescrub_112x112_norm_features'
    check_point_params = torch.load('/home/xiezheng/lidaiyuan/Face_integration/pretrain_model/checkpoint_49.pth')

    print("megaface_out = ", megaface_out)
    print("facescrub_out = ", facescrub_out)

    # model = MobileFaceNet()    # old: mobilefacenet_v1
    # model = resnet34()         # r34
    # model = pruned_Mobilefacenet(pruning_rate=0.25)  # old
    # model = Mobilefacenetv2_v3_width(embedding_size=512, width_mult=1.315)
    # model = Mobilefacenetv2_v3_width(embedding_size=128, width_mult=1.0)

    model = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0)
    print(model)

    model_state = check_point_params['model']
    # lfw_acc = check_point_params['val_acc']
    # lfw_acc = check_point_params['lfw_best_acc']

    print(model)
    # print("lfw_acc={}".format(lfw_acc))
    model.load_state_dict(model_state)
    print("model load success !!!")
    # torch.save(model_state, '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better'
    #                         '/mobilefacenet_p0.25_without_fc_checkpoint_22.pth')
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
            feature = get_feature(image_path, model, image_shape)

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
            count = process_batch_feature(args, image_path_list, megaface_out, dataset_path, model, image_shape)
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
        count = process_batch_feature(args, image_path_list, megaface_out, dataset_path, model, image_shape)
        succ += count
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=64)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--skip', type=int, help='', default=0)
    parser.add_argument('--mf', type=int, help='', default=1)

    # parser.add_argument('--algo', type=str, help='', default='mxsphereface20c')
    # parser.add_argument('--algo', type=str, help='', default='mobilefacenetNcnn')
    parser.add_argument('--algo', type=str, help='', default='mobilefacenetPytorch')
    # parser.add_argument('--algo', type=str, help='', default='prunedR34Pytorch')

    parser.add_argument('--gpu', type=str, help='', default='5')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
