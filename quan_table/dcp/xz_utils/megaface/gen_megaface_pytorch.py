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

# from models import MobileFaceNet
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
# import face_preprocess
# import facenet
# import lfw
# from caffe.proto import caffe_pb2

from dcp.models.insightface_resnet_pruned import pruned_LResNet34E_IR
from dcp.models.mobilefacenet_pruned import  pruned_Mobilefacenet

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


def main(args):
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    image_shape = [int(x) for x in args.image_size.split(',')]
    # for model in args.model.split('|'):
    #     vec = model.split(',')
    #     assert len(vec) > 1
    #     prefix = vec[0]
    #     epoch = int(vec[1])
    #     print('loading', prefix, epoch)
    #     net = edict()
    #     net.ctx = ctx
    #     net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    #     # net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    #     all_layers = net.sym.get_internals()
    #     net.sym = all_layers['fc1_output']
    #     net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
    #     net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
    #     net.model.set_params(net.arg_params, net.aux_params)
    #     # _pp = prefix.rfind('p')+1
    #     # _pp = prefix[_pp:]
    #     # net.patch = [int(x) for x in _pp.split('_')]
    #     # assert len(net.patch)==5
    #     # print('patch', net.patch)
    #     nets.append(net)


    megaface_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/megaface_112x112/lst"
    # 186
    facescrub_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2/small_lst"

    megaface_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned_mobilefacenet_v1/' \
                   'pytorch_pruned0.25_mobilefacenet_v1_lr0.01_checkpoint22_feature-result/' \
                   'fp_megaface_112x112_norm_features'
    facescrub_out = '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned_mobilefacenet_v1/' \
                    'pytorch_pruned0.25_mobilefacenet_v1_lr0.01_checkpoint22_feature-result/' \
                    'fp_facescrub_112x112_norm_features'

    dataset_path = "/home/dataset/xz_datasets/Megaface/Mageface_aligned"

    print("megaface_out = ", megaface_out)
    print("facescrub_out = ", facescrub_out)

    # for resnet34
    # prepare model
    # 读取的模型类型为DataParallel, 现将其读取到CPU上
    # check_point_params = torch.load('/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/check_point/checkpoint_48.tar')
    # # 取出DataParallel中的一般module类型
    # epoch = check_point_params['epoch']
    # print(epoch)
    # acc = check_point_params['lfw_acc']
    # print(acc)
    # model = check_point_params['model'].module  # 模型类型为：torch.nn.DataParallel
    # print(model)
    #
    # model_state = model.state_dict()
    # torch.save(model_state,
    #            '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/insightface_r34_epoch48.pth')


    # for pruned0.5_r34
    check_point_params = torch.load('/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned_mobilefacenet_v1/'
                                    'checkpoint_022.pth')

    # model = MobileFaceNet(embedding_size=128, blocks = [1,4,6,2])    # formobilefacenet_v1
    # model = MobileFaceNet(embedding_size=256, blocks = [2, 8, 16, 4])  # formobilefacenet_v2
    # model = pruned_LResNet34E_IR(pruning_rate=0.7)
    model = pruned_Mobilefacenet(pruning_rate=0.25)

    model_state = check_point_params['model']
    lfw_acc = check_point_params['val_acc']
    print(model)
    print("lfw_acc={}".format(lfw_acc))
    model.load_state_dict(model_state)
    print("model load success !!!")
    torch.save(model_state,
               '/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned_mobilefacenet_v1/'
               'pytorch_pruned0.25_mobilefacenet_v1_lr0.01_checkpoint22_feature-result/'
               'pruned0.25_mobilefacenet_v1_lr0.01_checkpoint_022.pth')
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
            # a = a.replace(' ', '_')
            # b = b.replace(' ', '_')
            out_dir = os.path.join(facescrub_out, a)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # file, ext = os.path.splitext(b)
            # image_id = int(file.split('_')[-1])
            # if image_id==40499 or image_id==10788 or image_id==2367:
            #  b = file
            # if len(ext)==0:
            #  print(image_path)
            #  image_path = image_path+".jpg"
            # if facescrub_aligned_root is not None:
            #  _vec = image_path.split('/')
            #  _image_path = os.path.join(facescrub_aligned_root, _vec[-2], _vec[-1])
            #  _base, _ext = os.path.splitext(_image_path)
            #  if _ext=='.gif':
            #    _image_path = _base+".jpg"
            #    print('changing', _image_path)
            #  if os.path.exists(_image_path):
            #    image_path = _image_path
            #    bbox = None
            #    landmark = None
            #    aligned = True
            #  else:
            #    print("not aligned:",_image_path)
            image_path = get_image_path(image_path, dataset_path, "MegaFace/")
            print(image_path)
            feature = get_feature(image_path, model, image_shape)
            # if feature is None:
            #     print('feature none', image_path)
            #     continue
            # print(np.linalg.norm(feature))
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
    print("get megaface features start!")
    for line in open(megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing megaface", i, succ)
        print('i=', str(i))
        # if i <= args.skip:
        #     continue
        image_path, label, bbox, landmark, aligned = parse_lst_line(line)
        # assert aligned == True
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # continue
        # print(landmark)

        image_path = get_image_path(image_path, dataset_path, "MegaFace/")
        feature = get_feature(image_path, model, image_shape)

        # if feature is None:
        #     continue
        out_path = os.path.join(out_dir, b + "_%s_%dx%d.bin" % (args.algo, image_shape[1], image_shape[2]))
        # print(out_path)
        write_bin(out_path, feature)
        succ += 1
        i += 1
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=100)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    # parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--mean', type=int, help='', default=0)
    parser.add_argument('--seed', type=int, help='', default=727)
    parser.add_argument('--skip', type=int, help='', default=0)
    parser.add_argument('--concat', type=int, help='', default=0)
    parser.add_argument('--fsall', type=int, help='', default=0)
    parser.add_argument('--mf', type=int, help='', default=1)

    # parser.add_argument('--algo', type=str, help='', default='mxsphereface20c')
    # parser.add_argument('--algo', type=str, help='', default='mobilefacenetNcnn')
    # parser.add_argument('--algo', type=str, help='', default='mobilefacenetPytorch')
    # parser.add_argument('--algo', type=str, help='', default='prunedR34Pytorch')
    parser.add_argument('--algo', type=str, help='', default='prunedMobilefacenetPytorch')

    parser.add_argument('--gpu', type=str, help='', default='3')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
