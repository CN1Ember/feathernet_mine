from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import struct
import sys

import numpy as np
from easydict import EasyDict as edict


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


def main(args):
    print(args)
    image_shape = [int(x) for x in args.image_size.split(',')]

    # 186
    megaface_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/megaface_112x112/lst"
    facescrub_lst = "/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2/small_lst"

    # new_mobilefacenet_sq_int8
    mobilefacenet_facescrub_txt = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/ncnn-int8/' \
                                  'mf-p0.5-without-p-fc-ncnn-int8-rgb-liujing-17-87.18_result/' \
                                  'facescrub_ncnn_112x112_3532_result.txt'
    mobilefacenet_megaface_txt = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/ncnn-int8/' \
                                 'mf-p0.5-without-p-fc-ncnn-int8-rgb-liujing-17-87.18_result/' \
                                 'megaface_ncnn_112x112_1027058_result.txt'

    megaface_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/ncnn-int8/' \
                   'mf-p0.5-without-p-fc-ncnn-int8-rgb-liujing-17-87.18_result/fp_megaface_112x112_norm_features'
    facescrub_out = '/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/ncnn-int8/' \
                    'mf-p0.5-without-p-fc-ncnn-int8-rgb-liujing-17-87.18_result/fp_facescrub_112x112_norm_features'

    print("mobilefacenet_facescrub_txt = ", mobilefacenet_facescrub_txt)
    print("mobilefacenet_megaface_txt = ", mobilefacenet_megaface_txt)
    print("megaface_out = ", megaface_out)
    print("facescrub_out = ", facescrub_out)

    facescrub_features = np.loadtxt(mobilefacenet_facescrub_txt, dtype=np.float32)
    print('facescrub_features size is: ', facescrub_features.shape)
    megaface_features = np.loadtxt(mobilefacenet_megaface_txt, dtype=np.float32)
    print('megaface_features size is: ', megaface_features.shape)
    print("feature txt file load finished!!")

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
            feature = facescrub_features[i]
            out_path = os.path.join(out_dir, b + "_%s_%dx%d.bin" % (args.algo, image_shape[1], image_shape[2]))
            write_bin(out_path, feature)
            succ += 1
            i += 1
        print('facescrub finish!', i, succ)

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

        feature = megaface_features[i]
        out_path = os.path.join(out_dir, b + "_%s_%dx%d.bin" % (args.algo, image_shape[1], image_shape[2]))
        # print(out_path)
        write_bin(out_path, feature)
        succ += 1
        i += 1
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--skip', type=int, help='', default=0)
    parser.add_argument('--mf', type=int, help='', default=1)
    parser.add_argument('--algo', type=str, help='', default='mobilefacenetNcnn')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
