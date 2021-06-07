from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import struct
import sys

import numpy as np

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
# import face_preprocess

feature_dim = 128
# feature_dim = 256
# feature_dim = 512
feature_ext = 1


def load_bin(path, fill=0.0):
    with open(path, 'rb') as f:
        bb = f.read(4 * 4)
        # print(len(bb))
        v = struct.unpack('4i', bb)
        # print(v[0])
        bb = f.read(v[0] * 4)
        v = struct.unpack("%df" % (v[0]), bb)
        feature = np.full((feature_dim + feature_ext,), fill, dtype=np.float32)
        feature[0:feature_dim] = v
        # feature = np.array( v, dtype=np.float32)
    # print(feature.shape)
    # print(np.linalg.norm(feature))
    return feature

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

def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

def main(args):
    print(args)
    args.facescrub_feature_dir_out = args.facescrub_feature_dir + '_cm'
    args.megaface_feature_dir_out = args.megaface_feature_dir + '_cm'

    out_algo = args.suffix
    if len(args.algo) > 0:
        out_algo = args.algo

    fs_noise_map = {}
    for line in open(args.facescrub_noises, 'r'):
        if line.startswith('#'):
            continue
        line = line.strip()
        fname = line.split('.')[0]
        p = fname.rfind('_')
        fname = fname[0:p]
        fs_noise_map[line] = fname   # key-value

    print("All facescrub noises number:", len(fs_noise_map))

    i = 0
    fname2center = {}
    noises = []
    for line in open(args.facescrub_lst, 'r'):
        if i % 1000 == 0:
            print("reading fs", i)
        i += 1
        image_path, label, bbox, landmark, aligned = parse_lst_line(line)
        assert aligned == True
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]   # a-name, b-name_id
        feature_path = os.path.join(args.facescrub_feature_dir, a, "%s_%s.bin" % (b, args.suffix))

        # if not os.path.exists(args.facescrub_feature_dir_out):
        #     os.makedirs(args.facescrub_feature_dir_out)

        feature_dir_out = os.path.join(args.facescrub_feature_dir_out, a)
        # print("feature_dir_out={}".format(feature_dir_out))
        if not os.path.exists(feature_dir_out):
            os.makedirs(feature_dir_out)
            # print("makedirs success!!")

        feature_path_out = os.path.join(args.facescrub_feature_dir_out, a, "%s_%s.bin" % (b, out_algo))
        # print(b)
        if not b in fs_noise_map:
            # shutil.copyfile(feature_path, feature_path_out)
            feature = load_bin(feature_path)
            write_bin(feature_path_out, feature)
            if not a in fname2center:
                fname2center[a] = np.zeros((feature_dim + feature_ext,), dtype=np.float32)
            fname2center[a] += feature
        else:
            # print('n', b)
            noises.append((a, b))
    print("This small-facescrub noises number:", len(noises))

    for k in noises:
        a, b = k
        assert a in fname2center
        center = fname2center[a]
        g = np.zeros((feature_dim + feature_ext,), dtype=np.float32)
        g2 = np.random.uniform(-0.001, 0.001, (feature_dim,))
        g[0:feature_dim] = g2
        f = center + g
        _norm = np.linalg.norm(f)
        f /= _norm
        feature_path_out = os.path.join(args.facescrub_feature_dir_out, a, "%s_%s.bin" % (b, out_algo))
        write_bin(feature_path_out, f)

    mf_noise_map = {}
    for line in open(args.megaface_noises, 'r'):
        if line.startswith('#'):
            continue
        line = line.strip()
        _vec = line.split("\t")
        if len(_vec) > 1:
            line = _vec[1]
        mf_noise_map[line] = 1
    print("All megaface noises number:", len(mf_noise_map))

    i = 0
    nrof_noises = 0
    for line in open(args.megaface_lst, 'r'):
        if i % 1000 == 0:
            print("reading mf", i)
        i += 1
        image_path, label, bbox, landmark, aligned = parse_lst_line(line)
        assert aligned == True
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        feature_path = os.path.join(args.megaface_feature_dir, a1, a2, "%s_%s.bin" % (b, args.suffix))
        feature_dir_out = os.path.join(args.megaface_feature_dir_out, a1, a2)
        if not os.path.exists(feature_dir_out):
            os.makedirs(feature_dir_out)
        feature_path_out = os.path.join(args.megaface_feature_dir_out, a1, a2, "%s_%s.bin" % (b, out_algo))
        bb = '/'.join([a1, a2, b])
        # print(b)
        if not bb in mf_noise_map:
            feature = load_bin(feature_path)
            write_bin(feature_path_out, feature)
            # shutil.copyfile(feature_path, feature_path_out)
        else:
            feature = load_bin(feature_path, 100.0)
            write_bin(feature_path_out, feature)
            # g = np.random.uniform(-0.001, 0.001, (feature_dim,))
            # print('n', bb)
            # write_bin(feature_path_out, g)
            nrof_noises += 1
    print("This small-megaface noises number:", nrof_noises)
    print(args)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--facescrub-noises', type=str, help='', default='./facescrub_noises.txt')
    parser.add_argument('--megaface-noises', type=str, help='', default='./megaface_noises.txt')
    parser.add_argument('--algo', type=str, help='', default='')

    # parser.add_argument('--suffix', type=str, help='', default='mobilefacenetMxnet_112x112')
    # parser.add_argument('--suffix', type=str, help='', default='mobilefacenetNcnn_112x112')
    parser.add_argument('--suffix', type=str, help='', default='mobilefacenetPytorch_112x112')

    # parser.add_argument('--suffix', type=str, help='', default='r34Pytorch_112x112')
    # parser.add_argument('--suffix', type=str, help='', default='resnet34Pytorch_112x112')
    # parser.add_argument('--suffix', type=str, help='', default='prunedR34Pytorch_112x112')

    # 186
    parser.add_argument('--megaface-lst', type=str, help='',
                        default='/home/dataset/xz_datasets/Megaface/Mageface_aligned/megaface_112x112/lst')
    parser.add_argument('--facescrub-lst', type=str, help='',
                default='/home/dataset/xz_datasets/Megaface/Mageface_aligned/Challenge1/facescrub_112x112_v2/small_lst')

    # parser.add_argument('--megaface-feature-dir', type=str, help='', default='/home/dataset/xz_datasets/Megaface/'
    #                     'iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage/fp_megaface_112x112_norm_features')
    # parser.add_argument('--facescrub-feature-dir', type=str, help='', default='/home/dataset/xz_datasets/Megaface/'
    #                     'iccv_ms1m_result/pytorch_mobilefacenet_baseline_two_stage/fp_facescrub_112x112_norm_features')

    # parser.add_argument('--megaface-feature-dir', type=str, help='', default='/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_with_fc/fp_megaface_112x112_norm_features')
    # parser.add_argument('--facescrub-feature-dir', type=str, help='', default='/home/dataset/xz_datasets/Megaface/pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_with_fc/fp_facescrub_112x112_norm_features')

    parser.add_argument('--megaface-feature-dir', type=str, help='',
                        default='/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0_without_p_fc/fp_megaface_112x112_norm_features')
    parser.add_argument('--facescrub-feature-dir', type=str, help='',
                        default='/home/dataset/xz_datasets/Megaface/iccv_ms1m_result/pytorch_mobilefacenet_p0_without_p_fc/fp_facescrub_112x112_norm_features')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    a = True
