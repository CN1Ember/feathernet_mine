
import argparse
import os

# from insightface_v2.insightface_data.data_pipe import load_bin, load_mx_rec
from insightface_v2.insightface_data.data_pipe import load_bin, load_mx_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    # 186
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/home/dataset/Face/FaceRecognition/faces_ms1m-refine-v2_112x112', type=str)

    # 230
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/home/dataset/Face/xz_FaceRecognition/faces_ms1m-refine-v2_112x112', type=str)
    # parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='faces_emore', type=str)

    # 231
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/home/datasets/Face/FaceRecognition/faces_ms1m-refine-v2_112x112/', type=str)
    # parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='faces_emore', type=str)

    # 241 dataset
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/home/datasets/Face/FaceRecognition/faces_ms1m-refine-v2_112x112/', type=str)

    # 241 ssd
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/mnt/ssd/faces/', type=str)

    # 247
    parser.add_argument("-p", "--data_path", help="data path",
                        default='/mnt/ssd/datasets/FaceRec', type=str)

    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='faces_emore', type=str)

    args = parser.parse_args()
    rec_path = os.path.join(args.data_path, args.rec_path)
    load_mx_rec(rec_path)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, (bin_files[i] + '.bin')), os.path.join(rec_path, bin_files[i]))
    print(args)