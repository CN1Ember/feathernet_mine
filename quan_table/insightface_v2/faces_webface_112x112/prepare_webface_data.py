
import argparse
import os

# from insightface_v2.insightface_data.data_pipe import load_bin, load_mx_rec
from insightface_v2.faces_webface_112x112.webface_data_pipe import load_bin, load_mx_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')

    # 241 ssd
    # parser.add_argument("-p", "--data_path", help="data path",default='/mnt/ssd/faces/', type=str)

    # # 230
    # parser.add_argument("-p", "--data_path", help="data path",
    #                     default='/home/dataset/Face/xz_FaceRecognition', type=str)

    # 246
    # parser.add_argument("-p", "--data_path", help="data path",default='/mnt/ssd/faces', type=str)

    # 247 ssd
    parser.add_argument("-p", "--data_path", help="data path", default='/mnt/ssd/datasets/Faces/train/', type=str)

    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='faces_webface_112x112', type=str)

    args = parser.parse_args()
    print(args)
    rec_path = os.path.join(args.data_path, args.rec_path)
    # load_mx_rec(rec_path)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw']
    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, (bin_files[i] + '.bin')), os.path.join(rec_path, bin_files[i]))
