import os

# src_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/imgs'
# dst_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/sample_imgs'
#
# txt_path = 'sample_folder_list.txt'

src_path = '/mnt/ssd/faces/faces_webface_112x112/imgs'

# dst_path = '/mnt/ssd/faces/faces_webface_112x112/verfication_0.7_imgs'
# dst_path = '/mnt/ssd/faces/faces_webface_112x112/train_0.3_imgs'
dst_path = '/mnt/ssd/faces/faces_webface_112x112/train_0.1_imgs'

# if not os.listdir(dst_path):
#     os.makedirs(dst_path)


# txt_path = '/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/' \
#            'sub_face_data/verification_folder_list.txt'
# txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/" \
#                  "sub_face_data/train_folder_list.txt"
txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/faces_webface_112x112/" \
                       "sub_face_data/train_small_folder_list.txt"

count = 1
with open(txt_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        src_dir_path = os.path.join(src_path, line)
        dst_dir_path = os.path.join(dst_path, line)
        print("count={}, {} -> {}".format(count, src_dir_path, dst_dir_path))
        os.symlink(src_dir_path, dst_dir_path)
        count += 1