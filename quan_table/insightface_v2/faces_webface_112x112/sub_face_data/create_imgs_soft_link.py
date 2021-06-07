import os


def create_soft_link(txt_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    count = 1
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            src_dir_path = line
            line_arr = line.split('/')
            dst_dir_path = os.path.join(dst_path, line_arr[-2])
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
            dst_dir_path = os.path.join(dst_dir_path, line_arr[-1])

            # print("{}-->{}".format(src_dir_path, dst_dir_path))
            print("count={}, {} -> {}".format(count, src_dir_path, dst_dir_path))
            os.symlink(src_dir_path, dst_dir_path)
            count += 1
            # return

# train_test_split
# train_txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/" \
#                  "faces_webface_112x112/sub_face_data/0.3_imgs_train.txt"
# test_txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/" \
#                 "faces_webface_112x112/sub_face_data/0.3_imgs_test.txt"
#
# train_dst_path = '/mnt/ssd/faces/faces_webface_112x112/0.3_imgs_train_data'
# test_dst_path = '/mnt/ssd/faces/faces_webface_112x112/0.3_imgs_test_data'


train_txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/" \
                 "faces_webface_112x112/sub_face_data/0.1_imgs_train.txt"
test_txt_path = "/home/xiezheng/program2019/insightface_DCP/insightface_v2/" \
                "faces_webface_112x112/sub_face_data/0.1_imgs_test.txt"

train_dst_path = '/mnt/ssd/faces/faces_webface_112x112/0.1_imgs_train_data'
test_dst_path = '/mnt/ssd/faces/faces_webface_112x112/0.1_imgs_test_data'

create_soft_link(train_txt_path, train_dst_path)
create_soft_link(test_txt_path, test_dst_path)