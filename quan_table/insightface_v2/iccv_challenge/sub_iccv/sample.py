import os
import random
import numpy

src_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/imgs'
dst_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/sample_imgs'

sample_rate = 0.1
folder_list = os.listdir(src_path)

folder_num = len(folder_list)
sample_num = int(folder_num * sample_rate)

folder_index_list = range(folder_num)
sample_folder_index_list = random.sample(folder_index_list, sample_num)
# print(sample_folder_index_list)

sample_folder_list = []
for i in range(len(sample_folder_index_list)):
    sample_folder_list.append(folder_list[sample_folder_index_list[i]])

with open('sample_folder_list.txt', 'w+') as f:
    for i in range(len(sample_folder_list)):
        f.write(sample_folder_list[i] + '\n')
# for folder_list