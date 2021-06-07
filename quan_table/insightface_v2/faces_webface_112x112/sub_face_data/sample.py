import os
import random
import numpy

src_path = '/mnt/ssd/faces/faces_webface_112x112/imgs'
# src_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/imgs'
# dst_path = '/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1/sample_imgs'

sample_rate = 1.0
verification_rate = 0.7
folder_list = os.listdir(src_path)

folder_num = len(folder_list)
sample_num = int(folder_num * sample_rate)

folder_index_list = range(folder_num)
sample_folder_index_list = random.sample(folder_index_list, sample_num)
print("sample_folder_index_list len={}".format(len(sample_folder_index_list)))

train_folder_list = []
verification_folder_list = []
for i in range(len(sample_folder_index_list)):
    if i < int(verification_rate * len(sample_folder_index_list)):
        verification_folder_list.append(folder_list[sample_folder_index_list[i]])
    else:
        # print("*"*20)
        train_folder_list.append(folder_list[sample_folder_index_list[i]])
        # print("i={}, dir={}".format(i, sample_folder_index_list[i]))


train_small_folder_list = []
train_sample_rate = 1.0/3
for j in range(len(train_folder_list)):
    if j < int(train_sample_rate * len(train_folder_list)):
        train_small_folder_list.append(train_folder_list[j])


with open('train_folder_list.txt', 'w+') as f:
    for i in range(len(train_folder_list)):
        f.write(train_folder_list[i] + '\n')

with open('verification_folder_list.txt', 'w+') as f:
    for i in range(len(verification_folder_list)):
        f.write(verification_folder_list[i] + '\n')

with open('train_small_folder_list.txt', 'w+') as f:
    for i in range(len(train_small_folder_list)):
        f.write(train_small_folder_list[i] + '\n')