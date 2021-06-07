# -*- coding: utf-8 -*-

import os
import random
import numpy

def test_list(pairs_list_name):
    with open(pairs_list_name, 'r') as f:
        true_pair = 0
        false_pair = 0
        valid_list = f.readlines()
        for i in range(len(valid_list)):
            valid_list[i] = valid_list[i].rstrip('\n').split(' ')
            if valid_list[i][0].split('\\')[0] == valid_list[i][1].split('\\')[0]:
                true_pair += 1
            if valid_list[i][0].split('\\')[0] != valid_list[i][1].split('\\')[0]:
                false_pair += 1
            # if valid_list[i][0].split('/')[0] in train_list:
            #     print("error!!!")
            # if valid_list[i][1].split('/')[0] in train_list:
            #     print("error!!!")
            # print(valid_list[i][0].split('/')[0])
            # print(valid_list[i][1].split('/')[0])

        print(true_pair)
        print(false_pair)

def generate_valid_pairs(src_path, pairs_list_name, true_num=40000, false_num = 120000):
    all_list = os.listdir(src_path)
    print("all_list_len: ")
    print(len(all_list))
    # print(all_list)

    # with open('sample_folder_list.txt', 'r') as f:
    #     train_list = f.readlines()
    #     for i in range(len(train_list)):
    #         train_list[i] = train_list[i].rstrip('\n')
    #     print("train_list_len: ")
    #     print(len(train_list))
    #     # print(train_list)


    # valid_list = list(set(all_list).difference(set(train_list)))
    valid_list = all_list

    print("valid_list_len: ")
    print(len(valid_list))
    # print(valid_list)

    # sample true pair  100,000
    with open(pairs_list_name, 'w+') as f:

        for pair_num in range(true_num):
            while True:
                rand_people = random.randint(0, len(valid_list) - 1)
                face_list = os.listdir(os.path.join(src_path, valid_list[rand_people]))
                if len(face_list) >= 2:
                    break
            rand_face = random.sample(range(len(face_list)), 2)

            face_path_1 = os.path.join(valid_list[rand_people], face_list[rand_face[0]])
            face_path_2 = os.path.join(valid_list[rand_people], face_list[rand_face[1]])

            f.write(face_path_1 + ' ')
            f.write(face_path_2 + ' ')
            f.write('1\n')


    # sample false pair  300,000
        for pair_num in range(false_num):
            rand_people = random.sample(range(len(valid_list)), 2)
            face_list_1 = os.listdir(os.path.join(src_path, valid_list[rand_people[0]]))
            face_list_2 = os.listdir(os.path.join(src_path, valid_list[rand_people[1]]))
            rand_face_1 = random.randint(0, len(face_list_1) - 1)
            rand_face_2 = random.randint(0, len(face_list_2) - 1)

            face_path_1 = os.path.join(valid_list[rand_people[0]], face_list_1[rand_face_1])
            face_path_2 = os.path.join(valid_list[rand_people[1]], face_list_2[rand_face_2])

            f.write(face_path_1 + ' ')
            f.write(face_path_2 + ' ')
            f.write('0\n')


if __name__ =="__main__":
    src_path = r'D:\working\ir_dataset_labeled\jidian_test_400_190727_align'
    pairs_list_name = 'sample_valid_list_for_jidian_test_400_190727_align.txt'
    generate_valid_pairs(src_path, pairs_list_name, true_num=10000, false_num=30000)
    test_list(pairs_list_name)

