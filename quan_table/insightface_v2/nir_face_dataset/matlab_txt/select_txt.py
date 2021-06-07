#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-11 下午8:07
# @Author  : xiezheng
# @Site    : 
# @File    : select_txt.py


from easydict import EasyDict as edict


def delete_txt(txt_path, new_txt_path):
    num = 1
    f = open(new_txt_path, 'a')

    for line in open(txt_path, encoding='gbk'):

        line = line.strip()
        if num == 1:
            print("line =", line)
            f.write(line + "\n")
        else:
            print("num = ", str(num), "\n", "line", line)
            strArray = line.split('\t\t')
            subdir1 = strArray[0]
            subdir2 = strArray[1]
            distance = float(strArray[2])

            if ('20190019' in subdir1 and '20190101' in subdir2) or \
                    ('20190019' in subdir2 and '20190101' in subdir1):
                print('delete !')
                num = num + 1
                continue

            new_line = subdir1 + "\t\t" + subdir2 +"\t\t" + str(distance)
            print("serlected line=", new_line)
            f.write(new_line + "\n")
        num = num + 1
    f.close()

def select_dir(txt_path, new_txt_path):
    num = 1
    f = open(new_txt_path, 'a')
    dir_list = []

    for line in open(txt_path, encoding='gbk'):
        skip_1, skip_2 = False, False
        line = line.strip()
        if num == 1:
            print("line =", line)
            f.write(line + "\n")
        else:
            print("num = ", str(num))
            strArray = line.split('\t\t')
            subdir1 = strArray[0].split('/')[0]
            subdir2 = strArray[1].split('/')[0]
            distance = float(strArray[2])
            # print('subdir1={}, subdir1={}'.format(subdir1, subdir2))
            # print('dir_list={}'.format(dir_list))

            if len(dir_list) != 0:
                for i in range(len(dir_list)):
                    if subdir1 == dir_list[i]:
                        skip_1 = True
                        print("{} is on list!".format(subdir1))
                        break

                for i in range(len(dir_list)):
                    if subdir2 == dir_list[i]:
                        skip_2 = True
                        print("{} is on list!".format(subdir2))
                        break

            if skip_1 == False:
                print('add {}'.format(subdir1))
                dir_list.append(subdir1)

            if skip_2 == False:
                print('add {}'.format(subdir2))
                dir_list.append(subdir2)
        num = num + 1
        # if num == 10:
        #     break

    for i in range(len(dir_list)):
        new_line = dir_list[i]
        print('i={}, new_line={}'.format(i, new_line))
        f.write(new_line + "\n")
    f.close()


if __name__ == '__main__':
    # txt_path = "./result_top5000_inter.txt"
    # new_txt_path = "./result_top5000_inter_select_split.txt"
    # delete_txt(txt_path, new_txt_path)

    txt_path = "./result_top5000_inter_select_split.txt"
    new_txt_path = "./select.txt"
    select_dir(txt_path, new_txt_path)