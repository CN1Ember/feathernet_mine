#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/5 21:39
# @Author  : xiezheng
# @Site    : 
# @File    : select_inter_score.py



def select_dir(txt_path, new_txt_path):
    num = 1
    f = open(new_txt_path, 'a')

    for line in open(txt_path, encoding='gbk'):
        line = line.strip()
        if num == 1:
            print("line =", line)
            f.write(line + "\n")
        else:
            strArray = line.split('\t')
            subdir1 = strArray[0].split('/')[0]
            subdir2 = strArray[1].split('/')[0]
            distance = float(strArray[2])

            if subdir1 == subdir2:
                num = num + 1
                continue
            else:
                print("num = ", str(num))
                if distance >= 0.5:
                    f.write(line + '\n')
        num = num + 1
    f.close()




if __name__ == '__main__':

    txt_path = "D:\\2019Programs\jidian2019\mobilefacenet\误识\\20191105高分误识\err_20191105_aligned_feature/" \
               "result.txt"
    new_txt_path = "D:\\2019Programs\jidian2019\mobilefacenet\误识\\20191105高分误识\err_20191105_aligned_feature/" \
               "result_0.5.txt"
    select_dir(txt_path, new_txt_path)