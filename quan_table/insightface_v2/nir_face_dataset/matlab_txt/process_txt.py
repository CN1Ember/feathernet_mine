#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-15 下午9:59
# @Author  : xiezheng
# @Site    : 
# @File    : process_txt.py

from easydict import EasyDict as edict


def simple_str(str):
    strArray = str.split('/')
    # print(strArray)
    label = strArray[-2]
    newStr = strArray[-2] + "/" + (strArray[-1].split('.'))[0]
    return newStr, label

def sort_data(data, top_k, reverse):
    result = sorted(data, key=lambda x: (x['distance']), reverse=reverse)
    select = []
    for i in range(top_k):
        select.append(result[i])
    return select


# server
# txt_path = "/home/dataset/xz_datasets/result_txt/" + "fp_mxnet_resnet34_result.txt"
# new_txt_path = "/home/dataset/xz_datasets/result_txt/" + "fp_mxnet_resnet34_result_select.txt"


# txt_path = "/home/dataset/xz_datasets/jidian_face_499/feature_result/" + "int8_ncnn_mobilefacenet_result.txt"
# new_txt_path = "/home/dataset/xz_datasets/jidian_face_499/feature_result/" \
#                + "int8_ncnn_mobilefacenet_result-5000.txt"

txt_path = "D:\\2019Programs\jidian2019\极点智能\Draw_ROC\jidian_test_200_190727_align_result\iccv_train_mf_p0.5_without_p_fc\pytorch_mf_p0.5_fp_nir_finetune_checkpoint29_last1\\result.txt"
new_txt_path = "D:\\2019Programs\jidian2019\极点智能\Draw_ROC\jidian_test_200_190727_align_result\iccv_train_mf_p0.5_without_p_fc\pytorch_mf_p0.5_fp_nir_finetune_checkpoint29_last1\\result_top5000.txt"


num = 1
f = open(new_txt_path, 'a+')
inter_data = []
intra_data = []
for line in open(txt_path, encoding='gbk'):
    line = line.strip()
    if num == 1 or num == 2:
        print("line", line)
        f.write(line + "\n")
    else:
        # print("num = ", str(num), "\n", "line", line)
        print("num = ", str(num), "\n")
        strArray = line.split('\t')
        str1, label1 = simple_str(strArray[0])
        str2, label2 = simple_str(strArray[1])
        fdata = edict()
        fdata.image1 = str1
        fdata.image2 = str2
        fdata.distance = float(strArray[2])
        if label1 == label2:
            intra_data.append(fdata)
        else:
            inter_data.append(fdata)
        # newLine = fdata.image1 + "\t\t" + fdata.image2 + "\t\t" + str(fdata.distance)
        # print("newline", newLine)
        # f.write(newLine + "\n")
    num = num + 1

inter_data_top_k = 5000
inter_title = str(len(inter_data)) + "\ninter pairs(height->low)" + "\n"
print(inter_title)
f.write(inter_title)

select_inter_data = sort_data(inter_data, inter_data_top_k, reverse=True)
for i in range(len(select_inter_data)):
    line = select_inter_data[i].image1 + "\t\t" + select_inter_data[i].image2 \
           + "\t\t" + str(select_inter_data[i].distance)
    print(line)
    f.write(line + "\n")


intra_data_top_k = 5000
intra_title = "\n\n" + str(len(intra_data)) + "\nintra pairs(low->height)" + "\n"
print(intra_title)
f.write(intra_title)

select_intra_data = sort_data(intra_data, intra_data_top_k, reverse=False)
for i in range(len(select_intra_data)):
    line = select_intra_data[i].image1 + "\t\t" + \
           select_intra_data[i].image2 + "\t\t" + str(select_intra_data[i].distance)
    print(line)
    f.write(line + "\n")

f.close()


