import os
from tqdm import tqdm
from numpy.lib.function_base import append

label_name1 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/exp_train_set_210318_exp_20210318122914_train_label.txt'
train_name1 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/exp_train_set_210318_exp_20210318122914_train.txt'

label_name2 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_nir_210510_exp_20210510171724_train_label.txt'
train_name2 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_nir_210510_exp_20210510171724_train.txt'

# 拿到四个list，遍历一个拼接到上一个，遍历的同时记录index，如果有重复记得把label也删了

def getlist(filename):
    with open(filename,'r') as f:
        list = f.read().splitlines()
    return list
def savelist(filename,list):
    with open(filename,'w') as f:
        str='\n'
        f.write(str.join(list))


list_train1 = getlist(train_name1)
list_train2 = getlist(train_name2)
list_train_label1 = getlist(label_name1)
list_train_label2 = getlist(label_name2)


print(len(list_train1))
print(len(list_train_label1))
print(len(list_train2))
print(len(list_train_label2))

# 整合filelist，去重
# print(‘3’ in list) 输出结果为True；
for i in tqdm( range(len((list_train2))) ):
    # print(list_train1[i])
    if list_train2[i] not in list_train1:
        list_train1.append(list_train2[i])
        list_train_label1.append(list_train_label2[i])



savelist('./train_file_list/nir_rgb_20210603_train_label.txt',list_train_label1)
savelist('./train_file_list/nir_rgb_20210603_train.txt',list_train1)

print(len(list_train1))
print(len(list_train_label1))
print(len(list_train2))
print(len(list_train_label2))
