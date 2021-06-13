import os
from tqdm import tqdm
from numpy.lib.function_base import append

label_name1 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_21060301_exp_20210603221606NIR_train_label.txt'
train_name1 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_21060301_exp_20210603221606NIR_train.txt'

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

list_check=[]
list_label_check=[]

# 整合filelist，去重
# print(‘3’ in list) 输出结果为True；
fold_path = '20210510_f3c_nir_print'

for i in tqdm( range(len((list_train2))) ):
    # print(list_train1[i])
    # 把list2里有，1里没有的拼到list1上去
    if list_train2[i] not in list_train1:
        if fold_path in list_train2[i]: # 指定文件夹下的才加入
            
            # check
            # list_check.append(list_train2[i])  
            # list_label_check.append(list_train_label2[i])

            # merge 
            list_train1.append(list_train2[i])  
            list_train_label1.append(list_train_label2[i])



savelist('./train_file_list/0609_train_label.txt',list_train_label1)
savelist('./train_file_list/0609_train.txt',list_train1)

print(len(list_train1))
print(len(list_train_label1))
print(len(list_train2))
print(len(list_train_label2))
print(len(list_check))
print(len(list_label_check))