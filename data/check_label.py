import os
from tqdm import tqdm

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210607_val.txt','r') as f:
    list_now = f.read().splitlines()

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210607_val_label.txt','r') as f:
    list_label = f.read().splitlines()  

print(len(list_now))
neg=0
pos=0
j=0


wrong_neg=0
wrong_pos=0

print(list_label[j])
for i in tqdm(list_now):
    # print(i)
    if ( max(i.find('neg'),i.find('fake')) > -1):
        neg=neg+1
        if(list_label[j]!='0'):
            # print("wrong here:",i)
            wrong_neg=wrong_neg+1
    elif ( max(i.find('pos'),i.find('live')) > -1):
        pos=pos+1
        if(list_label[j]!='1'):
            print("wrong here:",i)
            print("wrong label:",list_label[j])
            wrong_pos=wrong_pos+1

    else:
        print("not neg and pos:",i)
        # print("check here,",i)
    j=j+1
print("neg=",neg)
print("pos=",pos)
print("wneg=",wrong_neg)
print("wpos=",wrong_pos)
print(len(list_label))
print(len(list_now))
# with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_train.txt','w') as f:
#     str='\n'
#     f.write(str.join(list_old))
