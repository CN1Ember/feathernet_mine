import os
from tqdm import tqdm

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_train_old.txt','r') as f:
    list_old = f.read().splitlines()

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_train_label_old.txt','r') as f:
    list_label = f.read().splitlines()  

print(len(list_old))
j=0
for i in tqdm(list_old):
    if not os.path.exists(i):
        list_old.remove(i)
        list_label.pop(j)
        # print("check here,",i)
    j=j+1
print(len(list_old))

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_train.txt','w') as f:
    str='\n'
    f.write(str.join(list_old))

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_train_label.txt','w') as f:
    str='\n'
    f.write(str.join(list_label))
