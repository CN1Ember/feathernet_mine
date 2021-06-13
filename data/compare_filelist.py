from tqdm import tqdm

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210607_train.txt','r') as f:
    list1 = f.read().splitlines()

with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_21060301_exp_20210603221606VIS_train.txt','r') as f:
    list2 = f.read().splitlines()


list_diff=[]
for i in tqdm( range(len(list1)) ):
    # multithread
    if list1[i] not in list2:
        list_diff.append(list1[i])




# with open('./compare_result.txt','w') as f:
#     str='\n'
#     f.write(str.join(list_diff))
# result1是远子哥0603有我0607没有，result2是我有你没有
with open('./compare_result2.txt','w') as f:
    str='\n'
    f.write(str.join(list_diff))