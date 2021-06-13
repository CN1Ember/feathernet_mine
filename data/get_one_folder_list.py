
import os

def getlist(filepath):
    with open(filepath,'r') as f:
        list = f.read().splitlines()
    return list
def savelist(filename,list):
    with open(filename,'w') as f:
        str='\n'
        f.write(str.join(list))

if __name__ == 'main':

    path_train_0510 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_nir_210510_exp_20210510171724_train.txt'
    path_train_label_0510 = '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_nir_210510_exp_20210510171724_train_label.txt'
    list = getlist(path_train_0510)
    list_label = getlist(path_train_label_0510)
    
    fold_path = '20210510_f3c_nir_print'

    


