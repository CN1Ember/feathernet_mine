

filename='/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/exp_train_set_21060301_exp_20210603221606NIR_train_label.txt'

with open (filename,'r') as f:
    a=f.readlines()
    print('0 negtive,1 positive')
    print('a[1] is',a[1])
    print('all num is ',len(a))
    num1=a.count(a[1])
    print('num of a[1]',a.count(a[1]))
    print('rest',len(a)-num1)
    f.close()
print("read_ok")