# 根据ini配置文件扩展近红外活体判别训练集和验证集
# by lidaiyuan:2020-9-6
import os
import configparser
import argparse
import random
import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-c', '--train_set_config', default='./data/train_data_config/train_set_nir_210510_exp.ini',
                    type=str, help='path dataset configuration')
parser.add_argument('-s', '--src_train_filelist', default='./data/train_file_list/exp_train_set_nir_21042701_exp_20210427212131',
                    type=str, help='src path dataset configuration')
parser.add_argument('-p', '--root_path', default='/mnt/cephfs/dataset/face_anti_spoofing_lock/fas_nir_datasets/fas_dataset_nir_20200813_mask/trainset/',
                    type=str, help='path dataset configuration')
parser.add_argument('-d', '--dst_path', default='./data/train_file_list',
                    type=str, help='path dataset configuration')
args = parser.parse_args()
config = configparser.ConfigParser()

def main():
    global args

    dataset_root = args.root_path  # 根目录
    pos_dataset_root = os.path.join(dataset_root, 'positive')
    neg_dataset_root = os.path.join(dataset_root, 'negative')
    config_path = args.train_set_config  # 配置文件目录
    src_train_filelist_path = args.src_train_filelist
    dst_path = args.dst_path
    config.read(config_path)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    config_flag = config_path.split('/')[-1].split('.')[0]
    flog = open('./log/%s_%s.log'%(config_flag,timestamp),'w+')

    print('root path:', dataset_root)
    print('positive root path:', pos_dataset_root)
    print('negative root path:', neg_dataset_root)
    print('src train filelist path:', dataset_root)

    flog.write('root path:%s\n'%dataset_root)
    flog.write('src train filelist:%s\n'%src_train_filelist_path)
    flog.write('positive root path:%s\n'%pos_dataset_root)
    flog.write('negative root path:%s\n'%neg_dataset_root)
    flog.write('src train filelist path:%s\n'%dataset_root)

    print('config path:', config_path)
    print('generate time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


    pos_dir = config['positive']['positive_expand_subdir']  # 需要过滤的样本
    neg_dir = config['negative']['negative_expand_subdir']  # 来自公司成员的子文件夹

    train_file_list = open( src_train_filelist_path+'_train.txt','r').read().splitlines()
    train_file_label_list = open( src_train_filelist_path+'_train_label.txt','r').read().splitlines()
    val_file_list = open(src_train_filelist_path+'_val.txt','r').read().splitlines()
    val_file_label_list = open(src_train_filelist_path+'_val_label.txt','r').read().splitlines()

    print('src train train-data size:',len(train_file_list))
    flog.write('src train train-data size:%d\n'%len(train_file_list))
    print('src val train-data size:',len(val_file_list))
    flog.write('src val train-data size:%d\n'%len(val_file_list))

    pos_train_data_size = 0
    neg_train_data_size = 0
    pos_val_data_size = 0
    neg_val_data_size = 0

    for i in range(len(train_file_list)):
        if train_file_label_list[i] == '1':
            pos_train_data_size = pos_train_data_size + 1
        else:
            neg_train_data_size = neg_train_data_size + 1

    for i in range(len(val_file_list)):
        if val_file_label_list[i] == '1':
            pos_val_data_size = pos_val_data_size + 1
        else:
            neg_val_data_size = neg_val_data_size + 1

    print('src positive train-data size:',pos_train_data_size)
    flog.write('src positive train-data size:%s\n'%pos_train_data_size)
    print('src negative train-data size:',neg_train_data_size)
    flog.write('src negative train-data size:%s\n'%neg_train_data_size)

    print('src positive validation size:',pos_val_data_size)
    flog.write('src positive validation size:%s\n'%pos_val_data_size)
    print('src negative validation size:',neg_val_data_size)
    flog.write('src negative validation size:%s\n'%neg_val_data_size)

    print('***************************************************************************************')
    flog.write('***************************************************************************************\n')

    if pos_dir == '':
        pos_dir_list = []
    else:
        pos_dir_list = [c.strip('\n').replace('\\', '/')
                                for c in pos_dir.split(',')]
    if neg_dir == '':
        neg_dir_list = []
    else:
        neg_dir_list = [c.strip('\n').replace('\\', '/')
                                for c in neg_dir.split(',')]
    ratio = 0.8
    new_pos_train_data_size = 0
    new_neg_train_data_size = 0
    new_pos_val_data_size = 0
    new_neg_val_data_size = 0
    for pos_subdir in pos_dir_list:
        train_num = 0
        pos_subdir_path = os.path.join(pos_dataset_root,pos_subdir)
        child_item_dirs = os.listdir(pos_subdir_path)
        random.shuffle(child_item_dirs)
        total_child_item = len(child_item_dirs)
        for child_item in child_item_dirs:
            # print(child_item)
            train_num = train_num + 1
            if train_num < ratio * total_child_item:
                child_path = os.path.join(pos_subdir_path,child_item)
                flog.write('pos train:%s\n'%child_item)
                for root,dirs,filenames in os.walk(child_path):
                    for filename in filenames:
                        filepath = os.path.join(root,filename)
                        train_file_list.append(filepath)
                        train_file_label_list.append('1')
                        new_pos_train_data_size = new_pos_train_data_size + 1
            else:
                flog.write('pos val:%s\n'%child_item)
                child_path = os.path.join(pos_subdir_path,child_item)
                for root,dirs,filenames in os.walk(child_path):
                    for filename in filenames:
                        filepath = os.path.join(root,filename)
                        val_file_list.append(filepath)    
                        val_file_label_list.append('1')      
                        new_pos_val_data_size = new_pos_val_data_size + 1      
    train_num = 0
    print(neg_dir_list)
    for neg_subdir in neg_dir_list:
        train_num = 0
        neg_subdir_path = os.path.join(neg_dataset_root,neg_subdir)
        child_item_dirs = os.listdir(neg_subdir_path)
        total_child_item = len(child_item_dirs)
        # print(total_child_item)
        random.shuffle(child_item_dirs)
        for child_item in child_item_dirs:
            # print(child_item)
            train_num = train_num + 1
            if train_num < ratio * total_child_item:
                flog.write('neg train:%s\n'%child_item)
                child_path = os.path.join(neg_subdir_path,child_item)
                for root,dirs,filenames in os.walk(child_path):
                    for filename in filenames:
                        filepath = os.path.join(root,filename)
                        train_file_list.append(filepath)
                        train_file_label_list.append('0')
                        new_neg_train_data_size = new_neg_train_data_size + 1
            else:
                child_path = os.path.join(neg_subdir_path,child_item)
                flog.write('neg val:%s\n'%child_item)
                for root,dirs,filenames in os.walk(child_path):
                    for filename in filenames:
                        filepath = os.path.join(root,filename)
                        val_file_list.append(filepath)    
                        val_file_label_list.append('0')   
                        new_neg_val_data_size = new_neg_val_data_size + 1     

    ind = [i for i in range(len(train_file_list))]
    random.shuffle(ind)
    f1 = open('%s/exp_%s_%s_train.txt'%(dst_path,config_flag,timestamp),'w+')
    f2 = open('%s/exp_%s_%s_train_label.txt'%(dst_path,config_flag,timestamp),'w+')
    f3 = open('%s/exp_%s_%s_val.txt'%(dst_path,config_flag,timestamp),'w+')
    f4 = open('%s/exp_%s_%s_val_label.txt'%(dst_path,config_flag,timestamp),'w+')
    
    for i in ind:
        f1.write('%s\n'%(train_file_list[i]))
        f2.write('%s\n'%(train_file_label_list[i]))
    
    for i in range(len(val_file_list)):
        f3.write('%s\n'%(val_file_list[i]))
        f4.write('%s\n'%(val_file_label_list[i]))
        
    print('***************************************************************************************')
    print('train-data size:',len(train_file_list)) 
    print('new positive train-data size:',new_pos_train_data_size)
    print('new negative train-data size:',new_neg_train_data_size)

    flog.write('***************************************************************************************')
    flog.write('train-data size:%d\n'%len(train_file_list))
    flog.write('new positive train-data size:%d\n'%new_pos_train_data_size)
    flog.write('new negative train-data size:%d\n'%new_neg_train_data_size)

    
    print('validation-data size:',len(val_file_list))      
    print('new positive validation size:',new_pos_val_data_size)
    print('new negative validation size:',new_neg_val_data_size)
    flog.write('validation-data size:%d\n'%len(val_file_list))
    flog.write('new positive validation size:%d\n'%new_pos_val_data_size)
    flog.write('new negative validation size:%d\n'%new_neg_val_data_size)
        
if __name__ == '__main__':
    main()

        
        
