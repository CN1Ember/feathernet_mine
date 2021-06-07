import os
from tqdm import tqdm


root_path = '/mnt/cephfs/dataset/face_anti_spoofing_lock/fas_nir_datasets/fas_dataset_nir_20200813_mask/testset/nir/'
pos_path = os.path.join(root_path,'positive')
neg_path = os.path.join(root_path,'negative')
pos_img_list=[]
neg_img_list=[]

queue = []

# 获取当前目录下所有图片并拼成文件路径
def get_all_imgpath(path):
    queue.append(path)
    seen = set()
    seen.add(path)#防止重复访问
    img_list=[]
    # 起点path入队
    while(len(queue)>0):
        path_now = queue.pop(0)
        # print(path_now)
        path_list  = os.listdir(path_now)

        for i in path_list:
            i=os.path.join(path_now,i)
            if os.path.isdir(i):
                if not 'bmp' in i:
                    queue.append(i) 
            elif os.path.isfile(i):
                img_list.append(i)
    print(len(img_list))
    # print(img_list[:5])
    return img_list

# 写个递归，只要不是图片就把路径存下来，继续往下直到图片为止


def merge_filelist(list_new,list_old):
    # 整合filelist，去重
    # print(‘3’ in list) 输出结果为True；
    print(len(list_old))
    for i in tqdm( range(len((list_new))) ):
        # print(list_train1[i])
        if list_new[i] not in list_old:
            list_old.append(list_new[i])
            if 'positive' in list_new[i]:
                list_label.append('1')
            elif 'negative' in list_new[i]:
                list_label.append('0')  
            else:
                print("append wrong here")    
    print("apend num = ",len(list_new))
    print("len of old list after merge is ",len(list_old)) 
    print("check len of label",len(list_label))
    return list_old

def savelist(filename,list):
    with open(filename,'w') as f:
        str='\n'
        f.write(str.join(list))

if __name__ == "__main__":    

    with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_val.txt','r') as f:
        list_old = f.read().splitlines()
    with open('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/train_file_list/nir_rgb_20210603_val_label.txt','r') as f:
        list_label = f.read().splitlines()  


    list_new = get_all_imgpath(neg_path)
    list_old = merge_filelist(list_new,list_old)
    list_new = get_all_imgpath(pos_path)
    list_old = merge_filelist(list_new,list_old)

    savelist('./train_file_list/nir_rgb_20210607_val.txt',list_old)
    savelist('./train_file_list/nir_rgb_20210607_val_label.txt',list_label)
