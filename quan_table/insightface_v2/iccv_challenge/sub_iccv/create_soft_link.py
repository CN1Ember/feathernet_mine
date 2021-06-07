import os

src_path = '/mnt/ssd/Datasets/faces/ms1m-retinaface-t1/imgs_mxnet'
dst_path = '/mnt/ssd/Datasets/faces/ms1m-retinaface-t1/sample_imgs_1000_new'

txt_path = '/mnt/ssd/Datasets/faces/ms1m-retinaface-t1/sample_imgs_1000.txt'

with open(txt_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line_str = line.split('/')

        src_dir_path = os.path.join(src_path, line_str[-2], line_str[-1])
        dst_dir_path = os.path.join(dst_path, line_str[-2], line_str[-1])
        if not os.path.exists(os.path.join(dst_path, line_str[-2])):
            os.makedirs(os.path.join(dst_path, line_str[-2]))
        print(f'src_dir_path={src_dir_path}, dst_dir_path={dst_dir_path}')
        # os.symlink(src_dir_path, dst_dir_path)
        os.link(src_dir_path, dst_dir_path)