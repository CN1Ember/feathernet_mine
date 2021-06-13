python train_quan_nir.py --config='cfgs/FaceFeatherNetA-v2.yaml' \
 --b 64 --lr 0.0001 --every-decay 70 --fl-gamma 3 \
 --data_flag 'nir_rgb_20210607' \
 --resume '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/FaceFeatherNetA_mutift_nir_rgb_210607/_84_cut.pth.tar' \
 --table_path '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/table/FaceFeatherNetA_mutift_nir_rgb_210607_84_cut.table'