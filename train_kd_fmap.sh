python train_kd_fmap2.py --config='cfgs/FaceFeatherNetA-kd-mutifmap.yaml' \
--kd '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/ResNet18_0602_add_0607_64_0.01000_sgd_nir_rgb_20210603_mask_False/_62_best.pth.tar' \
--b 64 --lr 0.01 --every-decay 60 --fl-gamma 3 \
--data_flag 'nir_rgb_20210607' 