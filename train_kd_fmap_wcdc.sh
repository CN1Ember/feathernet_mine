python train_kd_fmap2_wcdc.py \
--arch=CDCFeatherNetA \
--config='cfgs/CDCFeatherNetA-v1.yaml' \
--kd '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/best_model/_21_best.pth.tar' \
--b 64 --lr 0.01 --every-decay 60 --fl-gamma 3 \
--data_flag 'exp_train_set_21060301_exp_20210603221606NIR' 