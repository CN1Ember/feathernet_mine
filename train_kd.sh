python train_kd.py \
--config='cfgs/FaceFeatherNetA-kd.yaml' \
--kd './checkpoints/ResNet18_0602_add_0607_64_0.01000_sgd_exp_train_set_nir_210510_exp_20210510171724_mask_False/_55_best.pth.tar' \
--b 64 \
--lr 0.01 \
--every-decay 70 \
--fl-gamma 3 \
--data_flag 'exp_train_set_nir_210510_exp_20210510171724' \
--gpus 0
# --resume '/home/lidaiyuan/feathernet2020/FeatherNet/checkpoints/FaceFeatherNetA_nose_nir_210216_kd_0.5_64_0.01000_exp_train_set_21030602_exp_20210318232832/_34_best.pth.tar'