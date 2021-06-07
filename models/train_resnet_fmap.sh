python train.py --config='cfgs/ResNet-18.yaml' \
 --b 64 --lr 0.0001 --every-decay 70 --fl-gamma 3 --optimize 'sgd' \
 --resume '/home/lidaiyuan/feathernet2020/FeatherNet/checkpoints/ResNet18_0602_add_0607_64_0.00010_sgd_exp_train_set_210224_exp_20210224231906_mask_False/_95_best.pth.tar' \
--data_flag 'exp_train_set_210226ADDGAN_exp_20210226220717'