python train.py \
--config='cfgs/ResNet-50.yaml' \
--b 64 \
--lr 0.01 \
--every-decay 70 \
--fl-gamma 3 \
--optimize 'sgd' \
--data_flag 'exp_train_set_21060301_exp_20210603221606NIR' 
# --resume './checkpoints/ResNet18_0602_add_0607_64_0.01000_sgd_exp_train_set_nir_210510_exp_20210510171724_mask_False/_40_best.pth.tar'
