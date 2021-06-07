python test.py --config='cfgs/FaceFeatherNetA-depth.yaml' \
--resume='./checkpoints/FaceFeatherNetA_depthmap_v2_64_0.01000_sgd_train_set_21030601_20210307104514_mask_False/_60_best.pth.tar' \
--val=True --val-save=True \
--data_flag 'depth_0520' 
