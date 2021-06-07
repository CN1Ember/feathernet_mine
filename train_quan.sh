python train_quan_nir.py --config='cfgs/FaceFeatherNetA-v2.yaml' \
 --b 64 --lr 0.0001 --every-decay 70 --fl-gamma 3 \
 --data_flag 'exp_train_set_nir_210510_exp_20210510171724' \
 --resume './checkpoints/FaceFeatherNetA_mutift_nir_210315_kd_0.5/_111_best_cut.pth.tar' \
 --table_path './checkpoints/table/FaceFeatherNetA_mutift_nir_210315_kd_0.5_111_best_cut.table'