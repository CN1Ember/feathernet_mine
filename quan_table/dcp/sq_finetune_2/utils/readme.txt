# readme
# get activation_scale
python utils/gen_activation_table.py

# after train model with sq, then get weight_scale
python utils/gen_weight_table.py --model_path=...

# merge and adjust table_file
python utils/adjust_table.py --weight_path=...
