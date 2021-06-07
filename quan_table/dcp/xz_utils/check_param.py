import os
import torch

model_path = '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/int8-fine-tuning/log_ft_mobilefacenet_v1_p0.5_ms1m_v2_bs256_e18_lr0.001_step[]_p0.5_cos_lr_w_a_int8_finetune_20190723/check_point/checkpoint_017.pth'
output_path = '/home/xiezheng/programs2019/insightface_DCP/dcp/insightface_dcp_log/int8-fine-tuning/log_ft_mobilefacenet_v1_p0.5_ms1m_v2_bs256_e18_lr0.001_step[]_p0.5_cos_lr_w_a_int8_finetune_20190723/check_point/group_param_checkpoint_017.txt'
# model_path = 'H:/TPAMI_channel_pruning/finetune/face/log_aux_mobilefacenetv2_baseline_0.5width_512_arcface_iccv_emore_bs200_e36_lr0.100_step[15, 25, 31]_0.5width_arcface_new_op_20190717/check_point/checkpoint_035.pth'
# output_path = 'H:/TPAMI_channel_pruning/finetune/face/log_aux_mobilefacenetv2_baseline_0.5width_512_arcface_iccv_emore_bs200_e36_lr0.100_step[15, 25, 31]_0.5width_arcface_new_op_20190717/group_param.txt'

print(model_path)
print(output_path)

checkpoint_param = torch.load(model_path, map_location='cpu')
model_param = checkpoint_param['model']

with open(output_path, 'w+') as f:
    for key, value in model_param.items():
        if 'conv' in key:
            num_group = value.shape[0]
            # print(num_group)
            for i in range(num_group):
                max_value = value[i].abs().max().item()
                if max_value < 0.0001:
                    scale_value = 0
                else:
                    scale_value = 127 / max_value
                f.write('Key: {}, Group: {}, Max: {}, Scale: {}\n'.format(key, i, max_value, scale_value))