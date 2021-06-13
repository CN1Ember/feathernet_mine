import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import sys
sys.path.insert(1,'../')
from models import *

checkpoint = torch.load('/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/FaceFeatherNetA_mutift_nir_rgb_210607/_84.pth.tar')
# print(checkpoint['model'])
# print(checkpoint['model'].keys())
remove_weight_list = []
for key in checkpoint['model'].keys():
    if key.startswith('localbk'):
        print(key)
        remove_weight_list.append(key)
print(checkpoint.keys())

for weight in remove_weight_list:
    checkpoint['model'].pop(weight)
# model.module.load_state_dict(checkpoint['model'])
# checkpoint['model'].pop('localbk.pwblock.0.weight')
# checkpoint['model'].pop('localbk.pwblock.0.weight')
# checkpoint['model'].pop('localbk.pwblock.0.weight')
# checkpoint['model'].pop('localbk.pwblock.0.weight')


# print(checkpoint['model'])
# torch.save('/home/xiezheng/lidaiyuan/feathernet_2020/FeatherNet/checkpoints/FaceFeatherNetA_nir_1021_ftmap_64_0.00010_Adam_train_set_1021_20201022024812_mask_True_train_set_1021_20201022024812_r_0.8/_63_remove_fmp.pth.tar',checkpoint)

net = FaceFeatherNet_v3()
# net = resnet18()

net.load_state_dict(checkpoint['model'])
print(net.state_dict())
torch.save(net.state_dict(), '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/checkpoints/FaceFeatherNetA_mutift_nir_rgb_210607/_84_cut.pth.tar')