from extract_features import inference_store_the_face_feat_as_str
from torchvision import transforms
from model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm
import cv2
import torch

print(1)

#prepare postiva data
pos_val_dir= '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/pos_filelist/21060301_pos_val.txt'
pos_train_dir= '/mnt/cephfs/home/chenguo/code/FAS/feathernet2021/feathernet_mine/data/pos_filelist/21060301_pos_train.txt'

with open(pos_val_dir, 'r') as f:
    pos_file_dir = f.read().splitlines()
with open(pos_train_dir, 'r') as f:
    pos_file_dir_train = f.read().splitlines()
# pos_file_dir = pos_file_dir[:100]
print(len(pos_file_dir))

# prepare feature extract model
pretrained_path = "./model_file/checkpoint_25.pth"
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
train_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    normalize])
model = Mobilefacenetv2_width_wm(embedding_size=128, mask_train=False, pruning_rate=0.5)  # 128
model.eval()
pretrained_state = torch.load(pretrained_path)
model_state = pretrained_state["model"]
model.load_state_dict(model_state)

# store val pos file
pos_feature_file = open('./val_pos_feature.txt','w')

# get features
# inference_store_the_face_feat_as_str(model, pos_file_dir, pos_feature_file)

# store train pos file

pos_feature_file_train = open('./train_pos_feature.txt','w')
inference_store_the_face_feat_as_str(model, pos_file_dir_train, pos_feature_file)
