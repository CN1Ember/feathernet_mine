from extract_features import inference_store_the_face_feat_as_str
from torchvision import transforms
from model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm
import cv2
import torch
# inference_store_the_face_feat_as_str(model, file_lst, store_file)
print(1)

with open(train_file_dir, 'r') as f:
    self.pos_file_dir = f.read().splitlines()
# with open(label_train_file, 'r') as f:
    # self.label_dir_train = f.read().splitlines()


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