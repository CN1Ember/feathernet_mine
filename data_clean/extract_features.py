from numpy.lib.type_check import imag
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm
# from ncnn_quantization import get_activation_value, replace
import os
from utils.utils import AverageMeter, accuracy, \
    get_logger, get_learning_rate, update_lr
import cv2
from tqdm import tqdm


live_path = '/mnt/cephfs/dataset/face_recognition/nir_data/jidian_nir_1224_clean/'
fake_path = '/mnt/cephfs/dataset/face_recognition/nir_data/jidian_nir_1224_clean'

# live_image_list = glob.glob('/mnt/cephfs/dataset/face_recognition/nir_data/jidian_nir_1224_clean/*.png')
# fake_image_list = glob.glob('/mnt/cephfs/dataset/face_recognition/nir_data/jidian_nir_1224_clean/*.png')

pretrained_path = "./model_file/checkpoint_25.pth"

def inference_store_the_face_feat_as_str(model, file_lst, store_file):

    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        normalize])
                                        
    for image_path in tqdm(file_lst):
        image = cv2.imread(image_path,1)
        # print(image.shape)
        image = train_transform(image)
        # print(image.shape)
        image = image.unsqueeze(0)
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # print(image.shape)
        
        output = model(image)
        output_list = output.detach().numpy().squeeze()
        feat_str = ','.join(list(map(str,output_list)))
        store_file.write(feat_str + '\n')


if __name__ == 'main':

    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        normalize])
    model = Mobilefacenetv2_width_wm(embedding_size=128, mask_train=False, pruning_rate=0.5)  # 128
    model.eval()
    pretrained_state = torch.load(pretrained_path)
    model_state = pretrained_state["model"]
    model.load_state_dict(model_state)
    # activation_value_list = get_activation_value("./model_file/iccv_mf_p0.5_wa.table")
    logger = get_logger("./log2", 'nir_face')
    livefile = open('./live_feat.txt','w')
    fakefile = open('./live_feat.txt','w')

    # model = replace(model, activation_value_list, logger)

    inference_store_the_face_feat_as_str(model, live_image_list, livefile)
    inference_store_the_face_feat_as_str(model, fake_image_list, fakefile)