from PIL import Image
import numpy as np
import cv2
import os

root_path = './depth_data/2/depth/'
img_name = os.listdir(root_path)
img_list = [ *map ( lambda x:os.path.join(root_path,x), img_name ) ]

for i in img_list:
    img = Image.open(i)
    min,max = img.getextrema()
    factor = 255/(max)
    if(max<300):
        print("already normalized")
        break
    # print(factor)
    img = img.point(lambda x:x*factor)
    img = img.convert('L')
    img_np = np.array(img)
    # print(np.unique(img_np))
    # new_path = './depth_data/1/align/'
    cv2.imwrite(i,img_np.astype(np.uint8))


# print(img_list)