from PIL import Image
import numpy as np
import cv2
import os

root_path = './depth_data/1/align/'
img_name = os.listdir(root_path)
img_list = [ *map ( lambda x:os.path.join(root_path,x), img_name ) ]
count = []
for i in img_list:
    img = Image.open(i)
    min,max = img.getextrema()
    

    # img = img.point(lambda x:x*factor)
    # img = img.convert('L')
    # img_np = np.array(img)

    
    depth = Image.open(i)
    
    hist = depth.histogram()
    depth,count = np.unique(depth,return_counts = True)

    flag_200low = False
    flag_200high = False
    #100~140（max<200-30）→ 70~110
    if (max<200 and flag_200low ==True):
        print("lower pixel less than 200")
        img = img.point(lambda x: x-30 if (x-30)>=0 else 0)
        
    #160-200 (max>200-90) → 70~110
    if (max>200 and flag_200high ==True):
        print("higher pixel more than 200")
        img = img.point(lambda x: x-90 if (x-90)>=0 else 0)
    print(np.unique(depth))
    print(count)

    # img_np = np.array(img)
    # cv2.imwrite(i,img_np.astype(np.uint8))

    # print(np.unique(img_np))
    # new_path = './depth_data/1/align/'
    

