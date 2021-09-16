'''
函数说明:  生成数据库文件database
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 16:58:29
'''
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
import scipy.misc
%matplotlib inline
np.random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
# from fr_utils import *
# from inception_blocks_v2 import *
from facenet_pytorch import MTCNN,InceptionResnetV1
# mtcnn=MTCNN(image_size=<image_size>,margin=<margin>)
resnet=InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all='true')
import os
a = os.path.join(os.path.dirname(__file__), './images')
os.listdir(a)
#改成你自己的 写成遍历什么的方式都行，我图省事就直接撸的
import cv2
img_path1 = r'images\你的图片1.JPG'
img1 = cv2.imread(img_path1)
img_path2 = r'images\你的图片2.JPG'
img2 = cv2.imread(img_path2)
img_path3 = r'images\你的图片3.JPG'
img3 = cv2.imread(img_path3)
img_path4 = r'images\你的图片4.JPG'
img4 = cv2.imread(img_path4)
img_path5 = r'images\你的图片5.JPG'
img5 = cv2.imread(img_path5)
img_path6 = r'images\你的图片6.PNG'
img6 = cv2.imread(img_path6)
imgdi = {'hongqing':img1,'liuyutao':img2,'liuxiaojun':img3,'qiaorui':img4,'zhangbo':img5,'hongqingkouzhao':img6}
for img in imgdi:
    imgdi[img]=mtcnn(img1)


res = {}
with torch.no_grad():
    for inpu in imgdi:
        out = resnet(imgdi[inpu])
        res[inpu] = out

import pickle
with open('database','wb') as file: 
    pickle.dump(res,file)
