'''
函数说明:  先去torchvdata生成数据库文件 这里只做推理
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 17:00:47
'''
import numpy as np
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
%matplotlib inline
np.random.seed(1)

import torch
#去 pip install facenet_pytorch
from facenet_pytorch import MTCNN,InceptionResnetV1
# mtcnn=MTCNN(image_size=<image_size>,margin=<margin>)
##推理用facenet
resnet=InceptionResnetV1(pretrained='vggface2').eval()
##mtcnn用于截脸
mtcnn = MTCNN(keep_all='true')
import pickle
with open('database','rb') as file: 
    database = pickle.load(file)

#这里用的摄像头 按Q截一张做推理，你可以改成图片形式
import cv2
cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 

img1 = frame
# img1 = cv2.resize(img1,dsize=(64,64),interpolation=cv2.INTER_CUBIC) 
min_dist=100
inpu1 = mtcnn(img1)
with torch.no_grad():
    out1 = resnet(inpu1)

for (name, db_enc) in database.items():
    # 进行对比
    dist = np.linalg.norm(out1-db_enc)

    # 保存差异最小的那个
    if dist < min_dist:
        min_dist = dist
        identity = name

if(min_dist>0.5):
    print('数据库不存在！')
else:
    print('这个人是:{},置信度为:{}'.format(str(identity),str(min_dist)))
