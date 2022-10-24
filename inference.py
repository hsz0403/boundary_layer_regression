import pytorch_lightning as pl
import pandas as pd
import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from data_utils import (BoundaryDataset, LightDataset)

from models import UNet,SegNet
from PIL import Image
import math
import data_processing
from data_processing import getline, get_results

#test number
#NUM=55
Y_LENGTH = 451
X_LENGTH = 1023
NOISE=10
for i in range(79):
    NUM = i

    path=os.path.join('data_PBL/test_images',"test_"+str(NUM).zfill(3)+".jpg")
        
    model_path="path/UNet/models-epoch=176-valid_loss=0.000.ckpt"



    if __name__ == '__main__':
        
        model = UNet.load_from_checkpoint(model_path)
        # disable randomness, dropout, etc...
        model.eval()
        img=cv2.imread(path)
        img_with_position=np.zeros((5,Y_LENGTH,X_LENGTH-1))
            
        
        img_with_position[0:3,:Y_LENGTH-NOISE,:]=np.array(img).transpose(2, 0, 1)[:,:Y_LENGTH-NOISE,:1022]# [3,y,x]
        for y in range(NOISE):
            img_with_position[0:3,Y_LENGTH-NOISE+y,:]=img_with_position[0:3,Y_LENGTH-NOISE-1,:]
    
        for y in range(Y_LENGTH):
            for x in range(X_LENGTH-1):
                img_with_position[3,y,x]=y
                img_with_position[4,y,x]=x
        
        #preprocess to image
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #test_input=torch.tensor(gray_img, dtype=torch.float).unsqueeze(0)#[Y,X] to [B,Y,X]
        # predict with the model
        y_path="data_PBL/test_labels/test_label_"+str(NUM).zfill(3)+".png"
        y=cv2.imread(y_path)
        #print(y)
        y_hat = model(torch.tensor(img_with_position, dtype=torch.float).unsqueeze(0))#[Y,X,3] to [B(1),Y,X,3]

###  This part need to add      
        image1 = y_hat.detach().squeeze(0)*(-1)
        print(image1,image1.shape)
        '''h,w = image1.shape
        image = torch.zeros([h,w])
        for i in range(h):
            for j in range(w):
                #a = image1[0,0,i,j]
                #b = image1[0,1,i,j]
                #image[i,j] = abs(math.pow(math.e,b)-math.pow(math.e,a))/(math.pow(math.e,a)+math.pow(math.e,b))
                
                #abs(math.pow(math.e,b)-math.pow(math.e,a))'''
        getline(image1,NUM)
###


        y_hat=y_hat.detach().squeeze(0).numpy()        
        new_y=np.zeros_like(y_hat)
        print(y_hat,y_hat.shape)
        print(y_hat[:,0].max())
        for i in range(450):
            for j in range(1022):
                if i>np.argmax(y_hat[:,j]):
                    new_y[i,j]=255
                else:
                    new_y[i,j]=0
        print(new_y)
        cv2.imwrite("examples/test_"+str(NUM)+".png", new_y)

        
        y=cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)#(Y,X) 0 or 255
        
        
        #label as number
        
        #read_M3 = np.load('data_PBL/labels1/train_labels.npz')
        #M3 = read_M3['arr_0']
        #label=M3[NUM]
        #label=torch.tensor(label, dtype=torch.float)
        #print(y_hat,label)
        
        #image_path = "examples/test_"+str(NUM)+".png"
        #image = cv2.imread(image_path)
        #print(image.shape)
    
## The next part is for R^2, RMSE, MAE
Datas = get_results()
np.savez('Results.npz',Datas)
