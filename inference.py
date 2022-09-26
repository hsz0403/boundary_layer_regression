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

from models import ECAPA_TDNN,TDNN,UNet

#test number
NUM=55

path=os.path.join('data_PBL/test_images',"test_"+str(NUM).zfill(3)+".jpg")
    
model_path="path/UNet/models-epoch=32-valid_loss=0.00.ckpt"



if __name__ == '__main__':
    
    model = UNet.load_from_checkpoint(model_path)
    # disable randomness, dropout, etc...
    model.eval()
    img=cv2.imread(path)
    #preprocess to image
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #test_input=torch.tensor(gray_img, dtype=torch.float).unsqueeze(0)#[Y,X] to [B,Y,X]
    # predict with the model
    y_path="data_PBL/test_labels/test_label_"+str(NUM).zfill(3)+".png"
    y=cv2.imread(y_path)
    #print(y)
    y_hat = model(torch.tensor(img, dtype=torch.float).unsqueeze(0))#[Y,X,3] to [B(1),Y,X,3]
    y_hat=y_hat.detach().squeeze(0)

    # [2,Y,X] to process
    y_hat=y_hat.argmax(dim=0).numpy()
    
    print(y_hat,y_hat.shape)
    y_hat  = np.where(y_hat == 1, 255, 0)

    print(y_hat)
    cv2.imwrite("examples/test_"+str(NUM)+".png", y_hat)

    
    y=cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)#(Y,X) 0 or 255
    
    
    #label as number
    '''
    read_M3 = np.load('data_PBL/labels1/train_labels.npz')
    M3 = read_M3['arr_0']
    label=M3[NUM]
    label=torch.tensor(label, dtype=torch.float)
    print(y_hat,label)
    '''
    
    