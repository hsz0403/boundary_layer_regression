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

NUM=23

if __name__ == '__main__':
    path=os.path.join('data_PBL/train_images',"train_"+str(NUM).zfill(3)+".jpg")
    
    model_path='path/models-epoch=57-valid_loss=0.00.ckpt'
    model = UNet.load_from_checkpoint(model_path)
    # disable randomness, dropout, etc...
    model.eval()
    img=cv2.imread(path)
    #preprocess to image
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #test_input=torch.tensor(gray_img, dtype=torch.float).unsqueeze(0)#[Y,X] to [B,Y,X]
    # predict with the model
    y_hat = model(torch.tensor(img, dtype=torch.float).unsqueeze(0))#[Y,X,3] to [B(1),Y,X,3]
    y_hat=y_hat.detach().squeeze(0).numpy().astype(int)
    print(y_hat)
    cv2.imwrite("examples/test.png", y_hat)


    #label as number
    '''
    read_M3 = np.load('data_PBL/labels1/train_labels.npz')
    M3 = read_M3['arr_0']
    label=M3[NUM]
    label=torch.tensor(label, dtype=torch.float)
    print(y_hat,label)
    '''
    
    