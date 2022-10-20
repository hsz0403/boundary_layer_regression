from turtle import position
import pytorch_lightning as pl
import pandas as pd
import cv2
import os

from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split

TRAIN_PATH = "data_PBL/train_images"
TEST_PATH = "data_PBL/test_images"
TRAIN_LABEL_PATH = "data_PBL/labels1/train_labels.npz"
TEST_LABEL_PATH = "data_PBL/labels1/test_labels.npz"
Y_LENGTH = 451
X_LENGTH = 1023
BATCH_SIZE=4


class BoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path):
        self.image = []
        self.label = []
        self.path = image_path
        self.image_size = X_LENGTH, Y_LENGTH
        #number as label only
        read_matrix = np.load(label_path)
        label_matrix = read_matrix['arr_0']
        
        for i in range(len(os.listdir(image_path))):
            
            img_path = os.path.join(image_path, image_path.split('/')[1].split('_')[0].strip()+"_"+str(i).zfill(3)+".jpg")
            img=cv2.imread(img_path)
            img_with_position=np.zeros((5,Y_LENGTH,X_LENGTH))
            
            img_with_position[0:3,:,:]=np.array(img).transpose(2, 0, 1)# [3,y,x]
            for y in range(Y_LENGTH):
                for x in range(X_LENGTH):
                    img_with_position[3,y,x]=y
                    img_with_position[4,y,x]=x
                    
            self.image.append(img_with_position)
            
            
            '''label_img_path = os.path.join(label_path, label_path.split('/')[1].split('_')[0].strip()+"_label_"+str(i).zfill(3)+".png")
            label_img=cv2.imread(label_img_path,cv2.IMREAD_GRAYSCALE)
            label_img  = np.where(label_img >= 1, 1, 0)
            '''
            #not image as label
            label_img_tensor=450-label_matrix[i,:].astype(np.int)# must 450-label to be the right target
            


            #print(img_path,label_img_path)
            self.label.append(label_img_tensor )
        
            

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):

        img = self.image[item]
        
        label = self.label[item]
       
        return {
            "x": torch.tensor(img, dtype=torch.float),
            "y": torch.tensor(label, dtype=torch.long),#can only be long when label is image else float
        }
        
class LightDataset(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):

        self.train_dataset = BoundaryDataset(TRAIN_PATH, TRAIN_LABEL_PATH)
        self.validation_dataset = BoundaryDataset(TEST_PATH, TEST_LABEL_PATH)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.validation_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False)
        return valid_loader
