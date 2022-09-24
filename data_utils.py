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
TRAIN_LABEL_PATH = "data_PBL/train_labels"
TEST_LABEL_PATH = "data_PBL/test_labels"
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
        #read_matrix = np.load(label_path)
        #label = read_matrix['arr_0']
        
        for i in range(len(os.listdir(image_path))):
            
            img_path = os.path.join(image_path, image_path.split('/')[1].split('_')[0].strip()+"_"+str(i).zfill(3)+".jpg")
            img=cv2.imread(img_path)
            self.image.append(img)
            
            
            label_img_path = os.path.join(label_path, label_path.split('/')[1].split('_')[0].strip()+"_label_"+str(i).zfill(3)+".png")
            label_img=cv2.imread(label_img_path,cv2.IMREAD_GRAYSCALE)
            #
            label_img  = np.where(label_img >= 1, 1, 0)


            #print(img_path,label_img_path)
            self.label.append(label_img )
        
            

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):

        img = self.image[item]
        
        label = self.label[item]
        
        return {
            "x": torch.tensor(img, dtype=torch.float),
            "y": torch.tensor(label, dtype=torch.long),#can only be float now
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
