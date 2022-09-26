
import pytorch_lightning as pl
import pandas as pd
import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.model_selection import train_test_split
from data_utils import (BoundaryDataset, LightDataset)

from models import ECAPA_TDNN,TDNN,UNet




if __name__ == '__main__':
    mod = UNet()# change model here       
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='./path/UNet',
        filename='models-{epoch:02d}-{valid_loss:.3f}',
        save_top_k=3,
        mode='min')

    
    ds = LightDataset()
    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=1000, callbacks=[checkpoint_callback])
    trainer.fit(model=mod, datamodule=ds)
