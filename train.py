
import pytorch_lightning as pl
import pandas as pd
import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.model_selection import train_test_split
from data_utils import (BoundaryDataset, LightDataset)

from models import UNet,SegNet




if __name__ == '__main__':
    mod = UNet()# change model here       
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        dirpath='./path/UNet',
        filename='models-{epoch:02d}-{valid_loss:.3f}',
        save_top_k=3,
        mode='min')

    
    ds = LightDataset()
    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=200, callbacks=[checkpoint_callback])
    trainer.fit(model=mod, datamodule=ds)


    '''dm = MyDataModule(args)
if not is_predict:# 训练
    # 定义保存模型的callback，仔细查看后文
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    # 定义模型
    model = MyModel()
    # 定义logger
    logger = TensorBoardLogger('log_dir', name='test_PL')
    # 定义数据集为训练校验阶段
    dm.setup('fit')
    # 定义trainer
    trainer = pl.Trainer(gpus=gpu, logger=logger, callbacks=[checkpoint_callback]);
    # 开始训练
    trainer.fit(dck, datamodule=dm)
else:
    # 测试阶段
    dm.setup('test')
    # 恢复模型
    model = MyModel.load_from_checkpoint(checkpoint_path='trained_model.ckpt')
    # 定义trainer并测试
    trainer = pl.Trainer(gpus=1, precision=16, limit_test_batches=0.05)
    trainer.test(model=model, datamodule=dm)
'''
