# 边界层识别
add by 黄索之:

## 摘要
整体基于pytorch_lightning实现

目前进度：

1.实验各种模型来初步完成实验

2.扩展数据集（添加噪声）

`data_utils.py` ：生成数据集
##### 参数 TRAIN_PATH , TEST_PATH, TRAIN_LABEL_PATH ,TEST_LABEL_PATH , Y_LENGTH , X_LENGTH, BATCH_SIZE


`inference.py` ：生成结果
##### 参数 NUM
`train.py` : 选取模型训练，保存模型

`models.py` ：存各种实验的模型 
##### 参数 Y_LENGTH , X_LENGTH , LR

#### 在这里写上各种model的实验注意事项和结果

loss function: crossentro 可以优化为focal 
https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py

赵老师的意见：图片处理完对每列取softmax,动态规划来做