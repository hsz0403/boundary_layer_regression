import numpy as np
import math
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch

h = 451
w = 1023-1
num = 10
c = 0.01

# the next function: calculate R^2, RMSE,
def get_results():
    Datas = np.zeros([3,79])
    path1 = 'data_PBL/labels1/test_labels.npz'
    read_M1 = np.load(path1)
    M1 = read_M1['arr_0']
    print(M1.shape)

    for num in range(79):
        NUM = num
        path2='lines/test_'+str(NUM)+".png"
        M2 = Image.open(path2)
        #print(M2.shape)
        data = np.array(M2)
        print(data.shape)

        List = []
        List1 = np.zeros([1023])
        List2 = []
        for i in range(1023): 
            List1[i] = M1[NUM,i]
            for j in range(451):  
                if data[j,i] > 1:
                    List.append(451-j)
                    List2.append(451-j-List1[i])
        #print(NUM)

        sum = 0
        for i in range(1023):
            sum+=List2[i]**2
        #print(sum)

        y = 0
        for i in range(1023):
            y += List[i]
        y = y/1023
        #print(y)

        sum1 = 0
        for i in range(1023):
            sum1+=(List[i]-y)**2
        #print(sum1)

        R = 1-sum/sum1
        #print('R^2 =',R)

        RMSE = 5000/451*math.pow((sum/1023),0.5)
        #print('RMSE =',RMSE)
        
        mae = 0
        for i in range(1023):
            mae+=abs(List2[i])
        MAE = mae/1023

        Datas[0,NUM] = R
        Datas[1,NUM] = RMSE
        Datas[2,NUM] = MAE

    for i in range(79):
        print(i, 'R^2 =',Datas[0,i], 'RMSE =',Datas[1,i])

    return Datas

#next is the algorithm that finds a line using DP
def findline(image): # input: output of the network, size: 451*1023
    h,w = image.shape 
    W = torch.zeros([h,w]) # this array stores the table of costs
    W1 = torch.zeros([h,w]) # this array stores the info of path, specifically, if [i,j] = k, then the path should be [k,j-1]->[i,j]
    line = []
    for j in range(w):
        #print(j)
        if (j == 0):
            for i in range(h):
                W[i,j] = image[i,j]
        else:
            for i in range(h):
                min = 100*w
                index = i
                '''for i1 in range(h):
                    t = W[i1,j-1]+c*(i1-i)*(i1-i)
                    if t <= min:
                        min = t
                        index = i1
                        #print(min)'''
                for i1 in range(i-num,i+num+1):
                    if(i1 < h and i1>=0):
                        t = W[i1,j-1]+c*(i1-i)*(i1-i)
                        if t <= min:
                            min = t
                            index = i1
                W[i,j] = min+image[i,j]
                W1[i,j] = index
                #print(i,j,W[i,j],W1[i,j])
    min1 = W[0,w-1]
    f = int(0)
    for m1 in range(h):
        if(W[m1,w-1])<min1:
            min1 = W[m1,w-1]
            f = int(m1)
            print(f,min1)
    line.append(f)
    for m2 in range(w-1):
        y = W1[f,w-m2-1]
        f = int(y)
        line.append(f)
    print(len(line))
    print(line)
    return line

def drawline(line):
    W = np.zeros([h,w],dtype = np.uint8)
    for i in range(w):
        j = int(line[w-i-1])
        for k in range(j-2,j+3):
            if (k<h and k>=0):
                W[j,i] = int(255)
    return W

def getline(image,NUM):
    line = findline(image)
    line_image = drawline(line)
    line_image1 = Image.fromarray(line_image)
    save_path = "lines/test_"+str(NUM)+".png"
    line_image1.save(save_path)
