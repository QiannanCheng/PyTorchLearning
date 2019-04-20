#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#注意：多进程需要在main函数中运行
#解决:
#1.加main函数，在main中调用
#2.num_workers改为0，单进程加载
import torch
import torch.utils.data as Data
torch.manual_seed(1) #reproducible

BATCH_SIZE=5 #批训练的数据个数
x=torch.linspace(1,10,10) #torch tensor
y=torch.linspace(10,1,10)

#先转换为torch能识别的Dataset
torch_dataset=Data.TensorDataset(x,y)
#把dataset放入DataLoader
loader=Data.DataLoader(
    dataset=torch_dataset, #torch TensorDataset format
    batch_size=BATCH_SIZE,
    shuffle=True,          #要不要打乱数据
    num_workers=2,         #多线程来读数据
)

if __name__ == '__main__':
    for epoch in  range(3): #训练整套数据三次
        for step,(batch_x,batch_y) in enumerate(loader): #每一步loader释放一小批数据用来学习
            #利用小批数据进行训练
            #... ...
            #打印数据
            print('Epoch:',epoch,'|Step:',step,'|batch_x:',batch_x.numpy(),'|batch_y:',batch_y.numpy())
























