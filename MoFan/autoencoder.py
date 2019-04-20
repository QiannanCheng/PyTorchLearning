#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
#3D plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

#超参数
EPOCH=10
BATCH_SIZE=64
LR=0.005
DOWNLOAD_MNIST=False
N_TEST_IMG=5

#Mnist digits dataset
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)#shape(60000,28,28) [0,1]
#batch shape (64,1,28,28)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #压缩
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3) #压缩为3个特征
        )
        #解压
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid() #激励函数让输出值在(0,1)
        )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded
autoencoder=AutoEncoder()

#训练
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func=nn.MSELoss()
for epoch in range(EPOCH):
    for step,(x,b_label) in enumerate(train_loader):
        #x shape(batch_size,1,28,28)
        b_x=x.view(-1,28*28)
        b_y=x.view(-1,28*28)
        encoded,decoded=autoencoder(b_x)
        loss=loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#3D可视化
view_data=train_data.data[:200].view(-1,28*28).type(torch.FloatTensor)/255
encoded_data,_=autoencoder(view_data)
fig=plt.figure(2)
ax=Axes3D(fig)
X=encoded_data.data[:,0].numpy()
Y=encoded_data.data[:,1].numpy()
Z=encoded_data.data[:,2].numpy()
values=train_data.targets[:200].numpy()
for x,y,z,s in zip(X,Y,Z,values):
    c=cm.rainbow(int(255*s/9)) #上色 [0,9]->[0,1]->[0,255]
    ax.text(x,y,z,s,backgroundcolor=c) #标位子
ax.set_xlim(X.min(),X.max())
ax.set_ylim(Y.min(),Y.max())
ax.set_zlim(Z.min(),Z.max())
plt.show()

























