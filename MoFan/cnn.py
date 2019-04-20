#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible

#Hyper Parameters
EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

#Mnist
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)
#批训练 50samples, 1 channel, 28x28(50,1,28,28)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
#只测试前2000个数据
test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255 # shape from (2000,28,28) to (2000,1,28,28) value in range(0,1)
test_y=test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential( #(1,28,28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 #如果想要con2d出来的图片长宽没有变化，padding=(kernel_size-1)/2 当stride=1
            ), #(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #(16,14,14)
        )
        self.conv2=nn.Sequential( #(16,14,14)
            nn.Conv2d(16,32,5,1,2), #(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) #(32,7,7)
        self.out=nn.Linear(32*7*7,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) #展平多维的卷积图形(batch_size,32*7*7)
        output=self.out(x)
        return output

cnn=CNN()
print(cnn) #net architecture

#训练
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
#training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader): #分配batch data, normalize x when iterate train_loader
        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #输出accuracy
        if step%50==0:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy=sum(pred_y==test_y.numpy())/test_y.size(0)
            print('Epoch: ',epoch,'| train loss: %.4f'%loss.data,'| test accuracy:',accuracy)
#print 10 predictions from test data
test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')

