#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#tensorflow自动调用最优资源的原则
#pytorch自己强调使用gpu还是cpu来进行运算
#如何修改pytorch代码使得可以在gpu上进行加速运算
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
#!!!!!修改test data形式!!!!!
#Tensor on GPU
test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000].cuda()/255 # shape from (2000,28,28) to (2000,1,28,28) value in range(0,1)
test_y=test_data.targets[:2000].cuda()

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
#!!!!!转换cnn去CUDA!!!!!
cnn.cuda() #Moves all model parameters and buffers(缓冲区) to the GPU
print(cnn) #net architecture

#训练
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
#training and testing
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader): #分配batch data, normalize x when iterate train_loader
        #!!!!!这里有修改!!!!!
        b_x=b_x.cuda() #Tensor on GPU
        b_y=b_y.cuda() #Tensor on GPU
        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #输出accuracy
        if step%50==0:
            test_output=cnn(test_x)
            #!!!!!这里有修改!!!!!
            pred_y=torch.max(test_output,1)[1].cuda().data.numpy().squeeze() #将操作放去GPU
            accuracy=sum(pred_y==test_y.numpy())/test_y.size(0)
            print('Epoch: ',epoch,'| train loss: %.4f'%loss.data,'| test accuracy:',accuracy)
#print 10 predictions from test data
test_output=cnn(test_x[:10])
#!!!!!这里有修改!!!!!
pred_y=torch.max(test_output,1)[1].cuda().data.numpy().squeeze() #将操作放去GPU
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')

#如果有些计算还是需要在cpu上进行，比如plt的可视化
#我们需要将这些计算或者数据转移至CPU
#cpu_data=gpu_data.cpu()






