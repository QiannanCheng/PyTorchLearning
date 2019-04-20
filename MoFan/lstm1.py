#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible

#Hyper Parameters
EPOCH=1
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.01
DOWNLOAD_MNIST=False

#训练数据
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
) #shape(60000,28,28) [0,1]
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
#测试数据
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)
test_x=test_data.data.type(torch.FloatTensor)[:2000]/255 #shape(2000,1,28,28) [0,1]
test_y=test_data.targets[:2000]

#RNN: LSTM
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True #input & output 会是以batch size为第一维度
        )
        self.out=nn.Linear(64,10)
    def forward(self,x):
        #h_n 保存着RNN最后一个时间步的隐状态
        #c_n 保存着RNN最后一个时间步的细胞状态
        #r_out shape(batch,time_step,output_size)
        #x shape(batch,time_step,input_size)
        r_out,(h_n,c_n)=self.rnn(x,None) #None表示h_0和c_0会用全0的状态
        #选取最后一个时间步的r_out输出
        #这里r_out[:,-1,:]的值也是h_n的值
        out=self.out(r_out[:,-1,:])
        return out
rnn=RNN()
print(rnn)

#训练
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() #the target label is not one-hotted
#training and testing
for epoch in range(EPOCH):
    for step,(x,b_y) in enumerate(train_loader): #x shape(64,1,28,28) [0,1]
        b_x=x.view(-1,28,28) #reshape x to (batch,time_step,input_size)
        ouput=rnn(b_x)
        loss=loss_func(ouput,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #输出accuracy
        if step%50==0:
            test_output=rnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy=sum(pred_y==test_y.numpy())/test_y.size(0)
            print('Epoch: ',epoch,'| train loss: %.4f'%loss.data,'| test accuracy: %.2f'%accuracy)
#print 10 predictions from test data
test_ouput=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_ouput,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')


































