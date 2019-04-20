#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F #激励函数
import matplotlib.pyplot as plt

#创建一些数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size()) #torch.rand 返回[0,1)之间的均匀分布

#画图
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

#建立神经网络
class Net(torch.nn.Module): #继承torch的Module
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__() #继承超类的__init__功能
        self.hidden=torch.nn.Linear(n_feature,n_hidden) #隐藏层线性输出
        self.predict=torch.nn.Linear(n_hidden,n_output) #输出层线性输出

    def forward(self,x): #这同时也是Module中的forward功能
        x=F.relu(self.hidden(x))
        x=self.predict(x) #不使用激励函数：输出值不能局限在某一个范围内
        return x

net=Net(n_feature=1,n_hidden=10,n_output=1)
print(net) #net的结构

plt.ion()
plt.show()

#训练网络
optimizer=torch.optim.SGD(net.parameters(),lr=0.2) #传入net的所有参数，学习率
loss_func=torch.nn.MSELoss() #预测值和真实值的误差计算公式（均方差）
for t in range(200):
    prediction=net(x)
    loss=loss_func(prediction,y)
    optimizer.zero_grad() #清空上一步的残余梯度
    loss.backward() #误差反向传播，计算新梯度
    optimizer.step() #将参数更新值施加到net的parameters上
    #可视化训练过程
    if t%5==0:
        #plot and show learning process
        plt.cla() #清除axes
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

