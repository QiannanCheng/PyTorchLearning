#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F #激励函数
import matplotlib.pyplot as plt

#fake data
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1) #shape=(100,2)
y0=torch.zeros(100) #shape=(100,)
x1=torch.normal(-2*n_data,1) #shape=(100,2)
y1=torch.ones(100) #shape=(100,)
#合并数据
x=torch.cat((x0,x1),0).type(torch.FloatTensor) #FloatTensor=32-bit floating
y=torch.cat((y0,y1),).type(torch.LongTensor) #LongTensor=64-bit integer
#画图
#plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
#plt.show()

#建立神经网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.out=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x
net=Net(n_feature=2,n_hidden=10,n_output=2)
print(net)

#训练网络
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
#真实值: 1D Tensor (batch,)
#预测值: 2D Tensor (batch,n_classes)
loss_func=torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for i in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #可视化训练过程
    if i%2==0:
        plt.cla()
        prediction=torch.max(F.softmax(out,1),1)[1]
        pred_y=prediction.data.numpy().squeeze()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y)/200
        plt.text(1.5,-4,'Accuracy=%.2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.2)
plt.ioff() #停止画图
plt.show()

