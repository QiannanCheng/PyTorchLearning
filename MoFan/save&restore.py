#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible

#fake data
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1) #shape=(100,1)
y=x.pow(2)+0.2*torch.rand(x.size()) #shape=(100,1)

#构建和保存网络
def save():
    #建网络
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.2)
    loss_func=torch.nn.MSELoss()
    #训练
    for t in range(100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #两种途径保存
    torch.save(net1,'net.pkl') #保存整个网络
    torch.save(net1.state_dict(),'net_params.pkl') #只保存网络中的参数（速度快，占内存少）
    #plot
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

def restore_net():
    #restore entire net1 to net2
    net2=torch.load('net.pkl')
    prediction=net2(x)
    # plot
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

#提取网络参数，放到新建的神经网络中
def restore_params():
    #新建net3
    net3=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    #将保存的参数复制到net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction=net3(x)
    # plot
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

#调用函数
save()
restore_net()
restore_params()
plt.show()

