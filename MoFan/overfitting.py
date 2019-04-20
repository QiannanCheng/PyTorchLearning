#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#过拟合出现的原因：
#1.数据量少 2.神经网络模型复杂，参数过多
#解决过拟合的方法
#方法1：增加数据量
#方法2：L1,L2.. regularization
#方法3：专门用于神经网络的正规化方法Dropout regularization。训练时，随机忽略一些神经元和神经的连接
#添加在Linear和activation之间 或者 activation和下一个Linear之间
import torch
import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible

N_SAMPLES=20
N_HIDDEN=300

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data
#plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
#plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
#plt.legend(loc='upper left')
#plt.ylim((-2.5, 2.5))
#plt.show()

#搭建神经网络
net_overfitting=torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1)
)
net_dropped=torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.Dropout(0.5), #drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.Dropout(0.5), #drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1)
)

#训练两个神经网络
optimizer_ofit=torch.optim.Adam(net_overfitting.parameters(),lr=0.01)
optimizer_drop=torch.optim.Adam(net_dropped.parameters(),lr=0.01)
loss_func=torch.nn.MSELoss()

plt.ion()

for t in range(500):
    pred_ofit=net_overfitting(x)
    pred_drop=net_dropped(x)
    loss_ofit=loss_func(pred_ofit,y)
    loss_drop=loss_func(pred_drop,y)
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t%10==0:
        #将神经网络转换为测试形式
        net_overfitting.eval()
        net_dropped.eval()

        #plotting
        plt.cla()
        test_pred_ofit=net_overfitting(test_x)
        test_pred_drop=net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        #将神经网络转换为训练形式
        net_overfitting.train()
        net_dropped.train()

plt.ioff()
plt.show()

