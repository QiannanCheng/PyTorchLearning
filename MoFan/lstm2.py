#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible

#Hyper Parameters
TIME_STEP=10
INPUT_SIZE=1
LR=0.02
DOWNLOAD_MNIST=False

#RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out=nn.Linear(32,1)
    def forward(self,x,h_state):
        #r_out (batch,time_step,hidden_size)
        r_out,h_state=self.rnn(x,h_state)
        outs=[] #保存所有时间点的预测值
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state

        #instead,for simplicity,you can replace above cades by follows
        #r_out=r_out.view(-1,32)
        #outs=self.out(r_out)
        #outs=outs.view(-1,TIME_STEP,1)
        #return outs,h_state

        #or even simpler,since nn.Linear can accept inputs of any dimision
        #and return outputs with the same dimention except for the last
        #outs=self.out(r_out)
        #return outs
rnn=RNN()
print(rnn)

#训练
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.MSELoss()
h_state=None
plt.figure(1,figsize=(12,5))
plt.ion()
for step in range(100):
    start,end=step*np.pi,(step+1)*np.pi
    #sin预测cos
    steps=np.linspace(start,end,10,dtype=np.float32)
    x_np=np.sin(steps)
    y_np=np.cos(steps)
    x=torch.from_numpy(x_np[np.newaxis,:,np.newaxis]) #(1,10,1)
    y=torch.from_numpy(y_np[np.newaxis,:,np.newaxis]) #(1,10,1)
    prediction,h_state=rnn(x,h_state)
    #下一步很重要
    h_state=h_state.data #要把h_state重新包装一下才能放入下一个iteration，不然会报错
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #plotting
    plt.plot(steps,y_np,'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.draw();plt.pause(0.05)

plt.ioff()
plt.show()

