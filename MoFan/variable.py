#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable

tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor, requires_grad=True)
print(tensor)
print(variable)

# calculate gradient
t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)
print(t_out)
print(v_out)
v_out.backward()
# v_out=1/4*sum(variable*variable)
# d(v_out)/d(variable)=1/4*2*variable=variable/2
print(variable.grad)

# get data in Variable
# Variable2tensor & Variable2numpy
print(variable)
print(variable.data)
print(variable.data.numpy())







