#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
学习两个函数
=================================
1.torch.nn.utils.rnn.pack_padded_sequence(input,length,batch_first=False)
将一个填充过的变长序列压紧。
输入的形状可以是(TxBx*)，T是最长序列长度，B是barch_size，*代表任意维度（可以是0）。
如果batch_first=True，相应的input size为(BxTx*)。
Variable中保存的序列，应该是按序列长度的长短排序，长的在前，短的在后。即input[:,0]代表的是最长的序列，input[:,B-1]代表的是最短的序列。
Note:只要是维度大于2的input都可以作为这个函数的参数。通过PackedSequence对象的.data属性可以获取Variable。
参数说明：
input(Variable)-变长序列，被填充后的batch
lengths(list[int])-Variable中每个序列的长度
batch_first(bool,optional)-如果是True，input的形状应该是(BxTx*)
返回值：
一个PackedSequence对象
=================================
2.torch.nn.utils.rnn.pad_packed_sequence(sequence,batch_first=False)
填充packed_sequence。
上面提到的函数的功能是将一个填充后的变长序列压紧，这个操作和pack_padded_sequence()是相反的，把压紧的序列再填充回来。
返回的Variable的值得size是(TxBx*)，T是最长序列的长度，B是batch_size，如果batch_first=True那么返回值是(BxTx*)。
Batch中的元素将以长度的逆序排列。
参数说明：
sequence(PackedSequence)-将要被填充的batch。
batch_first(bool,optional)-如果为True，返回的数据格式为(BxTx*)。
返回值：
一个tuple，包含被填充后的序列，和batch中序列的长度列表。
=================================
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
batch_size=2
max_length=3
hidden_size=2
n_layers=1

tensor_in=torch.FloatTensor([[1,2,3],[1,0,0]]).resize_(2,3,1)
tensor_in=Variable(tensor_in) #[batch,seq,feature],[2,3,1]
seq_lengths=[3,1] #list of integers holding information about the sequence step at each batch size

#pack it
pack=nn_utils.rnn.pack_padded_sequence(tensor_in,seq_lengths,batch_first=True)

#initialize
rnn=nn.RNN(1,hidden_size,n_layers,batch_first=True)
#函数torch.randn(*sizes,out=None)->tensor
#返回一个张量，包含了从标准正太分布（均值为0方差为1）中抽取的一组随机数。
#张量的形状由参数sizes定义。
#参数说明：
#size(int...)-整数序列，定义了输出张量的形状
#out(Tensor,optional)-结果张量
h0=Variable(torch.randn(n_layers,batch_size,hidden_size))

#forward
out,_=rnn(pack,h0)

#unpack
print(out.data.size()) #[4,2]
unpacked=nn_utils.rnn.pad_packed_sequence(out,batch_first=True)
print(unpacked[0].data.size()) #[2,3,2]















