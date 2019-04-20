#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#Author: Qiannan Cheng
#File discription: neural network layers
#添加注释

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as f

def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)

class EmbeddingLayer(nn.Module):
    '''Embedding class which include only an embedding layer'''

    def __init__(self,input_size,config):
        #input_size: vocab_num in vocabulary
        super(EmbeddingLayer,self).__init__()
        if config.emtraining: #training embedding layer
            self.embedding=nn.Sequential(OrderedDict([
                ('embedding',nn.Embedding(input_size,config.emsize)),
                ('dropout',nn.Dropout(config.dropout))
            ]))
        else:
            self.embedding=nn.Embedding(input_size,config.emsize)
            self.embedding.weight.requires_grad=False

    def forward(self,input_variable):
        #input_variable: (batch_size,max_query_length)
        return self.embedding(input_variable) #output: (batch_size,max_query_length,term_embedding)

    def init_embedding_weights(self,dictionary,embedding_index,embedding_dim):
        #dictionart.idx2word [word1,word2,...]
        #embedding_index {word1:vec1,word2:vec2,...}
        pretrained_weight=np.empty([len(dictionary),embedding_dim],dtype=float) #返回一个随机元素矩阵
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embedding_index:
                pretrained_weight[i]=embedding_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i]=initialize_out_of_vocab_words(embedding_dim)
        #pretrained_weight is a numpy matrix of shape (num_embeddings,embedding_dim)
        if isinstance(self.embedding,nn.Sequential):
            #第一个层，不包括dropout
            self.embedding[0].weight.data.copy_(torch.from_numpy(pretrained_weight))
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

class Encoder(nn.Module):
    "Encoder class of a sequence-to-sequence network"

    def __init__(self,input_size,hidden_size,bidirection,config):
        #input_size: emsize
        #hidden_size: nhid_query
        super(Encoder,self).__init__()
        self.config=config
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.bidirection=bidirection
        if self.config.model in ['LSTM','GRU']:
            self.rnn=getattr(nn,self.config.model)(self.input_size,self.hidden_size,self.config.nlayers,
                                                  batch_first=True,dropout=self.config.dropout,
                                                  bidirection=self.bidirection)
        else:
            try:
                nonlinearity={'RNN_TANH':'tanh','RNN_RELU':'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                options are ['LSTM','GRU','RNN_TANH','RNN_RELU']""")
            self.rnn=nn.RNN(self.input_size,self.hidden_size,self.config.nlayers,nonlinearity=nonlinearity,
                            batch_first=True,dropout=self.config.dropout,bidirection=self.bidirection)

    def forward(self,sent_variable,sent_len):
        #sent_variable: (query_num,max_query_length,emsize)
        #sent_len: (query_num) numpy_array [len1,len2,...]
        #Sort by length
        sent_len=np.sort(sent_len)[::-1] #从大到小顺序排序
        idx_sort=np.argsort(-sent_len) #下标
        idx_unsort=np.argsort(idx_sort)
        idx_sort=torch.from_numpy(idx_sort).cuda() if self.config.cuda else torch.from_numpy(idx_sort)
        sent_variable=sent_variable.index_select(0,Variable(idx_sort))

        #Handing padding in Recurrent Network
        sent_packed=nn.utils.rnn.pack_padded_sequence(sent_variable,sent_len,batch_first=True)
        sent_output=self.rnn(sent_packed)[0]
        sent_output=nn.utils.rnn.pad_packed_sequence(sent_output,batch_first=True) #(query_num,max_query_length,nhid_query*num*direction)

        #Un-sort by length
        idx_unsort=torch.from_numpy(idx_unsort).cuda() if self.config.cuda else torch.from_numpy(idx_unsort)
        sent_output=sent_output.index_select(0,Variable(idx_unsort))

        return sent_output

class EncoderCell(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self,input_size,hidden_size,bidirection,config):
        super(EncoderCell,self).__init__()
        self.config=config
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.bidirection=bidirection
        if self.config.model in ['LSTM','GRU']:
            #num_layers: Number of recurrent layers.
            #E.g.,setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
            self.rnn=getattr(nn,self.config.model)(self.input_size,self.hidden_size,self.config.nlayers,
                                                   batch_first=True,dropout=self.config.dropout,
                                                   bidirectional=self.bidirection)
        else:
            try:
                nonlinearity={'RNN_TANH':'tanh','RNN_RELU':'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn=nn.RNN(self.input_size,self.hidden_size,self.config.nlayers,nonlinearity=nonlinearity,
                            batch_first=True,dropout=self.config.dropout,bidirection=self.bidirection)

    def forward(self,input,hidden):
        #如果是lstm，hidden中包含hidden_state和cell_state
        output,hidden=self.rnn(input,hidden)
        return output,hidden

    #初始化initial state
    def init_weights(self,bsz):
        #bsz: batch_size
        weight=next(self.parameters()).data #next(): 获取迭代器的下一个值
        num_direction=2 if self.bidirection else 1
        if self.config.model=='LSTM':
            #返回两个：hidden_state和cell_state
            #new(): 构建一个具有相同数据类型的tensor，全0值,参数为tensor的维度
            return Variable(weight.new(self.config.nlayers*num_direction,bsz,self.hidden_size).zero()),Variable(
                weight.new(self.config.nlayers*num_direction,bsz,self.hidden_size).zero())
        else:
            #返回一个：hidden_state
            return Variable(weight.new(self.config.nlayers*num_direction,bsz,self.hidden_size).zero())

class DecoderCell(nn.Module):
    """Decoder class of a sequence-to-sequence network"""

    def __init__(self,input_size,hidden_size,ouput_size,config):
        #input_size: emsize
        #hidden_size: nhid_session
        #ouput_size: len(dictionary)
        super(DecoderCell,self).__init__()
        self.config=config
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.out=nn.Linear(hidden_size,ouput_size)
        if self.config.model in ['LSTM','GRU']:
            self.rnn=getattr(nn,self.config.model)(self.input_size,self.hidden_size,1,batch_first=True,
                                                   dropout=self.config.dropout)
        else:
            try:
                nonlinearity={'RNN_TANH':'tanh','RNN_RELU':'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn=nn.RNN(self.input_size,self.hidden_size,1,nonlinearity=nonlinearity,batch_first=True,
                            dropout=self.config.dropout)

    def forward(self,input,hidden):
        #input: (batch_size,1,emsize)
        #hidden: (1,batch_size,nhid_session)
        output,hidden=self.rnn(input, hidden)
        #ouput: (batch_size,output_size)
        ouput=f.log_softmax(self.out(output.squeeze(1)),1) #函数squeeze(arg): 表示第arg维的维度值为1，则去掉该维度，否则tensor不变
        return ouput,hidden
























