#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#Author: Qiannan Cheng
#添加注释

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as f
from MNSRF.nn_layer import EmbeddingLayer,Encoder,EncoderCell,DecoderCell

def mask(sequence_length, seq_idx):
    #sequence_length: (batch_size*(session_length-1))
    #seq_idx interger
    batch_size = sequence_length.size(0)
    seq_range = torch.LongTensor([seq_idx])
    seq_range_expand = seq_range.expand(batch_size) #(batch_size*(session_length-1))
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    #return: [0,1,0,...] 如果元素值满足< 则为1 否则为0
    return seq_range_expand < sequence_length

class NSRF(nn.Module):
    def __init__(self,dictionary,embedding_index,args):
        #dictionary.idx2word [word1,word2,...]
        #embedding_index: {word1:vec1,word2:vec2,...}
        super(NSRF,self).__init__()
        self.config=args
        self.num_directions=2 if self.confg.bidirection else 1
        self.embedding=EmbeddingLayer(len(dictionary),self.config)
        self.embedding.init_embedding_weights(dictionary,embedding_index,self.config.emsize)
        self.query_encoder=Encoder(self.config.emsize,self.config.nhid_query,self.config.bidirection,self.config)
        self.document_encoder=Encoder(self.config.emsize,self.config.nhid_doc,self.config.bidirection,self.config)
        self.session_query_encoder=EncoderCell(self.config.nhid_query*self.num_directions,self.config.nhid_session,
                                               False,self.config)
        self.projection=nn.Sequential(OrderedDict([
            ('linear',nn.Linear(
                self.config.nhid_query*self.num_directions+self.config.nhid_session,
                self.config.nhid_doc*self.num_directions)),
            ('tanh',nn.Tanh())
        ]))
        self.decoder=DecoderCell(self.config.emsize,self.config.nhid_session,len(dictionary),self.config)

    def forward(self,session_queries,session_query_length,rel_docs,rel_docs_length,doc_labels):
        """
        Forward function of the neural click model. Return average loss for a batch of sessions.
        :param session_queries: 3d tensor [batch_size x session_length x max_query_length]
        :param session_query_length: 2d tensor [batch_size x session_length]
        :param rel_docs: 4d tensor [batch_size x session_length x num_rel_docs_per_query x max_doc_length]
        :param rel_docs_length: 3d tensor [batch_size x session_length x num_rel_docs_per_query]
        :param doc_labels: 3d tensor [batch_size x session_length x num_rel_docs_per_query]
        :return: average loss over batch [autograd Variable]
        """
        #query encoding
        #(query_num,max_query_length)->(query_num,max_query_length,emsize)
        embedded_queries=self.embedding(session_queries.view(-1,session_queries.size(-1)))
        #(query_num,max_query_length,emsize)+(query_num)->(query_num,max_query_lenght,nhid_query*num_direction)
        encoded_queries=self.query_encoder(embedded_queries,session_query_length.view(-1).data.cpu().numpy())
        #(query_num,max_query_length,nhid_query*num_direction)->(query_num,nhid_query*num_direction)
        encoded_queries=self.apply_pooling(encoded_queries,self.config.pool_type)
        #(query_num,nhid_query*num_direction)->(batch_size,session_length,nhid_query*num_direction)
        encoded_queries=encoded_queries.view(*session_queries.size()[:-1],-1)

        #document encoding
        #(doc_num,max_doc_length)->(doc_num,max_doc_length,emsize)
        embedded_docs=self.embedding(rel_docs.view(-1,rel_docs.size(-1)))
        #(doc_num,max_doc_length,emsize)+(doc_num)->(doc_num,max_doc_length,nhid_doc*num_direction)
        encoded_docs=self.document_encoder(embedded_docs,rel_docs_length.view(-1).data.cpu().numpy())
        #(doc_num,max_doc_length,nhid_doc*num_direction)->(doc_num,nhid_doc*num_direction)
        encoded_docs=self.apply_pooling(encoded_docs,self.config.pool_type)
        #(doc_num,nhid_doc*num_direction)->(batch_size,session_length,num_rel_docs_per_query,nhid_doc*num_direction)
        encoded_docs=encoded_docs.view(*rel_docs.size()[:-1],-1)

        #session level encoding
        #value(batch_size: 共有多少个session)->(1,batch_size,nhid_session)
        sess_q_hidden=self.session_query_encoder.init_weights(encoded_queries.size(0))
        #初始化sess_q_out: (batch_size,1,nhid_session)
        sess_q_out=Variable(torch.zeros(session_queries.size(0),1,self.config.nhid_session))
        if self.config.cuda:
            sess_q_out=sess_q_out.cuda()

        hidden_states,cell_states=[],[] #element shape: (1,batch_size,nhid_session)
        click_loss=0
        #loop all the queries in a session
        for idx in  range(encoded_queries.size(1)): #session_length
            #(batch_size,nhid_query*num_direction)+(batch_size,nhid_session)->(batch_size,nhid_query*num_direction+mhid_session)
            combine_rep=torch.cat((encoded_queries[:,idx,:],sess_q_out.squeeze(1)),1) #query和session的联合表示
            #(batch_size,nhid_query*num_direction+nhid_session)->(batch_size,nhid_doc*num_direction)
            combine_rep=self.projection(combine_rep)
            #(batch_size,nhid_doc*num_direction)->(batch_size,num_rel_docs_per_query,nhid_doc*num_direction)
            combine_rep=combine_rep.unsqueeze(1).expand(*encoded_docs[:,idx,:,:].size())
            #mul: (batch_size,num_rel_docs_per_query,nhid_doc*num_direction)
            #sum: (batch_size,num_rel_docs_per_query) 每一个文档的score
            click_score=torch.sum(torch.mul(combine_rep,encoded_docs[:,idx,:,:]),2)
            #binary_cross_entropy_loss在所有query上的平均
            click_loss+=f.binary_cross_entropy_with_logits(click_score,doc_labels[:,idx,:])
            #update session_level query encoder state using query representation
            #sess_q_out: (batch_size,1,nhid_session)
            #sess_q_hidden: (1,batch_size,nhid_session)
            sess_q_out,sess_q_hidden=self.session_query_encoder(encoded_queries[:,idx,:].unsqueeze(1), #(batch_size,1,nhid_query*num_direction)
                                                                sess_q_hidden) #(1,batch_size,nhid_session)
            if self.config.model=='LSTM':
                #hidden_state/cell_state: [(batch_size,nhid_session),...] 表示每一个时间点的的session-level state
                hidden_states.append(sess_q_hidden[0][-1])
                cell_states.append(sess_q_hidden[1][-1])
            else:
                #hidden_state: [(batch_size,nhid_session),...] 表示每一个时间点的session-level state
                hidden_states.append(sess_q_hidden[-1])
        #L1: 对每个doc以及每个query取loss的平均值
        click_loss=click_loss/encoded_queries.size(1) #session length
        #torch.stack([],dim) 设有i个n行y列的矩阵。dim=1是将列表中每个矩阵的第一行组成第一维矩阵。size=(n,i,y)
        #[(batch_size,nhid_session),...]list中一共有session_length个元素->(batch_size,session_length,nhid_session)
        hidden_states=torch.stack(hidden_states,1)
        #remove the last hidden states which stand for the last queries in sessions
        #contiguous(): 把tensor变成在内存中连续分布的形式
        #(batch_size,session_length-1,nhid_session)->(batch_size*(session_length-1),nhid_session)
        #->(1,batch_size*(session_length-1),nhid_session)
        hidden_states=hidden_states[:,:-1,:].contiguous().view(-1,hidden_states.size(-1)).unsqueeze(0)
        if self.config.model=='LSTM':
            #(batch_size,session_length,nhid_session)
            cell_states=torch.stack(cell_states,1)
            #(1,batch_size*(session_length-1),nhid_session)
            cell_states=cell_states[:,:-1,:].contiguous().view(-1,cell_states.size(-1)).unsqueeze(0)
            #Initialize hidden states of decoder with the last hidden states of the session encoder
            decoder_hidden=(hidden_states,cell_states) #二元组
        else:
            #Initialize hidden states of decoder with the last hidden states of the session encoder
            decoder_hidden=hidden_states #并有进行论文中提到的非线性变换

        #train the decoder for all the queries in a session except the first
        #(query_num,max_query_length,emsize)->(batch_size,session_length,max_query_length,emsize)
        embedded_queries=embedded_queries.view(*session_queries.size(),-1)
        #(batch_size*(session_length-1),max_query_length,emsize)
        decoder_input=embedded_queries[:,1:,:,:].contiguous().view(-1,*embedded_queries.size()[2:])
        #ground truth
        #(batch_size,session_length,max_query_length)->(batch_size*(session_length-1),max_query_length)
        decoder_target=session_queries[:,1:,:].contiguous().view(-1,session_queries.size(-1))
        #(batch_size*(session_length-1))
        target_length=session_query_length[:,1:].contiguous().view(-1)
        decoding_loss,total_local_decoding_loss_element=0,0
        for idx in range(decoder_input.size(1)-1): #max_query_length-1
            #用目标query的第i-1个word作为输入，输出预测query的第i个word。
            #例如，利用用目标query的第一个单词<s>作为输入，输出预测query的第一个非<s>的单词。
            #(batch_size*(session_length-1),max_query_length,emsize)
            #->(batch_size*(session_length-1),1,emsize) 去除一个session中的first_query
            input_variable=decoder_input[:,idx,:].unsqueeze(1)
            #decoder_hidden: (1,batch_szie*(session_length-1),nhid_session) 去除了一个session中的last_query
            #decoder_output: (batch_size*(session_length-1),vocab_size)
            decoder_output,decoder_hidden=self.decoder(input_variable,decoder_hidden)
            local_loss,num_local_loss=self.compute_decoding_loss(decoder_output,decoder_target[:,idx+1],idx,
                                                                 target_length,self.config.regularize)
            decoding_loss+=local_loss
            total_local_decoding_loss_element+=num_local_loss
        if total_local_decoding_loss_element>0:
            #对每个word以及每个query取loss的平均值
            decoding_loss=decoding_loss/total_local_decoding_loss_element
        return click_loss+decoding_loss

    @staticmethod #静态方法无需实例化，即可调用
    def apply_pooling(encodings,pool_type):
        #encodings: (query_num,max_query_length,nhid_query*num_direction)
        if pool_type=='max': #right
            pooled_encodings=torch.max(encodings,1)[0].squeeze()  #(query_num,nhid_query*num_direction)
        elif pool_type=='mean': #wrong: 长度不足max_query_length的一些query进行padding，mean值不准确
            pooled_encodings=torch.sum(encodings,1).squeeze()/encodings.size(1) #(query_num,nhid_query*num_direction)
        elif pool_type=='last': #wrong: 长度不足max_query_length的一些query进行padding，last多为padding值
            pooled_encodings=encodings[:,-1,:] #(query_num,nhid_query*num_directions)
        return pooled_encodings

    @staticmethod
    def compute_decoding_loss(logits,target,seq_idx,length,regularize):
        """
        Compute negative log-likeihood loss for a batch of predictions.
        :param logits: decoder_ouput 2d (batch_size*(session_length-1),vocab_size)
        :param target: decoder_target[:,idx+1] 1d (batch_size*(session_length-1))
        :param seq_idx: idx interger
        :param length: target_length 1d (batch_size*(session_length-1))
        :return: total loss over the input minibatch [autograd Variable] and number of elements(预测的单词)
        """
        #L2: (batch_size*(session_length-1)) 每一个元素代表 -1*log(P(输出目标word))
        losses=-torch.gather(logits,dim=1,index=target.unsqueeze(1)).squeeze()
        #maskk: (batch_size*(session_length-1)) 元素为0/1
        maskk=mask(length,seq_idx)
        #(batch_size*(session_length-1))
        #若idx>=query_length，元素为预测每个query的第idx+1个单词所得loss，否则为0
        losses=losses*maskk.float()
        num_non_zero_elem=torch.nonzero(mask.data).size() #(non_zero_num,1)
        if regularize:
            #LR: (batch_size*(session_length-1))
            regularized_loss=logits.exp().mul(logits).sum(1).squeeze()*regularize
            #缺少：regularized_loss=regularized_loss*maskk.float()
            #在预测所有query第idx+1个单词时产生的loss和正则化项之和
            loss=losses.sum()+regularized_loss.sum()
            if not num_non_zero_elem:
                return loss,0
            else:
                return loss,num_non_zero_elem[0] #non_zero_num(计算loss时涉及的query数目)
        else:
            if not num_non_zero_elem:
                #在预测所有query第idx+1个单词时产生的loss之和
                return losses.sum(),0
            else:
                return losses.sum(),num_non_zero_elem[0] #non_zero_num(计算loss时涉及的query数目)

