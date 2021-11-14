'''
Created on 2021年1月28日

@author: Administrator
'''
#各个子任务的模型结构定义
#命名实体识别、事件抽取、文本分类、句子语义相似度
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time, load_models
from pytorch_pretrained_bert import BertModel, BertTokenizer

import torch.nn as nn
import torch
import random
random.seed(666)

#各个任务训练数据的批大小不一致，提升数据的利用率。
#每一轮都会学习主任务，其他任务随机学。这样可以突出主任务的重要性

class BERTBaseModel(nn.Module):
    
    def __init__(self, mode="work", with_bilstm=False):
        super(BERTBaseModel, self).__init__()
        
        print('初始化计算图')
        self.mode = mode
        self.with_bilstm = with_bilstm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("计算设备是", self.device)
        self.is_training = True if mode=="train" else False
        print("初始化硬共享的参数")
        #初始化bert
        print("预训练bert参数文件地址", run_time.PATH_BERT_BASE_DIR)
        self.bert = BertModel.from_pretrained(run_time.PATH_BERT_BASE_DIR)
        for param in self.bert.parameters():
            param.requires_grad = True
        if self.with_bilstm:
            self.encoder = nn.LSTM(input_size=768, hidden_size=run_time.LSTM_UNIT_NUM, bidirectional=True, dropout=0.3)
            self.encoder.flatten_parameters()
            self.main_model = nn.Linear(run_time.LSTM_UNIT_NUM*2*3 + 768, 2)
        else:
            self.main_model = nn.Linear(768*4, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs, print_weights=False):
        self.text_representation_of_each_task = {}
        self.seq_lens_of_each_task = {}
        token_ids, segment_ids, mask_ids, task_ids = inputs
        #使用bert表示文本
        input_representation, pooled = self.bert(token_ids, attention_mask=mask_ids, token_type_ids=segment_ids,\
                                                 output_all_encoded_layers=True, need_layer_no=12)
        input_representation = input_representation[-1]#取第k层的输出
        #input_representation = torch.cat(input_representation[-6:], 2)
        #input_representation = self.dropout(input_representation)
        if self.with_bilstm:
            input_representation, c = self.encoder(input_representation)
        # input_representation = self.dropout(input_representation)
        input_representation1 = torch.mean(input_representation, dim=1, keepdim=False)
        input_representation2, _ = torch.min(input_representation, dim=1, keepdim=False)
        input_representation3, _ = torch.max(input_representation, dim=1, keepdim=False)
        input_representation = torch.cat([input_representation1, input_representation2, input_representation3, pooled], 1)
        input_representation = self.dropout(input_representation)
        main_outputs = self.main_model.forward(input_representation)
        return main_outputs, main_outputs






