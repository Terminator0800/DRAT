'''
Created on 2021年1月28日

@author: Administrator
'''
from config import run_time
import torch
import torch.nn as nn

class TextClassification(nn.Module):
    
    def __init__(self, task_name, \
                                     hidden_units=run_time.LSTM_UNIT_NUM,
                                        class_num=None, highway=False, supervisor=False):
        super(TextClassification, self).__init__()
        self.task_name = task_name
        if supervisor==True:
            input_size = hidden_units*2
            dropout = 0.0
        else:
            input_size = 768
            dropout = 0.3
        self.supervisor = supervisor
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_units, bidirectional=True, num_layers=1, dropout=dropout)
        self.encoder.flatten_parameters()
        hidden_units *= 2 * 3#双向，用了3种pooling策略
        if highway==True: hidden_units += 768#增加CLS标签的向量
        self.fc = nn.Linear(hidden_units, class_num)
        self.dropout = nn.Dropout(dropout)
        self.highway = highway

    def forward(self,  input_representation_from_bert, cls_representation, segment_ids):
        special_input_representation, c = self.encoder(input_representation_from_bert)
        input_representation = special_input_representation

        input_representation1 = torch.mean(input_representation, dim=1, keepdim=False)
        input_representation2, _ = torch.min(input_representation, dim=1, keepdim=False)
        input_representation3, _ = torch.max(input_representation, dim=1, keepdim=False)
        if self.highway == True:
            input_vector = torch.cat([input_representation1, input_representation2, input_representation3, cls_representation], 1)
        else:
            input_vector = torch.cat([input_representation1, input_representation2, input_representation3], 1)
        input_vector = self.dropout(input_vector)
#         print("input_representation", input_representation.shape)
        out = self.fc(input_vector)
        return out, input_vector, input_representation, special_input_representation

