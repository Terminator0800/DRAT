'''
Created on 2021年8月6日

@author: Administrator
'''
#span预测式的阅读理解

from config import run_time
import torch
import torch.nn as nn

class MRCSpan(nn.Module):
    
    def __init__(self, task_name, mode="train", \
                        hidden_units=run_time.LSTM_UNIT_NUM,
                        highway=False, supervisor=False):
        
        super(MRCSpan, self).__init__()
        if supervisor: input_size = 100
        else: input_size=768
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_units, \
                              bidirectional=True, batch_first=True, num_layers=1, dropout=0.3)
        hidden_units *= 2#双向
        self.highway  = highway
        if self.highway==True:
            hidden_units += 768

        self.fc = nn.Linear(hidden_units, run_time.MAX_DOC_LEN)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(0.3)
        self.encoder.flatten_parameters()

    def forward(self, input_representation_from_bert, cls_representation, segment_ids):
        special_input_representation, c = self.encoder(input_representation_from_bert)
        input_representation_basic = self.dropout(special_input_representation)
        if self.highway == True:
            input_representation = torch.cat([input_representation_basic, input_representation_from_bert], 2)
        else:
            input_representation = input_representation_basic
        start_logits = self.fc(input_representation)[:, :run_time.MAX_DOC_LEN, :]
        out = [start_logits[:, :, 0], start_logits[:, :, 1]]
        return out, None, input_representation_basic, special_input_representation

