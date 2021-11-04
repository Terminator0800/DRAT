'''
Created on 2021年7月30日

@author: Administrator
'''
'''
Created on 2021年1月28日

@author: Administrator
'''
from config import run_time
import torch
import torch.nn as nn

class NERModel(nn.Module):
    
    def __init__(self, task_name, \
                                     hidden_units=run_time.LSTM_UNIT_NUM,
                            highway=False, class_num=None, supervisor=False):
        """
        class_num需要实现统计语料得到
        """

        super(NERModel, self).__init__()
        dropout = 0.3
        if supervisor: input_size = 100
        else: input_size=768
        self.highway  = highway

        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_units, \
                              bidirectional=True, batch_first=True, num_layers=1, dropout=dropout)
        hidden_units *= 2#双向
        if self.highway==True:
            hidden_units += 768
        self.encoder.flatten_parameters()
        print(task_name, "的特征宽度", hidden_units)
        self.fc = nn.Linear(hidden_units, class_num)#使用简单的线性映射来获得实体类型标签logits;
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_representation_from_bert, cls_representation, segment_ids):

        input_representation_basic, c = self.encoder(input_representation_from_bert)
        input_representation_special = self.dropout(input_representation_basic)
#             print("共享编码器", input_representation.size())
        if self.highway == True:
            input_representation = torch.cat([input_representation_special, input_representation_from_bert], 2)
        else:
            input_representation = input_representation_special
        input_representation = self.dropout(input_representation)
        out = self.fc(input_representation)
        return out, None, input_representation_special, input_representation_basic

