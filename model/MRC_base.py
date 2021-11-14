'''
Created on 2021年8月6日

@author: Administrator
'''
#span预测式的阅读理解

from config import run_time
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class MRCSpan(nn.Module):
    
    def __init__(self, mode="work", with_bilstm=False):
        super(MRCSpan, self).__init__()

        print('初始化计算图')
        self.mode = mode
        self.with_bilstm = with_bilstm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("计算设备是", self.device)
        self.is_training = True if mode == "train" else False
        print("初始化硬共享的参数")
        # 初始化bert
        print("预训练bert参数文件地址", run_time.PATH_BERT_BASE_DIR)
        self.bert = BertModel.from_pretrained(run_time.PATH_BERT_BASE_DIR)
        for param in self.bert.parameters():
            param.requires_grad = True
        hidden_units = run_time.LSTM_UNIT_NUM*2*3 + 768
        if self.with_bilstm:
            self.encoder = nn.LSTM(input_size=768, hidden_size=run_time.LSTM_UNIT_NUM, bidirectional=True, dropout=0.3)
            self.encoder.flatten_parameters()
            self.fc_start = nn.Linear(hidden_units, run_time.MAX_DOC_LEN)
            self.fc_end = nn.Linear(hidden_units, run_time.MAX_DOC_LEN)
        else:
            self.fc_start = nn.Linear(hidden_units, run_time.MAX_DOC_LEN)
            self.fc_end = nn.Linear(hidden_units, run_time.MAX_DOC_LEN)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        token_ids, segment_ids, mask_ids, task_ids = inputs
        #使用bert表示文本
        input_representation, pooled = self.bert(token_ids, attention_mask=mask_ids, token_type_ids=segment_ids,\
                                                 output_all_encoded_layers=True, need_layer_no=12)
        input_representation = input_representation[-1]#取第k层的输出
        #input_representation = torch.cat(input_representation[-6:], 2)
        #input_representation = self.dropout(input_representation)
        if self.with_bilstm:
            input_representation, c = self.encoder(input_representation)

        input_representation1 = torch.mean(input_representation, dim=1, keepdim=False)
        input_representation2, _ = torch.min(input_representation, dim=1, keepdim=False)
        input_representation3, _ = torch.max(input_representation, dim=1, keepdim=False)
        input_vector = torch.cat([input_representation1, input_representation2, input_representation3, pooled], 1)

        input_vector = self.dropout(input_vector)

        start_logits = self.fc_start(input_vector)
        end_logits = self.fc_end(input_vector)
        main_outputs = [start_logits, end_logits]
        return main_outputs, main_outputs

