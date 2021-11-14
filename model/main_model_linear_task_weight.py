'''
Created on 2021年1月28日

@author: Administrator
'''
#各个子任务的模型结构定义
#命名实体识别、事件抽取、文本分类、句子语义相似度
import copy
import sys, os
#使用注意力来辅助顶层模型分配对各个辅助任务模块的权重
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time, load_models
from pytorch_pretrained_bert import BertModel
from transformers import BertTokenizerFast,AutoModel
import torch.nn as nn
import torch
import random
import numpy as np
from model.model_base import get_model_base, get_supervisor_model
random.seed(666)
   
#各个任务训练数据的批大小不一致，提升数据的利用率。
#每一轮都会学习主任务，其他任务随机学。这样可以突出主任务的重要性

class MultiTaskModel(nn.Module):
    
    def __init__(self, mode="work", main_task="text_similarity_LCQMC", supervised=False,
                 has_shared_encoder=False, init_task_attention=None,
                 update_task_weight=True, part_of_transformer="encoder"):
        super(MultiTaskModel, self).__init__()
        
        print('初始化计算图')
        self.mode = mode
        self.main_task = main_task
        self.supervised = supervised
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("计算设备是", self.device)
        self.task_id_map = {i: run_time.ALL_AUXILIARY_TASK_LIST[i]  for i in range(len(run_time.ALL_AUXILIARY_TASK_LIST))}
        print("self.task_id_map", self.task_id_map)
        self.task_name_id_map = {v: k for k, v in self.task_id_map.items()}
        print("self.task_name_id_map", self.task_name_id_map)
        model_map = get_model_base()
        self.supervisor, model_map = get_supervisor_model(main_task, model_map)#获取监督模型，only main model with high way
        print("model_base_map", model_map.keys())

        all_task_id_model_map = { task_id: model_map[run_time.ALL_AUXILIARY_TASK_LIST[task_id]] \
                                                                    for task_id in range(len(run_time.ALL_AUXILIARY_TASK_LIST))}

        self.task_id_model_map = {self.task_name_id_map[task_name]: all_task_id_model_map[self.task_name_id_map[task_name]]\
                                    for task_name in run_time.ALL_AUXILIARY_TASK_LIST}
        print("模型们的id0", list(self.task_id_model_map.keys()))
        self.model_seq = nn.ModuleList([])
        for key in self.task_id_model_map:
            self.model_seq.append(self.task_id_model_map[key])
            
        self.is_training = True if mode=="train" else False
        print("初始化硬共享的参数")
        #初始化bert
        print("预训练bert参数文件地址", run_time.PATH_BERT_BASE_DIR)
        self.part_of_transformer = part_of_transformer
        if self.part_of_transformer=="encoder":
            self.bert = BertModel.from_pretrained(run_time.PATH_BERT_BASE_DIR)
        elif self.part_of_transformer=="decoder":#https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
            self.bert = AutoModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall')#AutoModel.from_pretrained("ckiplab/gpt2-base-chinese")
        else:#https://huggingface.co/fnlp/bart-base-chinese
            from transformers import BartModel
            self.bert = BartModel.from_pretrained("fnlp/bart-base-chinese")

        for param in self.bert.parameters():
            param.requires_grad = True

        #主任务模型
        p_init = 0.1#1.0/len(self.task_id_map)
        init_t = np.log(p_init/(1 - p_init))
        self.share_rate = nn.Parameter(torch.from_numpy(np.array([init_t])))
        if init_task_attention!=None:
            self.task_weight = nn.Parameter(torch.from_numpy(np.array(init_task_attention))*0.0)
        else:
            self.task_weight = nn.Parameter(torch.from_numpy(np.array([1.0] * len(run_time.ALL_AUXILIARY_TASK_LIST) )))  # 损失函数权重。给终端

        self.update_task_weight = update_task_weight
        if update_task_weight==False:
            self.stastic_task_weight = init_task_attention
        print("开始self.task_weight", self.task_weight)
        # inputsize = run_time.LSTM_UNIT_NUM * 2 if has_shared_encoder else run_time.LSTM_UNIT_NUM * 2

        print("task_weight", self.task_weight)
        dropout = 0.3
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, print_weights=False):
        self.text_representation_of_each_task = {}
        self.segment_ids_of_each_task = {}
        token_ids, segment_ids, mask_ids, task_ids = inputs
        #print("task_ids", task_ids)
        #使用bert表示文本
        if self.part_of_transformer=="encoder":
            self.input_representation, pooled = self.bert(token_ids, attention_mask=mask_ids,\
                                         token_type_ids=segment_ids, output_all_encoded_layers=True)
            self.input_representation = self.input_representation[-1]#取第k层的输出
        elif self.part_of_transformer=='decoder':
            result = self.bert(token_ids, attention_mask=mask_ids, token_type_ids=segment_ids)
            self.input_representation = result.last_hidden_state
            pooled = result.last_hidden_state[:, 0, :]
        else:
            result = self.bert(token_ids, attention_mask=mask_ids).last_hidden_state
            self.input_representation = result
            pooled = result[:, 0, :]
            
        # self.input_representation = self.dropout(self.input_representation)
        text_representation_of_each_task = {}
        cls_representation_of_each_task = {}
        segment_ids_of_each_task = {}
        outputs_map = {}
        main_task_representation_seq = None
        main_task_seq_lens = None
        main_task_cls_representation = None
        # print("token_ids", token_ids[:, 1])
        for task_id in range(len(run_time.ALL_AUXILIARY_TASK_LIST)):
            this_task_ids = torch.LongTensor([task_id] * token_ids.shape[0]).to(self.device)

            if_equel = task_ids.eq(this_task_ids).to(self.device)
            if_equel = torch.nonzero(if_equel)
            indexes = if_equel.squeeze(1)
            if indexes.shape[0]<1: 
                continue
            this_token_ids = \
                   torch.index_select(token_ids, dim=0, index=indexes).to(self.device)
            # print("this_token_ids", this_token_ids[:, 1])
            text_representation_of_each_task[task_id] = \
                   torch.index_select(self.input_representation, dim=0, index=indexes).to(self.device)
            cls_representation_of_each_task[task_id] = \
                               torch.index_select(pooled, dim=0, index=indexes).to(self.device)

            segment_ids_of_each_task[task_id] = \
                          torch.index_select(segment_ids, dim=0, index=indexes).to(self.device)
#                           
            if task_id==self.task_name_id_map[self.main_task]:#保存主任务的数据
                main_task_representation_seq = text_representation_of_each_task[task_id]
                main_task_segment_ids = segment_ids_of_each_task[task_id]
                main_task_cls_representation = cls_representation_of_each_task[task_id]
                
            outputs, _, input_representation_from_subtask_seq, _ = \
                          self.model_seq[task_id].forward(text_representation_of_each_task[task_id], \
                                                       cls_representation_of_each_task[task_id], \
                                                               segment_ids_of_each_task[task_id])
            outputs_map[task_id] = [outputs]#一些辅助任务有多个输出

        task_weight_exp = torch.exp(self.task_weight)#/(len(self.task_weight) - 1)
        z = torch.sum(task_weight_exp[1:])
        task_weight = task_weight_exp/z
        task_weight = (1.0 / (1.0 + torch.exp(-self.share_rate))) * task_weight  # (len(self.task_weight) - 1)

        # task_weight = torch.sigmoid(self.task_weight) / self.share_rate[0]

        task_weight[0] = 1.0
        ####使用各个子任务模型抽取的特征，计算主任务的最终输出#####
        if not self.supervised:
            main_outputs = []#outputs_map[self.task_name_id_map[self.main_task]]
        else:
            features_combination = 0.0
            if main_task_cls_representation==None:
                main_outputs = []
            else:
                _, _, _, special_input_representation_main = \
                    self.model_seq[0].forward(main_task_representation_seq, \
                                                     main_task_cls_representation, main_task_seq_lens)
                features_combination = torch.relu(special_input_representation_main) * task_weight[0]
                for task_id in range(1, len(run_time.ALL_AUXILIARY_TASK_LIST)):
                    outputs, _, input_representation_from_subtask_seq, special_input_representation = \
                                       self.model_seq[task_id].forward(main_task_representation_seq, \
                                                    main_task_cls_representation, main_task_seq_lens)
                    features_combination = features_combination + task_weight[task_id]*torch.relu(special_input_representation)
                features_combination = features_combination/torch.sum(task_weight)
                main_outputs, _, input_representation, special_input_representation = self.supervisor(features_combination, None, None)

        return outputs_map, main_outputs, task_weight, self.share_rate







