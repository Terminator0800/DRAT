'''
Created on 2021年1月28日

@author: Administrator
'''
#各个子任务的模型结构定义
#命名实体识别、事件抽取、文本分类、句子语义相似度
import sys, os
#使用注意力来辅助顶层模型分配对各个辅助任务模块的权重
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time, load_models
from pytorch_pretrained_bert import BertModel, BertTokenizer

import torch.nn as nn
import torch
from common.models import text_classification_model
import random,  pickle
import numpy as np
import torch.nn.functional as F
random.seed(666)

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.task_num = len(run_time.ALL_AUXILIARY_TASK_LIST)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, main_task_id, task_id, prediction, targets, task_weight, class_weight=None):
        task_name = run_time.ALL_AUXILIARY_TASK_LIST[task_id] if task_id < len(run_time.ALL_AUXILIARY_TASK_LIST) else run_time.ALL_AUXILIARY_TASK_LIST[0]
        if run_time.AUXILIARY_TASK_TYPE_MAP[task_name]=='NER':
            if len(prediction.size())==3: prediction = prediction.permute(0, 2, 1)
        # if run_time.AUXILIARY_TASK_TYPE_MAP[task_name]=='SPAN':
        # print(prediction)
        # print(targets)
        
        if run_time.AUXILIARY_TASK_TYPE_MAP[task_name] == 'MRC':
            # print("prediction", len(prediction), prediction[0].shape, prediction)
            # print("targets", targets.shape, task_name)
            loss= task_weight*F.cross_entropy(prediction[0], targets[:, 0]) + \
                  task_weight*F.cross_entropy(prediction[1], targets[:, 1])
        elif run_time.AUXILIARY_TASK_TYPE_MAP[task_name] in ['NER', "CLS"]:
            loss = task_weight * F.cross_entropy(prediction, targets)
        else:
            if class_weight!=None: 
                class_weight = torch.tensor(class_weight).cuda() if self.device == 'cuda' else torch.tensor(
                    class_weight)
                loss = task_weight * F.cross_entropy(prediction, targets , weight=class_weight)
            else:
                loss = task_weight * F.cross_entropy(prediction, targets)
        return loss



