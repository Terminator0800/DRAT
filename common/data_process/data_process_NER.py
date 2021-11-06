'''
Created on 2021年1月28日

@author: Administrator
'''
import sys, os
cwd = os.path.dirname(os.getcwd())
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))
from config import run_time
import json
import pickle
import numpy as np
import random
from common.data_process.base_processor import DataProcessor
import pandas as pd
import jieba

#一个适用于多种序列标注的框架。要求训练数据已经处理为CRF++风格，训练集中已经包含了所有需要预测的实体类型。


class DataProcessorNER(DataProcessor):
    
    def __init__(self):
        super(DataProcessorNER, self).__init__()
        self.tag_to_id = {}
        self.id_to_tag = {}
    
    def get_ner_tag_id(self, ner_tag): return self.tag_to_id[ner_tag]
    
    def get_ner_tag_ids(self, ner_tags):
        return [self.tag_to_id[tag] for tag in ner_tags]
    
    def padding_ner_tag_ids(self, ner_tag_ids, max_seq_len):
        if len(ner_tag_ids)>max_seq_len: ner_tag_ids = ner_tag_ids[:max_seq_len]
        if len(ner_tag_ids)<max_seq_len: 
            ner_tag_ids += [self.tag_to_id["O"]] * (max_seq_len - len(ner_tag_ids))
        return ner_tag_ids
                        
                        
    def get_train_data(self, task_name, path_corpus, train_set_size=-1, if_demo=False):
        data_list, text_list = [], []
        sentence, ner_tags = [], []
        count = 0
        #第一阶段，收集训练语料中的实体标签，形成映射
        self.tag_to_id["O"] = 0
        with open(path_corpus, 'r', encoding="utf8") as f:
            for line in f:
                line = line.replace("\n", "")
                if len(line)==0: continue
                token, tag = line.split("\t")
                if tag not in self.tag_to_id: self.tag_to_id[tag] = len(self.tag_to_id)
        print("试题标签类型个数是", len(self.tag_to_id))
        doc_len_list = []
        #第二阶段，将语料处理成模型的输入格式
        with open(path_corpus, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if len(line)==0:
                    text_list.append(sentence)
                    doc_len_list.append(len(sentence))
                    tokens = self.tokenizer.tokenize(sentence[:run_time.MAX_TEXT_LEN-2])
                    tokens = ["[CLS]"] + tokens + ["[SEP]"]
                    char_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    char_ids = self.padding_token_ids(char_ids, run_time.MAX_TEXT_LEN)
                    seq_len = len(tokens)
                    
                    segment_ids = [0] * run_time.MAX_TEXT_LEN
                    input_mask = [1] *  seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)   
                    
                    ner_tags = ["O"] + ner_tags[:run_time.MAX_TEXT_LEN-2] + ["O"]
                    ner_tag_ids = self.get_ner_tag_ids(ner_tags)
                    ner_tag_ids = self.padding_ner_tag_ids(ner_tag_ids, run_time.MAX_TEXT_LEN)
                    
                    data_list.append([char_ids, seq_len, segment_ids, input_mask, ner_tag_ids])
                    sentence, ner_tags = [], []
                    count +=1 
                    if train_set_size>0 and count==train_set_size: break
                    
                slices = line.strip().split("\t")
                if len(slices)!=2: continue
                token, ner_tag = slices
            
                sentence.append(token)
                ner_tags.append(ner_tag)
        print(task_name, "文档的平均长度是", np.mean(doc_len_list), len(data_list))
        print("############################")

        return data_list, text_list
                
                
class DataProcessorNER_ori(DataProcessor):
    
    def __init__(self):
        super(DataProcessorNER, self).__init__()
        self.load_tags()

    def load_tags(self):
        if not os.path.exists(run_time.PATH_MODEL_NER_TAG_ID): self.preprocess_tags()
        self.ner_tag_id_map = pickle.load(open(run_time.PATH_MODEL_NER_TAG_ID, 'rb'))

        self.ner_id_tag_map = {}
        for k, v in self.ner_tag_id_map.items(): self.ner_id_tag_map[v] = k

    
    def get_ner_tag_id(self, ner_tag): return self.ner_tag_id_map[ner_tag]
    
    def preprocess_tags(self, filter_event_types=False):
        
        entity_tag_id_map = {"O": 0}#实体类型标注任务是序列标注
        enitty_type_set = set({})
        with open(run_time.PATH_NER_TRAIN_CORPS, 'r', encoding="utf8") as f:
            for line in f:
                entity_type = line.strip().split("\t")[-1].split("-")[-1]
                if len(entity_type)<3: continue
                enitty_type_set.add(entity_type)
        print(enitty_type_set)
        #将所有的实体类型标签处理为序列标注的标签
        for entity_type in enitty_type_set:
            for pos in ["B-", "I-", "E-"]:
                entity_tag = pos + entity_type
                entity_tag_id_map[entity_tag] = len(entity_tag_id_map)
            entity_tag_id_map["S-" + entity_type] = len(entity_tag_id_map)
        print("实体类型标签是", len(entity_tag_id_map))
        pickle.dump(entity_tag_id_map, open(run_time.PATH_MODEL_NER_TAG_ID, 'wb'))
    
    def get_ner_tag_ids(self, ner_tags):
#         print("self.ner_id_tag_map", self.ner_id_tag_map)
#         print("ner_tags", ner_tags)
        return [self.ner_tag_id_map[tag] for tag in ner_tags]
    
    def padding_ner_tag_ids(self, ner_tag_ids, max_seq_len):
        if len(ner_tag_ids)>max_seq_len: ner_tag_ids = ner_tag_ids[:max_seq_len]
        if len(ner_tag_ids)<max_seq_len: 
            ner_tag_ids += [self.ner_tag_id_map["O"]] * (max_seq_len - len(ner_tag_ids))
        return ner_tag_ids
                        
                        
    def get_train_data(self, if_demo=False):
        data_list = []
        sentence, ner_tags = [], []
        count = 0
        with open(run_time.PATH_NER_TRAIN_CORPS, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if len(line)==0: 
                    tokens = self.tokenizer.tokenize(sentence[:run_time.MAX_TEXT_LEN-2])
                    tokens = ["[CLS]"] + tokens + ["[SEP]"]
                    char_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    char_ids = self.padding_token_ids(char_ids, run_time.MAX_TEXT_LEN)
                    seq_len = len(tokens)
                    
                    segment_ids = [0] * run_time.MAX_TEXT_LEN
                    input_mask = [1] *  seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)   
                    
                    ner_tags = ["O"] + ner_tags[:run_time.MAX_TEXT_LEN-2] + ["O"]
                    ner_tag_ids = self.get_ner_tag_ids(ner_tags)
                    ner_tag_ids = self.padding_ner_tag_ids(ner_tag_ids, run_time.MAX_TEXT_LEN)
                    
                    data_list.append([char_ids, seq_len, segment_ids, input_mask, ner_tag_ids])
                    sentence, ner_tags = [], []
                    count +=1 
                    if if_demo and count==100: break
                    
                slices = line.strip().split("\t")
                if len(slices)!=2: continue
                token, ner_tag = slices
            
                sentence.append(token)
                ner_tags.append(ner_tag)
        return data_list
                

if __name__ == '__main__':
    dp = DataProcessorNER()
    dp.get_train_data("../../data/corpus/NER/test_data")
    
    