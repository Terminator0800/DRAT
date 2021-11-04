
import sys, os
from numpy.random.mtrand import shuffle
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time
import json
import pickle
import numpy as np
from common.data_process.base_processor import DataProcessor
import pandas as pd
import jieba

class DataProcessorTextClassification(DataProcessor):
    
    
    def __init__(self):
        super(DataProcessorTextClassification, self).__init__()
        self.load_tags()

    def load_tags(self):
        if not os.path.exists(run_time.PATH_TEXT_CLASS_TAG_IDS): self.preprocess_tags()
        self.class_tag_id_map = pickle.load(open(run_time.PATH_TEXT_CLASS_TAG_IDS, 'rb'))

        self.class_id_tag_map = {}
        for k, v in self.class_tag_id_map.items(): self.class_id_tag_map[v] = k
    
    def preprocess_tags(self):
        class_tag_id_map = {}
        for label, _ in self._read_tsv(run_time.PATH_TEXT_CLASSIFICATION_TRAIN_DATA):
            if label not in class_tag_id_map: class_tag_id_map[label] = len(class_tag_id_map)
            
        print("文本类型个数是", len(class_tag_id_map))
        pickle.dump(class_tag_id_map, open(run_time.PATH_TEXT_CLASS_TAG_IDS, 'wb'))
        
    def _read_tsv(self, input_file):
        data = []
        with open(input_file, 'r', encoding='utf8') as f:
            for line in f:
                slices =  line.strip().split("\t")
                if len(slices)!=2: continue
                [label, text] = slices
                if len(text)<5: continue
                data.append([label, text.strip()])
        return data
    
    def get_data(self, task_name, file_name, standard_format, train_set_size=-1):
        
        data_list = self._read_tsv(input_file=file_name)
        import random
        random.seed(666)
        random.shuffle(data_list)
        doc_len_list = []
        samples, text_list = [], []
        for data in data_list:
            if train_set_size!=-1 and len(samples)==train_set_size: break
#             print("分类数据", count)
            text_list.append(data[1])
            doc_len_list.append(len(data[1]))
            text_cut = data[1][: run_time.MAX_TEXT_LEN - 2]
            tokens = self.tokenizer.tokenize(text_cut)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            seq_len = len(tokens)
            segment_ids = [0] * run_time.MAX_TEXT_LEN
            input_mask = [1] *  seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)   
            token_ids = self.padding_token_ids(token_ids, run_time.MAX_TEXT_LEN)
            if len(token_ids)<2: print(data)
            if not standard_format:
                class_label = self.class_tag_id_map[data[0]]
            else:
                class_label = int(data[0])
            samples.append([token_ids, seq_len, segment_ids, input_mask, class_label])
        print(task_name, "文档的平均长度是", np.mean(doc_len_list), len(samples))
        print("############################")

        return samples, text_list
    
    def get_train_data(self, task_name, file_name=run_time.PATH_TEXT_CLASSIFICATION_TRAIN_DATA, train_set_size=-1, standard_format=False):
        data = self.get_data(task_name, file_name, standard_format, train_set_size=train_set_size)
        return data

if __name__ == '__main__':
    dp = DataProcessorTextClassification()
    
    

