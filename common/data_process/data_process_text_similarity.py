
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time
import json
import pickle
import numpy as np
from common.data_process.base_processor import DataProcessor, _truncate_seq_pair
import pandas as pd
import jieba

class DataProcessorTextSimilarity(DataProcessor):
    
    
    def _read_tsv(self, input_file):
        data = []
        with open(input_file, 'r', encoding='utf8') as f:
            for line in f:
                slices = line.strip().split("\t")
                [s1, s2, label] = slices[-3:]
                data.append([list(s1), list(s2), int(label)])
        return data
    
    def get_data(self, task_name, file_name, train_set_size=-1):
        import random
        random.seed(666)
        data_list = self._read_tsv(input_file=file_name)
        samples, text_list = [], []
        random.shuffle(data_list)
        doc_len_list = []
        for data in data_list:
            if train_set_size!=-1 and len(samples)==train_set_size: break
            text_list.append(data[0])
            text_list.append(data[1])
            doc_len_list.append(len(data[0]))
            doc_len_list.append(len(data[1]))
            _truncate_seq_pair(data[0], data[1], run_time.MAX_TEXT_LEN-3)
            tokens1 = self.tokenizer.tokenize(data[0])
            tokens2 = self.tokenizer.tokenize(data[1])
            tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)#BPE
            seq_len = len(tokens)
            segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1) + [0] * (run_time.MAX_TEXT_LEN - seq_len)  
            input_mask = [1] * seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)
            
            #token_ids = self.get_token_ids(tokens)#字符粒度的切分

            token_ids = self.padding_token_ids(token_ids, run_time.MAX_TEXT_LEN)
            label = data[2]
#             label = [0, 1] if label==1 else [1, 0]
            samples.append([token_ids, seq_len, segment_ids, input_mask, label])

        print("例子数据", samples[0] if len(samples)>0 else "none")
        print(task_name, "文档的平均长度是", np.mean(doc_len_list), len(samples))
        print("############################")
        return samples, text_list
    
    def get_train_data(self, train_set_size=-1):
        data = self.get_data(run_time.PATH_TEXT_SIMILARITY_TRAIN_DATA, train_set_size=train_set_size)
        return data
    

if __name__ == '__main__':
    dp_text_similarity = DataProcessorTextSimilarity()
    text_similarity_train_data_list = dp_text_similarity.get_data(run_time.PATH_TEXT_SIMILARITY_CCKS2018_TRAIN_DATA)
    for line in text_similarity_train_data_list:
        print(line)
    
    

