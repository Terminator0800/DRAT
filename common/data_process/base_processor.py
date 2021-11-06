'''
Created on 2021年1月28日

@author: Administrator
'''
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time
import pickle
from common.pytorch_pretrained_bert import BertTokenizer


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    # if len(tokens_a) + len(tokens_b) > max_length:
    #     print("input -seq is too long")
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class DataProcessor:
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(run_time.PATH_BERT_BASE_DIR)
            
    def load_corpus(self):
        """加载语料"""
        pass
    
    def load_tags(self):
        """加载各种标签数据"""
        pass
    
    def get_token_id(self, token): 
        return self.token_id_map.get(token, self.token_id_map.get("[UNK]"))
    
    def get_train_data(self): 
        pass
    
    def get_dev_data(self):
        pass
    
    def data_into_batches(self, data_list, rander=None, batch_size=50, random_batch_size=False):
        data_batches = []
        a_batch = []
        if rander: rander.shuffle(data_list)
        for a_sample in data_list:
            a_batch.append(a_sample)
            if len(a_batch)==batch_size:
                data_batches.append(a_batch)
                a_batch = []
        if run_time.GPU_NUM==1:
            if len(a_batch)>0: data_batches.append(a_batch)#为了凑整，注释掉这一行
        return data_batches
    
    def get_token_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    
    def padding_token_ids(self, token_ids, max_seq_len):
        while len(token_ids)<max_seq_len: token_ids.append(0)
        return token_ids


if __name__ == '__main__':
    """继承BERT的词汇表，并从一份预训练的静态字向量数据中获取词汇表对应的部分"""
    dp = DataProcessor()
    dp.get_pretrained_token_embedding_vectors()
    
    