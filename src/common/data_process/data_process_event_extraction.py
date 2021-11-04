'''
Created on 2019年11月21日

@author: Administrator
'''
#将数据集处理为实体识别+实体关系分类的联合任务训练语料。
#输入使用3部分特征:
#(1)字向量；
#（2）ner标签，采用BIOES标签体系；用实体类型个数*5个标签表示出实体类型和位置
#（3）实体位置，这里使用10维向量表示出entity1的BIOES和entity2的BIOES

#输出使用onehot encoding表示关系类别
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from config import run_time
import json
import pickle
import numpy as np
from common.data_process.base_processor import DataProcessor
import pandas as pd
import jieba

#将文件中的数据读取出来，转换为训练可用的格式
class DataProcessor4EventExraction(DataProcessor):
    
    def __init__(self):
        super(DataProcessor4EventExraction, self).__init__()
        self.load_tags()

    def load_tags(self):
        if not os.path.exists(run_time.PATH_EVENT_EXTRACTION_TAG_ID_MAP):
            self.preprocess_tags()
        id_info_map = pickle.load(open(run_time.PATH_EVENT_EXTRACTION_TAG_ID_MAP, 'rb'))
        self.ner_tag_id_map = id_info_map['entity_tag_id_map']
        print("实体类型标签个数是", len(self.ner_tag_id_map))
        self.event_type_set = set(id_info_map["event_types"])
        print("事件类型个数是", len(self.event_type_set))
        self.entity_name_max_len  = id_info_map['entity_name_max_len']

        self.ner_id_tag_map = {}
        for k, v in self.ner_tag_id_map.items(): self.ner_id_tag_map[v] = k

    
    def get_ner_tag_id(self, ner_tag): 
        return self.ner_tag_id_map[ner_tag]
    
    def preprocess_tags(self, filter_event_types=False):
        #加载需要使用的事件类型
        activated_event_type_set = set({})
        df = pd.read_excel(run_time.PATH_EVENT_TYPE_STATUS, header=0)
        for i in range(df.shape[0]):
            line = df.iloc[i]
            event_type, status = line["类别"] + "-" + line["事件类型"], line["是否激活"]
            if status==1: activated_event_type_set.add(event_type)
        print("activated_event_type_set", len(activated_event_type_set), activated_event_type_set)
        
        entity_tag_id_map = {"O": 0}#实体类型标注任务是序列标注
        event_type_set = set({})
        argument_type_set = set({})
        entity_name_max_len = 0
        for data in self.load_corpus(file_name=run_time.PATH_EVENT_EXTRACTION_TRAIN_CORPUS):
            for event in data['event_list']:
                event_type = event['event_type']
                if run_time.IF_FILTER_EVENT_TYPES_IN_TRAINING and event_type not in activated_event_type_set: continue
                
                if len(event['trigger'])>entity_name_max_len: entity_name_max_len = len(event["trigger"])
                event_type_set.add(event_type)
                for argument in event["arguments"]:
                    if len(argument['argument'])>entity_name_max_len: 
                        entity_name_max_len = len(argument["argument"])
                    argument_type = argument['role']
                    argument_type_set.add(argument_type)
        
        #将所有的实体类型标签处理为序列标注的标签
        for entity_type in list(event_type_set | argument_type_set):
            for pos in ["B~", "I~", "E~"]:
                entity_tag = pos + entity_type
                entity_tag_id_map[entity_tag] = len(entity_tag_id_map)
            entity_tag_id_map["S~" + entity_type] = len(entity_tag_id_map)
        print('event_type_set', len(event_type_set), "argument_type_set", len(argument_type_set))
        print("实体类型标签是", len(entity_tag_id_map))
        print("实体名称的最大长度是", entity_name_max_len)
        pickle.dump({'entity_tag_id_map': entity_tag_id_map, \
                     "event_types": list(event_type_set), 
                     'entity_name_max_len': entity_name_max_len}, open(run_time.PATH_EVENT_EXTRACTION_TAG_ID_MAP, 'wb'))
    

    def load_corpus(self, file_name=None):
        if file_name==None: file_name = run_time.PATH_TRAIN_CORPUS_FILE
        with open(file_name, 'r', encoding='utf8') as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            data = json.loads(line)
            data_list.append(data)
        return data_list

    def preprocess_a_batch(self, a_batch):
        training_batch = [[], [], [], [], [], []]
        for index_in_batch in range(len(a_batch)):
            a_sample = a_batch[index_in_batch]
            for i in range(len(a_sample)-1):
                training_batch[i].append(a_sample[i])
        return training_batch
    
    def forward_mm_segment(self, text, vocab, entity_name_max_len):
        words = []
        text_len = len(text)
        i = 0
        while i < len(text):
            a_window = text[i: min(i + entity_name_max_len, text_len)]
            if_word_here = False
            for j in range(len(a_window), -1, -1):
                temp_str = a_window[0: j]
                if (temp_str, i) in vocab:
                    words.append(temp_str)
                    i += len(temp_str)
                    if_word_here = True
                    break
            if not if_word_here:
                words.append(text[i])
                i += 1
        return words
    
    
    #基于文本和事件标签，得到ner标注序列，以及触发词与事件元素的关系
    def feature_preprocess(self, text, event_list, max_seq_len):
        text_cut = text[:max_seq_len-2]
        tokens = self.tokenizer.tokenize(text_cut)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        char_ids = self.get_token_ids(tokens)
        char_ids = self.padding_token_ids(char_ids, max_seq_len)
        text_cut = "#" + text_cut + "#"
        seq_len = len(tokens)
        
        segment_ids = [0] * max_seq_len
        input_mask = [1] *  seq_len + [0] * (max_seq_len - seq_len)       

        #处理NER标签
        entity_type_map_here = {}#实体-类型信息
        trigger_argument_map = {}#triger与argument的对应关系
        ner_tag_ids = []
        for event in event_list:#每一个实体关系，都需要生成一条数据
            trigger_word, event_type, trigger_start_index = event['trigger'], event["event_type"], \
                                       event["trigger_start_index"] + 1#考虑到文本前后加了占位符，start_index需要加1
            entity_type_map_here[(trigger_word, trigger_start_index)] = event_type#使用在文本中的start_index辅助去重，避免同名实体的影响
 
            for argument in event["arguments"]:
                argument_name, argument_role, argument_start_index = argument["argument"], \
                                       argument["role"], argument["argument_start_index"] + 1
                entity_type_map_here[(argument_name, argument_start_index)] = argument_role
                if (trigger_word, trigger_start_index) not in trigger_argument_map: \
                    trigger_argument_map[(trigger_word, trigger_start_index)] = set({})
                trigger_argument_map[(trigger_word, trigger_start_index)].add((argument_name, argument_start_index))#考虑到事件元素可能重复，也带上start_index
            
        ##############开始制作实体类型标签#############
        entity_max_len = max(list(map(lambda x: len(x[0]), entity_type_map_here.keys()))) if len(entity_type_map_here)>0 else 1
#         print("entity_max_len", entity_max_len)

        #这里获得的是原始文本中的分词结果和位置，因此后面需要针对cls标签做加一操作
        words = self.forward_mm_segment(text_cut, entity_type_map_here, entity_max_len)

        word_head_index = 0#每个word末位字符的index
        for word in words:
            if (word, word_head_index) not in entity_type_map_here:
                for _ in word:
                    ner_tag_ids.append(0)
            else:
                entity_type = entity_type_map_here[(word, word_head_index)]
                if len(word)==1: ner_tag_ids.append(self.ner_tag_id_map.get("S~" + entity_type, 0))
                if len(word)==2:
                    ner_tag_ids.append(self.ner_tag_id_map.get("B~" + entity_type, 0))
                    ner_tag_ids.append(self.ner_tag_id_map.get("E~" + entity_type, 0))
                if len(word)>2:
                    ner_tag_ids.append(self.ner_tag_id_map.get("B~" + entity_type, 0))
                    for _ in range(len(word)-2):
                        ner_tag_ids.append(self.ner_tag_id_map.get("I~" + entity_type, 0))
                    ner_tag_ids.append(self.ner_tag_id_map.get("E~" + entity_type, 0))
            word_head_index += len(word)
        ##############完成制作实体类型标签#############                
#         print('ner_tag_ids', ner_tag_ids)#某种bug导致ner 标签全为0
#         print("text", text)
#         print('words', words)
#         print("entity_type_map_here", entity_type_map_here)
        ###########开始制作trigger与argument的对应关系矩阵################生成实体关系分布矩阵，维度分别对应：subject的tail位置,object的tail位置，关系种类
        relation_distribution_sparse_indexes = [0]*max_seq_len#如果一个token不是事件元素，指向<start>
        #样本在一个batch中的序号初始化为None，在后面统一处理为实际数字
        pair_set = set({})
        for (trigger_word, trigger_start) in trigger_argument_map:
            for (argument_name, argument_start) in trigger_argument_map[(trigger_word, trigger_start)]:
                if (trigger_start, argument_start) not in pair_set and \
                              trigger_start<max_seq_len and argument_start<max_seq_len:
                    pair_set.add((trigger_start, argument_start))
                else: continue

                argument_start_new = argument_start
                trigger_start_new = trigger_start
                if argument_start_new < len(relation_distribution_sparse_indexes) and \
                      trigger_start_new < len(relation_distribution_sparse_indexes):
                    relation_distribution_sparse_indexes[argument_start_new] = trigger_start_new

#         print('relation_distribution_sparse_indexes', relation_distribution_sparse_indexes)
        if len(ner_tag_ids)<max_seq_len: 
            ner_tag_ids += [self.ner_tag_id_map["O"]] * (max_seq_len - len(ner_tag_ids))

        a_sample = [char_ids, seq_len, segment_ids, input_mask, ner_tag_ids, relation_distribution_sparse_indexes]
        return a_sample

    def convert_index_from_char_style_to_BPE(self, tokens, entity_index_name_type_map, trigger_argument_map):
        entity_index_name_type_map_new, trigger_argument_map_new = {}, {}
        special_marks = {"[UNK]", "[SEP]", "[CSL]", "[MARK]"}
        index_delta = 0
        char_style_index = 0
        char_style_index_to_bert_style_map = {}#记录字符风格实体词位置到bert风格实体词位置的映射
        #获取bert风格的实体词语起始位置
        for index_in_bert in range(len(tokens)):
            token = tokens[index_in_bert]
#             print("char_style_index", char_style_index, "index_in_bert", index_in_bert, token, "index_delta", index_delta)
            if char_style_index in entity_index_name_type_map:#如果到了一个实体词处
                bert_style_index = char_style_index - index_delta#计算bert风格的实体词起始位置
                entity_index_name_type_map_new[bert_style_index] = entity_index_name_type_map[char_style_index]#存储
                char_style_index_to_bert_style_map[char_style_index] = bert_style_index
#                 print("char_style_index", char_style_index, "index_in_bert", index_in_bert, "index_delta", index_delta)
  
            if token not in special_marks:#字母、数字会形成长度超过1的BPE。
                #每出现一个BPE，后面所有实体词的start_index都需要减去len(token)-1，才能符合bert的要求
                index_delta += len(token) - 1
                
                char_style_index += len(token)
            else:
                char_style_index += 1
#         print("entity_index_name_type_map", entity_index_name_type_map)
#         print("char_style_index_to_bert_style_map", char_style_index_to_bert_style_map)
        #基于char_style_index_to_bert_style_map，获取新版的触发词和事件元素关系数据
        for key, value in trigger_argument_map.items():
            new_key = (key[0], char_style_index_to_bert_style_map[key[1]])
            new_value = list(map(lambda x: (x[0], char_style_index_to_bert_style_map[x[1]]), list(value)))
            trigger_argument_map_new[new_key] = set(new_value)
        return entity_index_name_type_map_new, trigger_argument_map_new
    
    def get_event(self, text, ner_tags, argument_relation_sparse):
        mention_index_ner_map = {}
        temp_entity = ""
        triger_word = None
        for index in range(len(text)):###这个环节计算得到的ner位置是错误的，有错位的情况
            if ner_tags[index]=="O": continue
            else:
                temp_entity += text[index]
                position, entity_type = ner_tags[index].split("~")
                if position in ["E", "S"]:
                    mention_index_ner_map[index - len(temp_entity) + 1] = {"name": temp_entity, "type": entity_type}
                    
                    if entity_type in self.event_type_set:#如果是事件类型
                        triger_word = [index, temp_entity, entity_type]
                    temp_entity = ""
#         if triger_word: print("事件触发词是", triger_word)#需要以触发词为种子，把事件元素收集起来，形成事件列表
#         print('mention_index_ner_map', mention_index_ner_map, np.sum(argument_relation_sparse))
        mention_index_list = self.get_subject_object_relation_type(argument_relation_sparse)
#         print('mention_index_list', mention_index_list)
        event_list = []
        for i in range(len(mention_index_list)):
            event = {"event_type": None, "arguments": []}
            mention_index = mention_index_list[i]
            triger = mention_index_ner_map.get(mention_index["trigger_index"], None)
            if triger==None: continue
            event["event_type"] = triger["type"]
            event["trigger"] = triger["name"]
            for argument_index in mention_index["argument_indexes"]:
                if argument_index in mention_index_ner_map:
                    argument_info = mention_index_ner_map[argument_index]
                    argument = {"role": argument_info["type"], "argument": argument_info["name"]}
                    event["arguments"].append(argument)
            event_list.append(event)
        return event_list

    def get_subject_object_relation_type(self, event_info, event_info_type="sparse", seq_len=None):
        events = []
        event_map = {}
        seq_len = seq_len if seq_len else event_info.shape[0]
        for token_index in range(seq_len):
            if event_info_type=="prob_distribution":
                cand_trigger_index = np.argmax(event_info[token_index])
            else:
                cand_trigger_index = event_info[token_index]
            if cand_trigger_index!=0 and cand_trigger_index!=token_index:
                if cand_trigger_index not in event_map: event_map[cand_trigger_index] = []
                event_map[cand_trigger_index].append(token_index)
        for cand_trigger_index in event_map:
            events.append({"trigger_index": cand_trigger_index, "argument_indexes": event_map[cand_trigger_index]})
        return events

    def remove_not_activated_event_types(self, event_list):
        new_event_list = []
        for event in event_list:
            if event["event_type"] in self.event_type_set:
                new_event_list.append(event)
        return new_event_list
    
    def get_features(self, data_list, if_train=True):
        samples = []
        for a_data in data_list:
            if len(a_data['text'])>1000: print("超长文本")
            text, event_list = a_data['text'][:20000], a_data['event_list']
            if run_time.IF_FILTER_EVENT_TYPES_IN_TRAINING: event_list = self.remove_not_activated_event_types(event_list)
            if if_train and len(event_list)==0: continue
            a_sample = self.feature_preprocess(text, event_list, run_time.MAX_TEXT_LEN)
            samples.append(a_sample)
        return samples
            
    def get_train_data(self, if_demo=False):
        train_data = self.load_corpus(run_time.PATH_EVENT_EXTRACTION_TRAIN_CORPUS)#[:1]
        if if_demo: train_data=train_data[:100]
        train_data = self.get_features(train_data)
        return train_data
    
    def get_dev_data(self, if_demo=False):
        dev_data = self.load_corpus(run_time.PATH_EVENT_EXTRACTION_DEV_CORPUS)#[:1]
        if if_demo: dev_data=dev_data[:100]
        dev_data = self.get_features(dev_data)
        return dev_data
    
    def text_list_into_a_batch(self, text_list):
        data_list = list(map(lambda x: {"text": x, "event_list": []}, text_list))
        data_list = self.get_features(data_list, if_train=False)
        token_ids_list, seq_lens = [], []
        for data in data_list:
            token_ids_list.append(data[0])
            seq_lens.append(data[1])
        return token_ids_list, seq_lens
             
        
if __name__ == '__main__':
    dp = DataProcessor4EventExraction()
    test_data = dp.load_corpus(run_time.PATH_EVENT_EXTRACTION_DEV_CORPUS)#[:1]
#     test_data_batch_list = dp.prerocess_data(test_data, batch_size=20, shuffle=False, size=None)
#     for line in test_data_batch_list: 
#         a_batch = dp.preprocess_a_batch(line)
#         break
# #         print(a_batch)
    
    
    