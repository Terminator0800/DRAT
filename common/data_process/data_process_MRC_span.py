'''
Created on 2021年8月6日

@author: Administrator
'''
#对阅读理解数据集进行预处理，只支持squad风格
import sys, os
cwd = os.path.dirname(os.getcwd())
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))
from config import run_time
import json
import pickle
import numpy as np
import random
from common.data_process.base_processor import DataProcessor, _truncate_seq_pair
import pandas as pd
import jieba
from common.utils import splitSentence

class DataProcessorMRC(DataProcessor):
    
    def __init__(self):
        super(DataProcessorMRC, self).__init__()  
                        
    def get_train_data(self, path_corpus, train_set_size=-1, if_demo=False):
        samples, text_list = [], []
        dp = DataPreprocess4Corpus()
        data_list = dp.load_corpus(path_corpus)
        data_list = dp.corpus_to_input_style(data_list)
        data_list = dp.convert_answer_span_from_char_style_to_BPE(data_list) 
        for data in data_list:
            span, doc, tokens, question = data
            text_list.append(doc)
            question_tokens = self.tokenizer.tokenize(question)
            _truncate_seq_pair(tokens, question_tokens, run_time.MAX_TEXT_LEN-3)
            tokens = ["[CLS]"] + tokens + ["[SEP]"] + question_tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)#BPE
            seq_len = len(tokens)
            segment_ids = [0] * (len(tokens) + 2) + [1] * (len(question_tokens) + 1) + [0] * (run_time.MAX_TEXT_LEN - seq_len)  
            input_mask = [1] * seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)
            token_ids = self.padding_token_ids(token_ids, run_time.MAX_TEXT_LEN)
            samples.append([token_ids, seq_len, segment_ids, input_mask, span])
        return samples, text_list
    

class DataPreprocess4Corpus(DataProcessor):
    def __init__(self):
        super(DataPreprocess4Corpus, self).__init__()

    #将语料从繁体字转换为简体字https://github.com/DRCKnowledgeTeam/DRCD
    def transform_traditional_corpus_into_simple(self):
        for file_name in os.listdir(run_time.PATH_TRADITIONAL_CHN_CORPUS):
            lines = list(open(run_time.PATH_TRADITIONAL_CHN_CORPUS + '/' + file_name, 'r', encoding='utf8').readlines())
            with open(run_time.PATH_CORPUS + '/' + file_name, 'w', encoding='utf8') as f:
                f.writelines(lines)
            
    def load_corpus(self, file_name, if_demo=False):
        if ".txt" in file_name:
            lines = list(open(file_name, 'r', encoding='utf8').readlines())
            data = json.loads(lines[0].strip())["data"]
        else:
            data = json.load(open(file_name, 'r', encoding='utf8'))['data']
        
#         random.shuffle(data)
#         data = data[:10]
#         print("正在加载原始数据:", file_name)
        print("文档总数是", len(data))
        
        ############统计文本数据分布############
        para_num = 0
        max_para_len, max_question_len = 0, 0
        for doc in data:
            for para in doc['paragraphs']:
                para_num += 1
                if len(para['context'])>max_para_len: max_para_len = len(para['context'])
                for qa in para['qas']:
                    if len(qa['question'])>max_question_len: max_question_len = len(qa['question'])
    #             print(para)
#         print("段落总数树", para_num)
#         print("最常的段落里字数是", max_para_len, "最长的问题里字数是", max_question_len)#最常为1765，对于序列模型来说有点长，需要分拆。首先尝试简单的分拆策略，将段落拆分为500字、重叠200字的片段。
        
        ###############将所有文档分裂为段落，超长段落分拆为较短片段#################
        all_question_set = set({})
        data_list = []
        for doc in data:
            for para in doc['paragraphs']:
                samples, question_set = self.split_paragraph(para)
                data_list += samples
                all_question_set = all_question_set | question_set
        print("样本的个数是", len(data_list))

        #构造没有答案的数据
        '''
        all_questions = list(all_question_set)
        none_answer_QAs = []
        for i in range(len(data_list)):
            if random.uniform(0, 1) > 0.0: continue
            rand_question_index = random.randint(0, len(all_questions)-1)#这个选择方法比较粗糙，有可能选到相似甚至相同的问句
            rand_question = all_questions[rand_question_index]
            none_answer_QA = copy.deepcopy(data_list[i])
            none_answer_QA['question'] = rand_question
            none_answer_QA['answers'] = None
            none_answer_QAs.append(none_answer_QA)
        data_list +=  none_answer_QAs
        '''
        #######控制没有答案的样本的比例########
        new_data_list = []
        for data in data_list:
            #print(data["answers"])
            if data["answers"]==None or len(data["answers"])==0:
                continue
            new_data_list.append(data)
        random.shuffle(new_data_list)
        return new_data_list
    
    def corpus_to_input_style(self, data_list):
        new_data_list = []
        for data in data_list:
            answers = data["answers"]
            if answers:
                span = answers[0]['answer_start'] + 1
                start, end = span, span + len(answers[0]["text"]) + 1
            else:
                start, end = 0, 0
                
            if start>0:
                new_data = [[start, end], data["context"], data["question"]]
                new_data_list.append(new_data)
        return new_data_list
    
    def split_paragraph(self, para):
        samples = []#每次回答一个问题，训练数据里一个样本一个问题
        context = para['context']
        texts = self.split_context_sentence_version(context, run_time.MAX_DOC_LEN-2)
        questin_set = set({})
        for text in texts:
            base_index = text['start_index']#当前文本的起始点在原始文档中的位置
            for qa in para['qas']:#比那里所有的问答对，将落在当前文本范围内的挑出来，组成样本
                this_sampe = {"context": text['content'], 'sentences': splitSentence.getSentences(text['content']), \
                                                             "question": qa['question'][: run_time.MAX_QUESTION_LEN-1], 'answers': None}

                this_answer = []
                for answer in qa['answers']:
                    answer_start = answer['answer_start']#原始起点
                    answer_end = answer['answer_start'] + len(answer['text'])#原始结束点
                    if ("id" not in answer or answer['id'] in ['1', None]) and answer_start>=text['start_index'] and answer_end<=text['end_index']:
                        new_answer = {"text": answer['text'], "answer_start": answer['answer_start'] - base_index}
                        #print(this_sampe["question"], this_sampe['context'][new_answer["answer_start"]: new_answer["answer_start"] + len(answer['text']) ])
                        this_answer.append(new_answer)
                        
                #核对答案span正误
                if len(this_answer)>0:
                    start = this_answer[0]["answer_start"]
                    end = start + len(this_answer[0]["text"])
                    answer_text = this_sampe["context"][start: end]
                    if answer_text!=this_answer[0]["text"]:
                        print(this_answer[0], answer_text, start, end)
                        print(this_sampe["question"], "#", this_sampe["context"])

                # if "欲爱指的是什么" in qa["question"]: print(this_answer)
                this_sampe['answers'] = this_answer
                samples.append(this_sampe)
                questin_set.add(qa['question'])
        return samples, questin_set


    def split_context_sentence_version(self, context, max_text_len):
        texts = []
        sentences = splitSentence.getSentences(context)
        char_start_index = 0
        for sentence_no in range(len(sentences)):
            a_text = ""
            sentence_no_to_add = sentence_no
            while sentence_no_to_add< len(sentences) and len(a_text) + len(sentences[sentence_no_to_add])<max_text_len:
                a_text += sentences[sentence_no_to_add]
                
                sentence_no_to_add += 1
            end_index = char_start_index + len(a_text)
            if len(a_text)>5: 
                text = {"content": a_text, 'start_index': char_start_index, 'end_index': end_index}
                texts.append(text)
            char_start_index += len(sentences[sentence_no])

        return texts
    
    def convert_answer_span_from_char_style_to_BPE(self, data_list):
        special_marks = {"[UNK]", "[SEP]", "[CSL]", "[MARK]"}
        def update_index(an_index, token):
            if token in special_marks:
                an_index += 1
            else:
                new_token = ""
                for a_char in token:
                    if a_char!="#": new_token += a_char
                #new_token = token.replace("#", "")
                an_index += len(new_token)
            return an_index
        
        new_data_list = []
        for data in data_list:
            ori_span, context, question = data
            span = [ori_span[0]-1, ori_span[1]-1]
            tokens = self.tokenizer.tokenize(context)
            #print(tokens)
            BPE_start_index, BPE_end_index = 0, 0
            char_style_index = 0
            got_BPE_style_span = False
            for BPE_start_index in range(len(tokens)):
                if char_style_index==span[0]:
                    # print(char_style_index, span, context[span[0]: span[1]], tokens[BPE_start_index: BPE_start_index + 10])
                    for BPE_end_index in range(BPE_start_index, len(tokens)):
                        char_style_index = update_index(char_style_index, tokens[BPE_end_index])
                        # if "欲爱指的是什么" in question:
                        #     print(char_style_index, span, BPE_start_index, BPE_end_index,tokens[BPE_end_index])
#                         print(char_style_index)
                        if char_style_index + 1 ==span[1]:
                            got_BPE_style_span = True
                            break

                char_style_index = update_index(char_style_index, tokens[BPE_start_index])
                if got_BPE_style_span: break
            # if "欲爱指的是什么" in question:
            #     print(question, "got_BPE_style_span", got_BPE_style_span, context[ori_span[0]-1: ori_span[1]])
            if got_BPE_style_span:
                BPE_start_index += 1
                BPE_end_index += 1
                # if "欲爱指的是什么" in question:
                #     print(question, "BPE_start_index", tokens[BPE_start_index: BPE_end_index])
                new_data_list.append([[BPE_start_index, BPE_end_index], context, tokens, data[2]])
        return new_data_list

    def get_data(self, task_name, file_name, train_set_size=-1):

        data_list = self.load_corpus(file_name)

        data_list = self.corpus_to_input_style(data_list)
        data_list = self.convert_answer_span_from_char_style_to_BPE(data_list)
        text_list = [line[1] for line in data_list]
        samples = []
        no_answer_count = 0
        doc_len_list = []
        for line in data_list:
            span, text, question = line[0], line[1], line[3]
            doc_len_list.append(len(text))
            tokens1 = self.tokenizer.tokenize(text)
            tokens2 = self.tokenizer.tokenize(question)
            _truncate_seq_pair(tokens1, tokens2, run_time.MAX_TEXT_LEN - 3)
            tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # BPE
            seq_len = len(tokens)
            segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1) + [0] * (run_time.MAX_TEXT_LEN - seq_len)
            input_mask = [1] * seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)
            token_ids = self.padding_token_ids(token_ids, run_time.MAX_TEXT_LEN)
            label = span
            if label[0]==label[1]==0: no_answer_count += 1
            samples.append([token_ids, seq_len, segment_ids, input_mask, label])
        if train_set_size>0:
            samples = samples[:train_set_size]
        print("无答案的样本书", no_answer_count, '/', len(samples))
        print(task_name, "文档的平均长度是", np.mean(doc_len_list), len(samples))
        print("############################")
        return samples, text_list

if __name__ == '__main__':
    dp = DataPreprocess4Corpus()
    data_list = dp.load_corpus(r"../../../data/corpus/MRC/DRCD/DRCD_sample.json")
    data_list = dp.corpus_to_input_style(data_list)
    data_list = dp.convert_answer_span_from_char_style_to_BPE(data_list)

    for line in data_list:
        span, contenxt, question = line
#         ori_span, ori_contenxt, ori_question = ori_data
        print(question, contenxt[span[0]-1: span[1]-1])
                