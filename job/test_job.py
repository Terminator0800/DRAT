'''
Created on 2021年1月29日

@author: Administrator
'''
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import json
from config import run_time
from baseline.MRC_base import MRCSpan
from common.data_process.base_processor import DataProcessor, _truncate_seq_pair
from common.data_process import base_processor
import numpy as np
import torch

dp_base = base_processor.DataProcessor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("设备", device)

def preprocess(question_text_pairs):
    token_ids_list, seq_len_list, segment_ids_list, input_mask_list, label_list = [], [], [], [], []
    token_list = []
    for question, text in question_text_pairs:
        tokens1 = dp_base.tokenizer.tokenize(text)
        tokens2 = dp_base.tokenizer.tokenize(question)
        _truncate_seq_pair(tokens1, tokens2, run_time.MAX_TEXT_LEN - 3)
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        token_ids = dp_base.tokenizer.convert_tokens_to_ids(tokens)  # BPE
        seq_len = len(tokens)
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1) + [0] * (run_time.MAX_TEXT_LEN - seq_len)
        input_mask = [1] * seq_len + [0] * (run_time.MAX_TEXT_LEN - seq_len)
        token_ids = dp_base.padding_token_ids(token_ids, run_time.MAX_TEXT_LEN)
        label = [0, 0]
        token_ids_list.append(token_ids)
        seq_len_list.append(seq_len)
        segment_ids_list.append(segment_ids)
        input_mask_list.append(input_mask)
        label_list.append(label)
        token_list.append(tokens)

    return torch.LongTensor(token_ids_list).to(device), \
           torch.LongTensor(seq_len_list).to(device),\
           torch.LongTensor(segment_ids_list).to(device), \
           torch.LongTensor(input_mask_list).to(device), \
           torch.LongTensor(label_list).to(device), \
           token_list

def postprocess(question_text_pairs, token_list, main_outputs):
    for i in range(len(token_list)):
        BPE_start = np.argmax(main_outputs[0][i].cpu().detach().numpy())
        BPE_end = np.argmax(main_outputs[1][i].cpu().detach().numpy()) + 1
        index, char_start, char_end = 0, 0, 0
        for j in range(len(token_list[i])):
            token = token_list[i][j]
            if j==BPE_start: char_start = index
            if j==BPE_end: char_end = index
            index += len(token.replace("#", ""))
        print(BPE_start, BPE_end, token_list[i])
        char_start -= 5
        char_end = char_end - 5
        print(question_text_pairs[i], char_start, char_end, question_text_pairs[i][1][char_start: char_end])


def demo():
    model = MRCSpan(mode="work", with_bilstm=True)
    model_paras = torch.load('../../data/models/model_MRC.pth')
    model_paras = {k.replace("module.", ""): v for k, v in model_paras.items() if k in model_paras}
    model.load_state_dict(model_paras)
    model.eval()
    model.to(device)
    question_text_pairs = [["今天是几号", "似乎今天的日期是16日啊。"], ["谁是共产党的创始人?", "共产党是毛泽东他们建立的。"]]
    token_ids_list, seq_len_list, segment_ids_list, input_mask_list, label_list, token_list = preprocess(question_text_pairs)
    main_outputs, main_outputs = model.forward([token_ids_list, segment_ids_list, input_mask_list, seq_len_list])
    answers = postprocess(question_text_pairs, token_list, main_outputs)


if __name__ == '__main__':
    demo()
