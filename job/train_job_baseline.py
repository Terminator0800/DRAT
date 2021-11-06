# -*- coding: utf-8 -*-  
'''
Created on 2021

@author: Administrator
'''
import json
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from model.bert_base import BERTBaseModel
from config import run_time
from common.data_process import data_loader
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
import time
import torch
import numpy as np
from common.data_process import base_processor, data_process_text_similarity
import random as rander
rander.seed(666)

class Trainer():
  
    def __init__(self, main_task="text_similarity_LCQMC",  with_bilstm=False):
        self.learning_rate_for_BERT = 5e-6#rbt
        self.learning_rate_for_lstm = 1e-3
        # self.learning_rate_for_BERT = 1e-6#zen和ernie
        # self.learning_rate_for_lstm = 1e-3


        self.main_task = main_task
        self.with_bilstm = with_bilstm

    def init_optimizer(self, model, total_step_num):
        #################bert和基层模型的优化器#####################
        all_parameters = list(model.named_parameters())
        BERT_paprameters_names, LSTM_paprameters_names = [], []
        LSTM_paprameters_names = list(map(lambda x: "main_model." + x[0], list(model.main_model.named_parameters())))
        if self.with_bilstm == True:  # 如果需要吧top模型loss反馈给主任务模型，需要将主任务模型的参数加上
            LSTM_paprameters_names += list(map(lambda x: "encoder." + x[0], list(model.encoder.named_parameters())))
        LSTM_paprameters_names += list(map(lambda x: "main_model." + x[0], list(model.main_model.named_parameters())))

        BERT_paprameters = []
        LSTM_paprameters = []
        for para in all_parameters:
            if para[0] in LSTM_paprameters_names: LSTM_paprameters.append(para)
            else: BERT_paprameters.append(para)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in BERT_paprameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in BERT_paprameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.bert_optimizer = BertAdam(optimizer_grouped_parameters, lr=self.learning_rate_for_BERT,
                             warmup=0.1, t_total=total_step_num)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in LSTM_paprameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in LSTM_paprameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.lstm_optimizer = BertAdam(optimizer_grouped_parameters, lr=self.learning_rate_for_lstm,
                             warmup=0.1, t_total=total_step_num)



    def process_a_batch(self, training_data_batch):
        data = [[], []]#元素为输入和输出
        data[0] = [training_data_batch['inputs']["token_ids"],
                 training_data_batch['inputs']["segment_ids"],
                 training_data_batch['inputs']["input_mask"],
                 training_data_batch['inputs']["task_ids"]]
        #增加ner数据
        data[1] = [[] for _ in range(len(self.task_id_map))]#各个任务的输出放在一个列表中
        #吧输出数据收集起来
        for i in range(len(run_time.ALL_AUXILIARY_TASK_LIST)):
            data[1][self.task_name_id_map[run_time.ALL_AUXILIARY_TASK_LIST[i]]] = training_data_batch["outputs"][i]

        #把输入数据转移到显存
        for i in range(len(data[0])):
            data[0][i] = torch.LongTensor(data[0][i]).to(self.device)
        #把输出数据转移到现显存
        for i in range(len(data[1])):
            for j in range(len(data[1][i])):
                if not torch.is_tensor(data[1][i][j]) or \
                        (torch.is_tensor(data[1][i][j]) and data[1][i][j].device=="cpu"):
                    data[1][i][j] = torch.LongTensor(data[1][i][j]).to(self.device)
        return data
        
    def fit(self, model, mini_computer=False, if_demo=False, epoch_num=50, main_task_batch_size=8, trainingset_size=100,
                                  class_weight=None, devset_size=-1):
        #加载训练数据
        batch_size_rate=1
        if if_demo: batch_size=16
        else: batch_size=24
        self.task_id_map = {i: run_time.ALL_AUXILIARY_TASK_LIST[i]  for i in range(len(run_time.ALL_AUXILIARY_TASK_LIST))}
        self.task_name_id_map = {v: k for k, v in self.task_id_map.items()}
        batch_size = batch_size * batch_size_rate
        task_info = {
                     "text_similarity_LCQMC":  {"batch_size": main_task_batch_size, "role": "auxiliary", "attention": 1, "train_set_size": trainingset_size},
                     }

        print("加载所有的数据")
        data_for_all_tasks_map, class_weight = data_loader.load_data_for_all_tasks(trainingset_size, devset_size=devset_size)
        print("data_for_all_tasks_map", data_for_all_tasks_map.keys())
        print("class_weight", class_weight)
        for key in list(task_info.keys()):
            if key not in run_time.ALL_AUXILIARY_TASK_LIST: del task_info[key]
        self.device = model.device
        task_info[self.main_task]["role"] = "main"
        
        print("预处理开发集")
        data_loader.all_data_into_batches(data_for_all_tasks_map, main_task_batch_size, this_rander=rander)
        dev_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map,
                                                              job_stage="dev")
        test_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map,
                                                              job_stage="test")
        print("预处理训练集")
        training_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map)
        time_cost = 0
        best_f1score = 0.0
        self.init_optimizer(model, len(training_data_batches)*epoch_num)
        model.to(self.device)
        model = torch.nn.DataParallel(model)#, device_ids=run_time.GPU_IDS)#数据并行训练
        batch_num_in_a_batch = int(len(training_data_batches)/epoch_num)
        global_step = 0
        trainng_records = []
        # print("###############训练之前的能力###################")
        # self.evaluate(-1, dev_data_batches, model, trainng_records)
        print("开始训练")
        for epoch in range(epoch_num):
            t1 = time.time()
            #rander.seed(int(time.time()))
            data_loader.all_data_into_batches(data_for_all_tasks_map, main_task_batch_size, this_rander=rander)
            training_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map)
            batch_num_in_a_batch = len(training_data_batches)
            losses, mian_losses, top_losses = [], [], []
            for batch_index in range(batch_num_in_a_batch):
                global_step += 1
                model.train()
                a_batch = training_data_batches[batch_index]
                a_batch = self.process_a_batch(a_batch)
                    
                #计算输出
                outputs, main_outputs = model.forward(a_batch[0])
                #计算损失值
                targets = a_batch[1][0][0]
                # print("main_outputs", main_outputs.size())
                # print("targets", targets[0])
                if class_weight!=None:
                    loss = F.cross_entropy(main_outputs, targets, weight=torch.tensor(class_weight).cuda())
                else:
                    loss = F.cross_entropy(main_outputs, targets)

                #基于所有基层任务的误差更新各个模型的参数
                model.zero_grad()
                #计算梯度梯度
                #loss = loss.mean()
                loss.backward()
                #更新参数
                self.bert_optimizer.step()
                if self.with_bilstm:
                    self.lstm_optimizer.step()
                
                losses.append(loss.item())
                #top_losses.append()

                trainng_progress = {"epoch": epoch, "loss": np.mean(losses),
                                    "进度是": str(batch_index) + "/" + str(batch_num_in_a_batch)}
                if batch_index%1000==0 and loss:
                    trainng_records.append(trainng_progress)
                if global_step%100==0:
                    print(trainng_progress)
                    losses = []
                del loss
            #print("epoch", epoch, "loss", str(np.mean(losses))[:5])

            #做一次测试
            if (epoch + 1)%5==0:
                print("###############测试结果###################")
                f1score = self.evaluate(epoch, dev_data_batches, model, trainng_records, mode="dev")
                if f1score>best_f1score:
                    self.evaluate(epoch, test_data_batches, model, trainng_records, mode="test")
                    best_f1score = f1score
            t2 = time.time()
            time_cost = t2 - t1
            #print("speed", len(data_for_all_tasks_map["text_similarity_LCQMC"]["train"]["samples"])/time_cost)
        return trainng_records
                
            
    def evaluate(self, epoch, dev_data_batches, model, trainng_records, mode='dev'):
        
        def get_confusion_matrix(confusion_matrix, predict_logits, real_labels):
            predict_outputs = list(map(lambda x: 1 if x[1]>x[0] else 0, predict_logits))
            for i in range(len(predict_outputs)):
                prediction, real_label = predict_outputs[i], real_labels[i]
                confusion_matrix[prediction][real_label] += 1
        
        def get_evaluation_metrics(loss, confusion_matrix, stage):
            print(confusion_matrix)
            pred_0_real_0 = confusion_matrix[0][0]
            pred_0_real_1 = confusion_matrix[0][1]
            pred_1_real_0 = confusion_matrix[1][0]
            pred_1_real_1 = confusion_matrix[1][1]
            accuracy = (pred_0_real_0 + pred_1_real_1)/np.sum(confusion_matrix)
            #计算对正例的查全率、查准率和f1
            recall = pred_1_real_1/(pred_0_real_1 + pred_1_real_1 + 1e-5)
            precision = pred_1_real_1/(pred_1_real_0 + pred_1_real_1 + 1e-5)
            f1_score = 2*recall*precision/(recall + precision + 1e-5)

            trainng_progress = {"epoch": epoch, "loss": np.mean(loss), "accuracy": accuracy, \
                   "recall": recall,  'precision': precision, "f1score": f1_score,\
                                "confusion_matrix": confusion_matrix, "stage": stage}

            trainng_records.append(trainng_progress)
            print(trainng_progress)
            return f1_score + accuracy
        
        with torch.no_grad():
            model.eval()
            confusion_matrix_main = [[0, 0], [0, 0]]
            confusion_matrix_aux = [[0, 0], [0, 0]]
            loss_list = []
            for a_batch_ori in dev_data_batches:
                a_batch = self.process_a_batch(a_batch_ori)
                outputs, main_outputs = model.forward(a_batch[0])
                get_confusion_matrix(confusion_matrix_main, main_outputs, a_batch_ori["outputs"][0][0])
                get_confusion_matrix(confusion_matrix_aux, outputs, a_batch_ori["outputs"][0][0])
                loss = F.cross_entropy(main_outputs, a_batch_ori["outputs"][0][0])
#                 predict_logits = outputs[self.task_name_id_map["text_similarity_LCQMC"]][0]
                loss_list.append(loss.item())
            #outputs, main_outputs = model.forward(a_batch[0], print_weights=True)
            print("主模型的结果:")
            score = get_evaluation_metrics(np.mean(loss_list), confusion_matrix_main, mode)
        return score

def create_a_dir(dir_name):
    if os.path.exists(dir_name):
        #os.rmdir(dir_name)
        os.system("rm -rf " + dir_name)
    os.mkdir(dir_name)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"#"1, 0, 2, 3"
    
    run_time.GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
#     run_time.GPU_IDS = [0]
    devset_size = -1#500
    run_time.ALL_AUXILIARY_TASK_LIST = ["text_similarity_LCQMC"]

    PATH_BERT_BASE_DIR_list = [
        "../../data/pretrained_models/model-ernie-gram-zh.1_torch",
        "../../data/pretrained_models/model-ernie1.0.1",
                               ]

    for PATH_BERT_BASE_DIR in PATH_BERT_BASE_DIR_list:
        run_time.PATH_BERT_BASE_DIR = PATH_BERT_BASE_DIR
        for trainingset_size in [200, 500, 1000]:
            for with_bilstm in [True]:
                for main_task_batch_size in [50]:
                    dir_name = "../../data/results/baseline_" + str(trainingset_size) + "_smaples_"
                    dir_name += PATH_BERT_BASE_DIR.split("/")[-1]
                    print("dir_name", dir_name, PATH_BERT_BASE_DIR)
                    if with_bilstm: dir_name += "with_BiLSTM/"
                    else: dir_name += "no_BiLSTM/"
                    create_a_dir(dir_name)
                    for test_number in range(10):
                        model_plan = {"with_bilstm": with_bilstm, "trainingset_size": trainingset_size}
                        model = BERTBaseModel(mode="train", with_bilstm=with_bilstm)

                        trainer = Trainer(main_task="text_similarity_LCQMC", with_bilstm=with_bilstm)
                        records = trainer.fit(model, mini_computer=True, if_demo=True, main_task_batch_size=main_task_batch_size, \
                                              trainingset_size=trainingset_size, devset_size=devset_size, epoch_num=50)

                        file_name = dir_name + str(test_number) + ".txt"
                        with open(file_name, 'w', encoding='utf8') as f:
                            f.write(json.dumps(model_plan, ensure_ascii=False) + "\n")
                            for record in records:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")

