# -*- coding: utf-8 -*-  
'''
Created on 2021

@author: Administrator
'''
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from model.loss import MultiTaskLoss
from model.main_model_linear_task_weight import MultiTaskModel as MultiTaskModelLinearTaskWeight

import json
from config import run_time
from common.data_process import data_loader
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW
import time
import torch
import numpy as np
import random as rander
rander.seed(666)
import copy
from model import evaluation_tools

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
class Trainer():
  
    def __init__(self, supervised=True, separated_training=False, total_step=-1, part_of_transformer="encoder"):

        #'''
        #ernie
        self.learning_rate_for_models = 2e-6#2e-5#1e6e rnirgram bertbase
        self.learning_rate_for_sub_task_model = 1e-3
        self.learning_rate_for_supervisor = 1e-3
        self.learning_rate_for_task_weight = 1e-2
        #'''

        self.supervised = supervised
        self.separated_training = separated_training
        self.total_step = total_step
        self.part_of_transformer = part_of_transformer
        print("训练不熟是", total_step)

    def init_optimizer(self, model, multi_task_loss, total_step_num):
        total_step_num = self.total_step
        #################bert和基层模型的优化器#####################
        all_parameters = list(model.named_parameters())
       # print("souyou de canshu ", list(map(lambda x: x[0], all_parameters)))
        #print("所有的参数", list(map(lambda x: x[0], all_parameters)))

        sub_task_special_parameter_names = []#所有的子任务模型参数名
        main_task_parameter_names = []#main任务模型参数名
        supervisor_parameter_names = []#监督模型参数名
        task_weight_parameter_names = []#任务权重
        base_model_parameter_names = []#编码器权重
        print("model.task_weight", model.task_weight)
        supervisor_parameter_names = list(map(lambda x: "supervisor." + x[0], list(model.supervisor.named_parameters())))

        print("supervisor_parameter_names", supervisor_parameter_names)
        print("缓存的模型个数是", len(model.model_seq))
        sub_task_special_parameter_names_map = {}
        for task_id in range(len(run_time.ALL_AUXILIARY_TASK_LIST)):
            sub_task_special_parameter_names_map[task_id] = \
                list(map(lambda x: "model_seq." + str(task_id) + "." + x[0], list(model.model_seq[task_id].named_parameters())))
            #print(task_id, sub_task_special_parameter_names_map[task_id])

        task_weight_parameter_names = ["task_weight", "share_rate"]
        base_model_parameter_names = list(map(lambda x: "bert." + x[0], list(model.bert.named_parameters())))

        supervisor_model_paprameters = []
        base_models_parameters = []
        sub_task_special_parameters_map = {task_id: [] for task_id in sub_task_special_parameter_names_map.keys()}
        task_weight_parameters = []
        shared_encoder_parameters = []
#         print("sub_task_special_parameter_names_map", sub_task_special_parameter_names_map)
        for para in all_parameters:
            #print("para", para[0])
            if para[0] in supervisor_parameter_names:
                supervisor_model_paprameters.append(para)
            for task_id in sub_task_special_parameter_names_map:
                if para[0] in sub_task_special_parameter_names_map[task_id]:
                    #print("sub_task_special_parameter_names_map[task_id]", task_id, sub_task_special_parameter_names_map[task_id])
                    sub_task_special_parameters_map[task_id].append(para)
                    
            if para[0] in task_weight_parameter_names:
                task_weight_parameters.append(para)
            if para[0] in base_model_parameter_names:
                base_models_parameters.append(para)

        # print("sub_task_special_parameter_names_map【0】", sub_task_special_parameter_names_map[0])
        #print(sub_task_special_parameter_names)
        print("顶层模型的参数", list(map(lambda x: x[0], supervisor_model_paprameters)))
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        ###编码器
        warmup = 0.05
        optimizer_grouped_parameters = [
            {'params': [p for n, p in base_models_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in base_models_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # if self.if_shared_encoder_bert_family:
        self.base_model_optimizer = BertAdam(optimizer_grouped_parameters, lr=self.learning_rate_for_models,
                         warmup=warmup, t_total=total_step_num)
        # else:
        #     self.base_model_optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate_for_models)

        ################顶层模型的优化器##################
        supervisor_model_paprameters = [
            {'params': [p for n, p in supervisor_model_paprameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in supervisor_model_paprameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.supervisor_model_optimizer = BertAdam(supervisor_model_paprameters, lr=self.learning_rate_for_supervisor,
                             warmup=warmup, t_total=total_step_num)
        # self.supervisor_model_optimizer = \
        #        torch.optim.Adam(supervisor_model_paprameters, lr=self.learning_rate_for_supervisor)
        ################任务权重的优化器##################
        task_weight_optimizer_grouped_parameters = [
            {'params': [p for n, p in task_weight_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in task_weight_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.task_weight_optimizer = BertAdam(task_weight_optimizer_grouped_parameters, lr=self.learning_rate_for_task_weight,
                             warmup=warmup, t_total=total_step_num)
        # self.task_weight_optimizer = \
        #        torch.optim.Adam(task_weight_optimizer_grouped_parameters, lr=self.learning_rate_for_task_weight)

        #子任务模型优化器
        self.sub_task_optimizer_map = {}
        self.sub_task_optimizer_map_II = {}
        for task_id in sub_task_special_parameters_map.keys():
            task_parameters = sub_task_special_parameters_map[task_id]
            task_parameters = [
                {'params': [p for n, p in task_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in task_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            self.sub_task_optimizer_map[task_id] = BertAdam(task_parameters, lr=self.learning_rate_for_sub_task_model,
                                 warmup=warmup, t_total=total_step_num)
            # self.sub_task_optimizer_map[task_id] = torch.optim.SGD(task_parameters, lr=self.learning_rate_for_sub_task_model)
            # self.sub_task_optimizer_map_II[task_id] = \
            #     torch.optim.Adam(task_parameters, lr=self.learning_rate_for_models)


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

        
    def fit_pretraining(self, epoch_num=1, batch_size=8, train_set_size=1000, devset_size=100, \
                        init_task_attention=None, stage=None, update_task_weight=True, stastics_mode=False, remove_some_tasks=False,
                                   remove_rate=0.0):
        multi_task_loss = MultiTaskLoss()
        #加载训练数据d
        data_for_all_tasks_map, class_weight = data_loader.load_data_for_all_tasks(train_set_size, devset_size=devset_size)
        print("class_weight", class_weight)
        #############初始化模型###################
        main_task_name = run_time.ALL_AUXILIARY_TASK_LIST[0]
        if task_attention_method == "linear":
            print("是线性加权模型")
            model = MultiTaskModelLinearTaskWeight(mode="train", main_task=main_task_name, \
                                                   supervised=supervised,
                                                   init_task_attention=init_task_attention,
                                                   update_task_weight=update_task_weight,
                                                   part_of_transformer=self.part_of_transformer)
        #############################
        self.main_task = main_task_name
        self.task_id_map = model.task_id_map
        self.task_name_id_map = model.task_name_id_map
        self.device = model.device

        print("预处理开发集")
        data_loader.all_data_into_batches(data_for_all_tasks_map, batch_size, this_rander=rander)
        dev_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map,
                                                              job_stage="dev")

        test_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map,
                                                              job_stage="test")
        print("预处理训练集")
        training_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map)
        time_cost = 0
        
        self.init_optimizer(model, multi_task_loss, len(training_data_batches)*epoch_num)
        model.to(self.device)
        # model = torch.nn.DataParallel(model)#, device_ids=run_time.GPU_IDS)#数据并行训练
        score_best = 0
        trainng_records = []#训练过程的记录
        print("开始训练")
        global_step = 0
        learned_sample_count_each_task = {k: 0 for k in run_time.ALL_AUXILIARY_TASK_LIST}
        for epoch in range(epoch_num):
            t1 = time.time()
            data_loader.all_data_into_batches(data_for_all_tasks_map, batch_size, this_rander=rander)
            training_data_batches = data_loader.combine_all_task_samples(self.main_task, data_for_all_tasks_map)
            batch_num_in_a_epoch = len(training_data_batches)
            losses, mian_losses, top_losses = [], [], []
            for batch_index in range(batch_num_in_a_epoch):
                global_step += 1
                model.train()
                multi_task_loss.train()
                a_batch = training_data_batches[batch_index]
                a_batch = self.process_a_batch(a_batch)
              
                #计算输出
                outputs, main_outputs, task_attention, share_rate = model.forward(a_batch[0])
                task_attention = task_attention.cpu().detach().numpy().tolist()

                task_attention_for_encoder = list(task_attention)
                if remove_some_tasks == True:
                    task_attention_list = sorted(zip(task_attention, list(range(len(task_attention)))), key=lambda x: x[0])
                    t_index = task_attention_list[int(remove_rate*len(task_attention_list))][1]
                    for i in range(len(task_attention_list)):
                        if task_attention_for_encoder[i]<task_attention[t_index]: task_attention_for_encoder[i] = 0.0
                # print("task_attention_for_encoder", task_attention_for_encoder)
                #计算损失值
                loss, main_loss, top_loss = 0, 0, None
                losses_in_each_task = [0 for _ in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
                loss_map_this_batch = {}
                for task_id in range(len(self.task_id_map)):#辅助任务
                    # print("任务是", task_id, a_batch[1][task_id])
                    if task_id not in outputs or len(outputs[task_id])==0: continue
                    for sub_task_id in range(len(a_batch[1][task_id])):#这个辅助任务的各个监督信号
                        real_outputs = a_batch[1][task_id][sub_task_id]
                        # print(task_id, "#############", sub_task_id, "数据量", len(real_outputs))
                        loss_of_this_task = 0
                        if task_id != self.task_name_id_map[self.main_task]:
                            loss_of_this_task = multi_task_loss.forward(self.task_name_id_map[self.main_task], \
                                                           task_id, outputs[task_id][sub_task_id], real_outputs,
                                                                1)#task_attention[task_id])
                            # print("loss_of_this_task", loss_of_this_task.item(), task_attention[task_id], task_attention_for_encoder[task_id] , task_attention_for_encoder[task_id]* loss_of_this_task.item())
                            # if loss_of_this_task < 0.0001:
                            #     task_attention_for_encoder[task_id] = 0.0
                            #     print("任务", task_id, "exit")
                            loss = loss + task_attention_for_encoder[task_id] * loss_of_this_task
                            losses_in_each_task[task_id] = losses_in_each_task[task_id] + loss_of_this_task
                            loss_map_this_batch[task_id] = loss_of_this_task
                        else: # 遇到主任务，顺便计算顶层主任务的loss
                            loss_of_this_task = multi_task_loss.forward(self.task_name_id_map[self.main_task], \
                                                           task_id, outputs[task_id][sub_task_id], real_outputs,
                                                                           1, class_weight=class_weight)
                            losses_in_each_task[task_id] = losses_in_each_task[task_id] + loss_of_this_task
                            loss_map_this_batch[task_id] = loss_of_this_task
                            if loss_of_this_task!=None:
                                loss = loss + loss_of_this_task
                                main_loss = loss_of_this_task.item()
                            else:
                                main_loss = 0
                            top_task_id = self.task_name_id_map[self.main_task]
                            if self.supervised:
                                # print("self.main_task", self.main_task)
                                top_loss = multi_task_loss.forward(self.task_name_id_map[self.main_task], \
                                            top_task_id, main_outputs, real_outputs, 1, class_weight=class_weight)#top_loss为top任务上的loss
                        if loss_of_this_task>0:
                            learned_sample_count_each_task[run_time.ALL_AUXILIARY_TASK_LIST[task_id]] += 1

                #计算梯度梯度
                
                #基于顶层模型的误差，更新所有参数
                if self.separated_training:#分别学习基层任务和顶层任务
                    #更新编码器
                    model.zero_grad()
                    loss.backward(retain_graph=True)
                    self.base_model_optimizer.step()

                    for task_id in self.sub_task_optimizer_map:
                        if task_id not in loss_map_this_batch: continue
                        #     #print("主任务不学习")
                        #     continue
                        model.zero_grad()
                        this_loss = loss_map_this_batch[task_id]#/task_attention[task_id])#loss_map_this_batch[task_id]#
                        this_loss.backward(retain_graph=True)
                        #     if task_id == 0: continue
                        self.sub_task_optimizer_map[task_id].step()


                    #更新监督模型
                    if self.supervised==True and top_loss!=None and top_loss!=0:
                        model.zero_grad()
                        top_loss.backward()
                        self.supervisor_model_optimizer.step()
                        self.task_weight_optimizer.step()
                        top_loss = top_loss.item()
                    else:
                        top_loss = 0

                else:
                    loss = loss.mean()# + top_loss.mean()
                    loss.backward()
                    self.supervisor_model_optimizer.step()
                    for task_id in self.sub_task_optimizer_map:
                        self.sub_task_optimizer_map[task_id].step()
                    self.base_model_optimizer.step()
                    self.task_weight_optimizer.step()

                losses.append(loss.item())
                if main_loss!=None: mian_losses.append(main_loss)
                if top_loss!=None: top_losses.append(top_loss)
                trainng_records.append({"task_attention": task_attention})
                trainng_progress = {"epoch": epoch, "batch_index": batch_index, "batch_num_in_a_epoch":batch_num_in_a_epoch , "loss": np.mean(losses), "main_task_loss": np.mean(mian_losses), \
                                    "top_loss": np.mean(top_losses),
                                    "进度是": str(batch_index) + "/" + str(batch_num_in_a_epoch), \
                                    "losses_in_each_task": list(map(lambda x: 0 if x==0 else x.item(), losses_in_each_task)),
                                    'stage': "pretrain" , "task_weight": task_attention, "learned_sample_count_each_task": learned_sample_count_each_task}

                if stastics_mode==False:
                    if batch_index%10==0 and loss:
                        trainng_records.append(trainng_progress)

                    if global_step % 30 == 0:
                        print([[k,v] for k,v in trainng_progress.items() if k in ["epoch", "进度是", "main_task_loss", "top_loss", "losses_in_each_task"]])
                        print("share_rate", share_rate.cpu().detach().numpy().tolist()[0], "task_attention", task_attention)

                    del loss, main_loss, top_loss
                    if stage!="getweight" and ((epoch%2==0 and batch_index + 1==batch_num_in_a_epoch) or \
                                               (batch_index + 1)%max(16, int(0.05*len(run_time.ALL_AUXILIARY_TASK_LIST)*train_set_size/batch_size))==0):#

                        if run_time.AUXILIARY_TASK_TYPE_MAP[self.main_task] in ['SIM']:
                            score = evaluation_tools.evaluate_SIM(epoch, dev_data_batches, model, trainng_records, self, mode='dev', task_weight=task_attention)
                        if run_time.AUXILIARY_TASK_TYPE_MAP[self.main_task] == 'MRC':
                            score = evaluation_tools.evaluate_MRC(epoch, dev_data_batches, model, trainng_records, self, mode='dev')
                        if run_time.AUXILIARY_TASK_TYPE_MAP[self.main_task] == 'NER':
                            score = evaluation_tools.evaluate_NER(epoch, dev_data_batches, model, trainng_records, self, mode='dev')
                        if run_time.AUXILIARY_TASK_TYPE_MAP[self.main_task] == 'SEN':
                            score = evaluation_tools.evaluate_CLS(epoch, dev_data_batches, model, trainng_records, self, mode='dev')
                        print("this score", score)
                        if score > score_best:
                            torch.save(obj=model.state_dict(), f="../../data/models/model_stage_I.pth")
                            score_best = score
                            print("最佳开发set", score_best)
                            trainng_progress["best_record"] = True
                            trainng_records.append(trainng_progress)
                            if self.main_task=="text_similarity_LCQMC":
                                print("测试set")
                                evaluation_tools.evaluate_SIM(epoch, test_data_batches, model, trainng_records, self, mode='test')
                        print("###############")
                else:
                    if batch_index%10==0: print(epoch, batch_index, "learned_sample_count_each_task", learned_sample_count_each_task)
        return model, trainng_records, task_attention

def create_a_dir(dir_name):
    if os.path.exists(dir_name):
        #os.rmdir(dir_name)
        os.system("rm -rf " + dir_name)
        #os.remove(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"#"1, 0, 2, 3"
    
    run_time.GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
#     run_time.GPU_IDS = [0]
    auxiliary_task_list = run_time.ALL_AUXILIARY_TASK_LIST#     auxiliary_task_list = ["text_similarity_LCQMC"]
    task_attention_method = "linear"#"#计算任务权重的方法

    model_plan =  {"plan_name": "有监督，分段训练，监督模型的误差不反馈给主任务模型，现行价全", "supervised": False,
         "separated_training": True, "comment": "最好"}

    axuliary_task_plans = [
         ["text_similarity_LCQMC", "text_classification_THUCNews", "text_smililarity_atec", "text_classification_toutiao","text_similarity_CCKS2018",
                       "text_classification_tansongbo_hotel_comment", "NER_rmrb", "NER_CLUENER_public", "MRC_DRCD", "MRC_cmrc2018", \
                           "text_classification_chnsenticorp", "text_similarity_ChineseSTS", "text_classification_simplifyweibo_4_moods",
                           "NER_MSAR", "NER_rmrb_2014", "NER_boson", "NER_weibo", "MRC_chinese_SQuAD", "MRC_CAIL2019", "NER_CMNER",
                           "NER_CCKS2019_task1_Yidu_S4K", "NER_baidu2020_event", "NLI_cmnli_public", "NLI_ocnli_public", "sentiment_analysis_dmsc_v2",
                             "sentiment_analysis_online_shopping_10_cats", "sentiment_analysis_waimai_10k", "sentiment_analysis_weibo_senti_100k",
                           "sentiment_analysis_yf_dianping"],
        ["text_similarity_LCQMC", "text_classification_THUCNews", "text_smililarity_atec"],
        ]

           # [["text_classification_THUCNews", "text_smililarity_atec", "text_classification_toutiao","text_similarity_CCKS2018",
           #             "text_classification_tansongbo_hotel_comment", "NER_rmrb", "NER_CLUENER_public", "MRC_DRCD", "MRC_cmrc2018"]]

        # ["text_similarity_CCKS2018", "NER_rmrb", "NER_CLUENER_public"],
    model_plan["supervised"] = True
    run_time.ALL_AUXILIARY_TASK_LIST = ["text_similarity_LCQMC"]#["MRC_DRCD"]#主任务##MRC_DRCD#text_classification_chnsenticorp#text_smililarity_atec#NER_MSAR
    stastics_mode = False#是否只计算任务权重
    remove_rate = 0.67
    pretrain_epoch, finetune_epoch = 5, 20#few tasks 10 epoch, many tasks 5 epoch
    part_of_transformer = "encoder"
    devset_size = -1
    PATH_BERT_BASE_DIR_list = [
        "../../data/pretrained_models/model-ernie-gram-zh.1_torch",
                               ]
    for subtask_list in axuliary_task_plans:
        subtask_list = [subtask_list] if type(subtask_list) == str else subtask_list
        subtask_list = list(subtask_list)
        if run_time.ALL_AUXILIARY_TASK_LIST[0] in subtask_list: subtask_list.remove(run_time.ALL_AUXILIARY_TASK_LIST[0])#删除主任务
        run_time.ALL_AUXILIARY_TASK_LIST += subtask_list
        print(run_time.ALL_AUXILIARY_TASK_LIST)
        for PATH_BERT_BASE_DIR in PATH_BERT_BASE_DIR_list:
            run_time.PATH_BERT_BASE_DIR = PATH_BERT_BASE_DIR
            for remove_some_tasks in [True]:#是否在训练过程中删除权重较低的任务
                for train_set_size in [100]:#21064

                    for batch_size in [30]:
                        supervised, separated_training = model_plan["supervised"], model_plan["separated_training"]
                        separated_loss = "separated_loss" if model_plan.get("separated_loss")==True else "not_separated_loss"
                        if type(subtask_list)==str:#单个辅助任务
                            dir_name = "../../data/results/stage_II_" + str(train_set_size) + "_smaples_" + "[" + subtask_list + ']'
                        else:
                            if len(subtask_list) + len(PATH_BERT_BASE_DIR.split("/")[-1])> 5: subtask_list = [line.split("_")[-1][-5:] for line in subtask_list]
                            dir_name = "../../data/results/stage_II_" + str(train_set_size) + "_smaples_" + "[" + "-".join(subtask_list) + ']'
                        if remove_some_tasks==True:
                            dir_name += "_" + "all_remove_task_dynamically"
                        else:
                            dir_name += "_" + "all"
                        dir_name += PATH_BERT_BASE_DIR.split("/")[-1]

                        create_a_dir(dir_name)

                        model_plan["batch_size"] = batch_size
                        print("######################################################")
                        print(model_plan)
                        for test_number in range(10):

                            model_plan["test_number"] = test_number
                            training_records_ori = []
                            init_task_attention =[1.0] * len(run_time.ALL_AUXILIARY_TASK_LIST)

                            trainer = Trainer(supervised=supervised, \
                                              separated_training=separated_training, \
                                              total_step=int(len(run_time.ALL_AUXILIARY_TASK_LIST)*(pretrain_epoch + 1) * train_set_size / batch_size),
                                                                                                               part_of_transformer=part_of_transformer)
                            model, training_records_ori, task_attention = trainer.fit_pretraining(batch_size=batch_size, \
                                                                    train_set_size=train_set_size, devset_size=devset_size, epoch_num=pretrain_epoch,\
                                                                                                                init_task_attention=init_task_attention,
                                                                                                                update_task_weight=True,
                                                                                                                stastics_mode=stastics_mode,
                                                                                                                remove_some_tasks=remove_some_tasks,
                                                                                                                remove_rate=remove_rate)
                            #time.sleep(10)
                            del trainer, model
                            training_records = copy.deepcopy(training_records_ori)
                            print("等待开始微调")
                            main_task_batch_size, aux_task_batch_size = [50, 0]

                            # training_records += finetune(main_task_batch_size, train_set_size, epoch_num=20, class_weight=class_weight)
                            file_name = dir_name + "/" + str(test_number) + ".txt"
                            with open(file_name, "w", encoding='utf8') as f:
                                f.write(json.dumps(model_plan, ensure_ascii=False) + "\n")
                                for training_progress in training_records:
                                    f.write(json.dumps(training_progress, ensure_ascii=False) + "\n")

