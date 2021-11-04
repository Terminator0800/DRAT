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
from job.train_job_baseline import Trainer
import random as rander
rander.seed(666)
import torch
evaluation_result_file = "../../src_v0.9/job/evaluation_results.txt"

def create_a_dir(dir_name):
    if os.path.exists(dir_name):
        #os.rmdir(dir_name)
        os.system("rm -rf " + dir_name)
    os.mkdir(dir_name)

def finetune(main_task_batch_size, trainingset_size, epoch_num=50, class_weight=None):
    model = BERTBaseModel(mode="train", with_bilstm=True)
    model_stage_II_paras = model.state_dict()
    #使用继续与训练过的模型参数初始化
    model_stage_I = torch.load('../../data/models/model_stage_I.pth')
    model_stage_I_paras =model_stage_I#.state_dict()
    model_stage_I_dict = {k: v for k, v in model_stage_I_paras.items() if k in model_stage_II_paras}
    # 更新现有的model_dict
    model_stage_II_paras.update(model_stage_I_dict)
    module_name_in_stage_I = "module.bert."
    main_encoder_paras = {"bert." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)] == module_name_in_stage_I}
    model_stage_II_paras.update(main_encoder_paras)

    module_name_in_stage_I = "module.model_seq.0.encoder."
    main_encoder_paras = {"encoder." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)]==module_name_in_stage_I}
    model_stage_II_paras.update(main_encoder_paras)
    module_name_in_stage_I = "module.model_seq.0.fc."
    main_encoder_paras = {"main_model." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)]==module_name_in_stage_I}
    # print("main_encoder_paras", main_encoder_paras, model_stage_I.keys())
    # print("model.encoder.named_parameters()", list(map(lambda x: x[0], model.named_parameters())))
    model_stage_II_paras.update(main_encoder_paras)
    model.load_state_dict(model_stage_II_paras)
    trainer = Trainer(main_task="text_similarity_LCQMC", with_bilstm=True)
    records = trainer.fit(model, mini_computer=True, if_demo=True, main_task_batch_size=main_task_batch_size, \
                                          trainingset_size=trainingset_size, class_weight=class_weight, epoch_num=epoch_num)
    return records

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"#"1, 0, 2, 3"
    
    run_time.GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
#     run_time.GPU_IDS = [0]
    run_time.ALL_AUXILIARY_TASK_LIST = ["text_similarity_LCQMC"]

    for trainingset_size in [100]:
        for with_bilstm in [True]:
            for main_task_batch_size in [50]:
                dir_name = "../../data/results/stage_II_baseline_" + str(trainingset_size) + "_smaples_"
                if with_bilstm: dir_name += "with_BiLSTM/"
                else: dir_name += "no_BiLSTM/"
                create_a_dir(dir_name)
                for test_number in range(30):
                    model_plan = {"with_bilstm": with_bilstm, "trainingset_size": trainingset_size}
                    model = BERTBaseModel(mode="train", with_bilstm=with_bilstm)
                    model_stage_II_paras = model.state_dict()
                    #使用继续与训练过的模型参数初始化
                    model_stage_I = torch.load('../../data/models/model_stage_I.pth')
                    model_stage_I_paras =model_stage_I#.state_dict()
                    model_stage_I_dict = {k: v for k, v in model_stage_I_paras.items() if k in model_stage_II_paras}
                    # 更新现有的model_dict
                    model_stage_II_paras.update(model_stage_I_dict)
                    module_name_in_stage_I = "module.bert."
                    main_encoder_paras = {"bert." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)] == module_name_in_stage_I}
                    model_stage_II_paras.update(main_encoder_paras)

                    module_name_in_stage_I = "module.model_seq.0.encoder."
                    main_encoder_paras = {"encoder." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)]==module_name_in_stage_I}
                    model_stage_II_paras.update(main_encoder_paras)
                    module_name_in_stage_I = "module.model_seq.0.fc."
                    main_encoder_paras = {"main_model." + k[len(module_name_in_stage_I):]: v for k, v in model_stage_I.items() if k[:len(module_name_in_stage_I)]==module_name_in_stage_I}
                    # print("main_encoder_paras", main_encoder_paras, model_stage_I.keys())
                    # print("model.encoder.named_parameters()", list(map(lambda x: x[0], model.named_parameters())))
                    model_stage_II_paras.update(main_encoder_paras)
                    model.load_state_dict(model_stage_II_paras)
                    if with_bilstm==False: lr = 1e-6
                    else: lr = 1e-6
                    trainer = Trainer(main_task="text_similarity_LCQMC", with_bilstm=with_bilstm)
                    records = trainer.fit(model, mini_computer=True, if_demo=True, main_task_batch_size=main_task_batch_size, \
                                          trainingset_size=trainingset_size)

                    file_name = dir_name + str(test_number) + ".txt"
                    with open(file_name, 'w', encoding='utf8') as f:
                        f.write(json.dumps(model_plan, ensure_ascii=False) + "\n")
                        for record in records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

