#分析
import json
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from training_result_loader import load_training_result_map
from analysis_tools import annalysis_model_evaluation_records,\
                    show_boxes_of_f1scores_when_dataset_size_changes,\
                    show_boxes_of_f1scores_when_pretrained_model_changes,show_boxes_of_scores_when_task_changes,\
                    f1score_deltas_of_each_models,\
                    get_task_infuence_scores
from config.run_time import ALL_AUXILIARY_TASK_LIST
import numpy as np

def models_in_qq_matching_task():
    training_result_map_100 = load_training_result_map("good/ernie/qq_matching/relu_100_samples")
    training_result_map_200 = load_training_result_map("good/ernie/qq_matching/relu_200_samples")
    training_result_map_500 = load_training_result_map("good/ernie/qq_matching/relu_500_samples")
    training_result_map_1000 = load_training_result_map("good/ernie/qq_matching/relu_1000_samples")

    model_socres100 = annalysis_model_evaluation_records(training_result_map_100)
    model_socres200 = annalysis_model_evaluation_records(training_result_map_200)
    model_socres500 = annalysis_model_evaluation_records(training_result_map_500)
    model_socres1000 = annalysis_model_evaluation_records(training_result_map_1000)
    
#     #加载全量数据的训练结果
#     from data_analysis import high_resource_experiment_results
#     model_socres_all = {"test": {"baseline_230000_smaples_with_BiLSTM": {"acc_each_training": []},
#                                  "stage_II_230000_smaples_all_remove_task_dynamically": {"acc_each_training": []}}, \
#                         "dev": {"baseline_230000_smaples_with_BiLSTM": {"acc_each_training": []},
#                                 "stage_II_230000_smaples_all_remove_task_dynamically": {"acc_each_training": []}}}
#     for data in high_resource_experiment_results.drat_data_list:
#         model_socres_all["dev"]["stage_II_230000_smaples_all_remove_task_dynamically"]["acc_each_training"].append(data[1]['accuracy'])
#         model_socres_all["test"]["stage_II_230000_smaples_all_remove_task_dynamically"]["acc_each_training"].append(data[0]['accuracy'])
#     for data in high_resource_experiment_results.stl_data_list:
#         model_socres_all["dev"]["baseline_230000_smaples_with_BiLSTM"]["acc_each_training"].append(data[0]['accuracy'])
#         model_socres_all["test"]["baseline_230000_smaples_with_BiLSTM"]["acc_each_training"].append(data[1]['accuracy'])


    #分析100个样本时，辅助任务单独存在对主任务模型的提升情况
    model_version_task_weight_map = {}#需要把权重加载进来
    task_influence_scores = get_task_infuence_scores(training_result_map_100)
    for i in range(len(ALL_AUXILIARY_TASK_LIST)):
        model_version_task_weight_map[ALL_AUXILIARY_TASK_LIST[i]] = task_influence_scores[i]

    f1score_deltas_of_each_models(model_socres100, model_version_task_weight_map)
    
    #分析训练集增大时，各个版本模型的效果变化
    show_boxes_of_f1scores_when_dataset_size_changes({100: model_socres100, 200: model_socres200,\
                                                      500: model_socres500, 1000: model_socres1000})
    print("model_socres100", model_socres100.keys())
def compare_kinds_of_pretrained_models():
    training_result_map_100_ernie = load_training_result_map("good/ernie/qq_matching/relu_100_samples")
    training_result_map_100_ernie1base = load_training_result_map("good/ernie1_base/qq_matching/relu_100_samples")
    training_result_map_100_bertbase = load_training_result_map("good/bertbase/qq_matching/relu_100_samples")
    training_result_map_100_zen = load_training_result_map("good/zen/qq_matching/relu_100_samples")
    training_result_map_100_chn_roberta_base = load_training_result_map("good/chn_roberta_base/qq_matching/relu_100_samples")
    training_result_map_100_chn_roberta_wwm = load_training_result_map("good/chn_roberta_wwm/qq_matching/relu_100_samples")
    training_result_map_100_chn_macbert = load_training_result_map("good/macbert/qq_matching/relu_100_samples")

    model_socres100_ernie = annalysis_model_evaluation_records(training_result_map_100_ernie)
    model_socres100_ernie1base = annalysis_model_evaluation_records(training_result_map_100_ernie1base)

    model_socres100_bertbase = annalysis_model_evaluation_records(training_result_map_100_bertbase)
    model_socres100_zen = annalysis_model_evaluation_records(training_result_map_100_zen)
    model_socres100_chn_roberta_base = annalysis_model_evaluation_records(training_result_map_100_chn_roberta_base)
    model_socres100_chn_roberta_wwm = annalysis_model_evaluation_records(training_result_map_100_chn_roberta_wwm)
    model_socres100_chn_macbert = annalysis_model_evaluation_records(training_result_map_100_chn_macbert)

    show_boxes_of_f1scores_when_pretrained_model_changes({"bert": model_socres100_bertbase, "zen": model_socres100_zen,
                                                          "rbt": model_socres100_chn_roberta_base, "ernie-g": model_socres100_ernie,
                                                          "rbt_wwm": model_socres100_chn_roberta_wwm,
                                                          "macbert": model_socres100_chn_macbert,
                                                          "ernie1": model_socres100_ernie1base})#, "ERNIE2.0": model_socres100_ernie)

#在开发集中的表现
def model_in_kinds_of_tasks():
    def get_score_list(score_map):
        score_list_map = {}
        for model_name in score_map:
            score_list_map[model_name] = list(map(lambda x:x["accuracy"], \
                                                         score_map[model_name]['dev']["best_records"]))
        return score_list_map


    training_result_map_100_ernie_qq = load_training_result_map("good/ernie/qq_matching/relu_100_samples")
    training_result_map_100_ernie_atec = load_training_result_map("good/ernie/atec/relu_100_samples")
    training_result_map_100_ernie_drcd = load_training_result_map("good/ernie/drcd/relu_100_samples")
    training_result_map_100_ernie_chnsenticorp = load_training_result_map("good/ernie/chnsenticorp/relu_100_samples")
    training_result_map_100_ernie_masr = load_training_result_map("good/ernie/masr/relu_100_samples")

    socres100_ernie_qq = get_score_list(training_result_map_100_ernie_qq)
    socres100_ernie_atec = get_score_list(training_result_map_100_ernie_atec)

    socres100_ernie_drcd = get_score_list(training_result_map_100_ernie_drcd)
    socres100_ernie_chnsenticorp = get_score_list(training_result_map_100_ernie_chnsenticorp)
    socres100_ernie_masr = get_score_list(training_result_map_100_ernie_masr)

    show_boxes_of_scores_when_task_changes({"ATEC": socres100_ernie_atec,
                                                          "DRCD": socres100_ernie_drcd,
                                              "chnsenticorp": socres100_ernie_chnsenticorp,
                                            "masr": socres100_ernie_masr
    })#"LCQMC": socres100_ernie_qq,

def if_highway_for_main_task_model_works():
    def get_score_list(score_map):
        score_list_map = {}
        for model_name in score_map:
            score_list_map[model_name] = list(map(lambda x:x["accuracy"], \
                                                         score_map[model_name]['dev']["best_records"]))
        return score_list_map
    def key_metrics(data):
        print("median", np.median(data)*100, "max", np.max(data)*100 , "std", np.std(data)*100)
    training_result_map_100_ernie_qq = load_training_result_map("good/ernie/qq_matching/relu_100_samples")
    socres100_ernie_qq = get_score_list(training_result_map_100_ernie_qq)
    key_metrics(socres100_ernie_qq['stage_II_100_smaples_all_remove_task_dynamically'])
    key_metrics(socres100_ernie_qq['stage_II_100_smaples_all_no_highway'])

def annalysis_all():
    # models_in_qq_matching_task()#分析各种模型在LCQMC数据集中的表现
    # compare_kinds_of_pretrained_models()#用LCQMC比较各种预训练模型
    model_in_kinds_of_tasks()#考察模型在各类主任务中的表现
    # if_highway_for_main_task_model_works()#if highway for main task model works

def online_annalysis(model_name):
    training_result_map_100 = load_training_result_map(model_name)
    model_socres100 = annalysis_model_evaluation_records(training_result_map_100)
    print("model_socres100", model_socres100)
    f1score_deltas_of_each_models(model_socres100, model_version_task_weight_map)

if __name__ == '__main__':
    annalysis_all()#全体训练的chuqi
    # online_annalysis("good/ernie/qq_matching/relu_100_samples")
#     training_result_map_100_ernie = load_training_result_map("good/ernie/qq_matching/relu_100_samples")
#     get_task_infuence_scores(training_result_map_100_ernie)


