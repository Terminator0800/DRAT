'''
Created on 2021年9月17日

@author: Administrator
'''
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, norm
import seaborn as sns
import re
from scipy.stats import pearsonr, kendalltau

def mean_of_lists(data_lists):
    mean_list = data_lists[0]
    try:
        for data_list in data_lists[1:]:
            for i in range(len(data_list)):
                mean_list[i] += data_list[i]
        for i in range(len(mean_list)):
            mean_list[i] /= len(data_lists)
    except:
        pass
    return mean_list

def median_of_lists(data_lists):
    m_list = [[] for _ in range(len(data_lists[0]))]
    for data_list in data_lists[1:]:
        for i in range(len(data_list)):
            m_list[i].append(data_list[i])
    for i in range(len(data_lists[0])):
        m_list[i] = np.median(m_list[i])
    return m_list

# def plot_upper_lower_limits():
#     colors = plt.cm.plasma(linspace(0, 1, len(result_map)*2))
#     random.shuffle(colors)
# color = colors[model_no]
#     plt.plot(x_labels, lower_limit, color=color, mfc=color, linestyle="dashdot")
#     plt.plot(x_labels, upper_limit, color=color, mfc=color, linestyle="dashdot")
#     plt.fill_between(x_labels, lower_limit, upper_limit, color=color, alpha=0.3)#, facecolor='green'
#     plt.ylim([0.65, 0.85])
#     plt.legend()
#     plt.show()

def annalysis_model_evaluation_records(result_map, model_version_task_weight_map={}):

    from numpy import random, linspace
    model_socres = {"test": {},  'dev': {}}
    f1score_list_map = {"test": {},  'dev': {}}
    for model_version in result_map:

        for mode in ['dev', "test"]:
            # print("model_version", model_version, result_map[model_version][mode]["best_records"])#, result_map[model_version]["dev"]["f1score_list"][0])
            best_record_index = np.argmax(list(map(lambda x: x["f1score"] + x["accuracy"], result_map[model_version][mode]["best_records"])))

            best_model_score = result_map[model_version][mode]["best_records"][best_record_index]
            f1scores_of_this_model = [x['f1score'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            accuracy_of_this_model = [x['accuracy'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            precision_of_this_model = [x['precision'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            recall_of_this_model = [x['recall'] for x in result_map[model_version][mode]["best_records"] if x['epoch'] >=0]

            print("f1scores_of_this_model", f1scores_of_this_model)
            print(mode, model_version)
            print("f1: max ", np.max(f1scores_of_this_model), "mean ", np.mean(f1scores_of_this_model), "std ", np.std(f1scores_of_this_model))
            print("accuracy: max ", np.max(accuracy_of_this_model), "mean ", np.mean(accuracy_of_this_model), "std ",np.std(accuracy_of_this_model))
            print("precision: max ", np.max(precision_of_this_model), "mean ", np.mean(precision_of_this_model), "std ",np.std(precision_of_this_model))
            print("recall: max ", np.max(recall_of_this_model), "mean ", np.mean(recall_of_this_model), "std ", np.std(recall_of_this_model))
            # print("best_model_score", best_model_score)
            socres_of_this_model = {"model_name": model_version, "accuracy": best_model_score['accuracy']*100, "recall": best_model_score['recall']*100,
                                    "precision": best_model_score['precision']*100,\
                                     "F1score": best_model_score['f1score']*100,
                                    "F1score_mean": np.mean(f1scores_of_this_model)*100, \
                                    "F1socre_std": np.std(f1scores_of_this_model)*100,
                                    "f1scores_each_training": f1scores_of_this_model,
                                    "acc_each_training": accuracy_of_this_model,
                                    "task_weight_std": np.std(result_map[model_version][mode].get("task_weights", [])),
                                    "F1score_median": np.median(f1scores_of_this_model)*100,
                                    "acc_median": np.median(accuracy_of_this_model) * 100,
                                    # "task_weight": np.mean(result_map[model_version][mode].get("task_weights", []))}
                                    "task_weight": model_version_task_weight_map.get(model_version, 0)}  #
            model_socres[mode][model_version] = socres_of_this_model
            f1score_list_map[mode][model_version] = f1scores_of_this_model
            print("############################")
    
    #计算测试集中，各模型与基线模型的差异，以及显著性 
    for model_name in f1score_list_map["test"]:
        p = ttest_ind(f1score_list_map["test"]["baseline_100_smaples_with_BiLSTM"], f1score_list_map["test"][model_name])
        delta = model_socres['test'][model_name]['acc_median'] - model_socres['test']['baseline_100_smaples_with_BiLSTM']["acc_median"]
        model_socres['test'][model_name]["p"] = p[1]
        model_socres['test'][model_name]["delta"] = delta
    df = pd.DataFrame(model_socres['dev'])
    df.to_excel("evaluation.xlsx", "dev")
    df = pd.DataFrame(model_socres['test'])
    df.to_excel("evaluation.xlsx", "test")
    return model_socres

#将不同训练集大小时的模型f1水平使用箱型图展示出来
def show_boxes_of_f1scores_when_dataset_size_changes(datasize_model_socres_map, mode='test'):
    datasize_model_socres = datasize_model_socres_map.items()
    datasize_model_socres = sorted(datasize_model_socres, key=lambda x: x[0])
    subfigure_num = len(datasize_model_socres)
    fig, axes = plt.subplots()
    model_name_set = set({})
    for i in range(subfigure_num):
        data = []
        subfigure_id = i + 1
        #plt.subplot(subfigure_num,1,subfigure_id)
#         print(datasize_model_socres[i][1].keys())
        print(datasize_model_socres[i][1][mode].keys())
        for model_name in datasize_model_socres[i][1][mode]:
            training_set_size = datasize_model_socres[i][0]
            if str(training_set_size) not in model_name:
                continue
            if model_name in model_name_set: continue
            model_name_set.add(model_name)
            # print(model_name)
            if "baseline" in model_name or "all" in model_name:
                print(subfigure_id, model_name)
                for acc in datasize_model_socres[i][1][mode][model_name]["acc_each_training"]:
                    model_type = model_name[-5:]#.split("samples")[-1]
                    if model_type=="ual_w": model_type = "EATW"
                    if model_type == "iLSTM": model_type = "STL"
                    if model_type == "s_all": model_type = "DATW"
                    if model_type == "cally": model_type = "DRAT"
                    data.append({"train method": model_type, "ACC":acc, \
                                 "training set size": datasize_model_socres[i][0]})
        #print("data", data)
        data = pd.DataFrame(data)

        # sns.boxplot(x="training set size", y="ACC", data=data, ax=axes, hue="train method")
        bp1 = axes.boxplot(data[data["train method"] == "STL"]["ACC"] * 100, positions=[0.3 + i*2], widths=0.3,medianprops={'color':'black', 'linewidth': 1.0},
                           manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="C0"))
        bp2 = axes.boxplot(data[data["train method"] == "EATW"]["ACC"] * 100, positions=[0.7 + i*2], widths=0.3,medianprops={'color':'black', 'linewidth': 1.0},
                           manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="C1"))
        bp3 = axes.boxplot(data[data["train method"] == "DATW"]["ACC"] * 100, positions=[1 + i*2], widths=0.3,medianprops={'color':'black', 'linewidth': 1.0},
                           manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="C2"))
        bp4 = axes.boxplot(data[data["train method"] == "DRAT"]["ACC"] * 100, positions=[1.3 + i*2], widths=0.3,medianprops={'color':'black', 'linewidth': 1.0},
                           manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="C3"))

    axes.legend([bp1["boxes"][0], bp2['boxes'][0], bp3['boxes'][0], bp4['boxes'][0]], ["STL", "EATW", "DATW", "DRTA"], loc="lower right")
    plt.vlines(2, 65, 82, color="k", linestyle=':')  # 竖线
    plt.vlines(4, 65, 82, color="k", linestyle=':')  # 竖线
    plt.vlines(6, 65, 82, color="k", linestyle=':')  # 竖线
    plt.xticks([1, 3, 5, 7], [100, 200, 500, 1000])
    plt.xlabel("training set size")
    plt.ylabel("accuracy(%)")
    fig.tight_layout()
    plt.show()


#将不同训练集大小时的模型f1水平使用箱型图展示出来
def show_boxes_of_f1scores_when_pretrained_model_changes(pretrained_model_name_model_socres_map, mode='test'):
    pretrained_model_names = pretrained_model_name_model_socres_map.items()
    pretrained_model_names = list(sorted(pretrained_model_names, key=lambda x: x[0]))
    subfigure_num = len(pretrained_model_names)
    data = []
    model_name_set = set({})
    fig, axes = plt.subplots()
    pretrained_models = []
    for i in range(subfigure_num):
        subfigure_id = i + 1
        #plt.subplot(subfigure_num,1,subfigure_id)
#         print(datasize_model_socres[i][1].keys())
#         print("pretrained_model_names",  pretrained_model_names[i][0], pretrained_model_names[i][1][mode].keys())
        data = []
        for model_name in pretrained_model_names[i][1][mode]:
            # print(model_name)
            if "baseline" in model_name or "dynamically" in model_name:

                for acc in pretrained_model_names[i][1][mode][model_name]["acc_each_training"]:
                    if "baseline" in model_name:
                        model_type = 'STL'
                    else:
                        model_type = "DRAT"
                    data.append({"train method": model_type, "ACC":acc, \
                                 "pretrained_model": pretrained_model_names[i][0]})
        pretrained_models.append(pretrained_model_names[i][0])
        data = pd.DataFrame(data)

        bp1 = axes.boxplot(data[data["train method"]=="STL"]["ACC"]*100, positions=[0.3 + i], widths=0.3,manage_ticks=False, \
                           patch_artist=True, boxprops=dict(facecolor="C0"),medianprops={'color':'black', 'linewidth': 1.0})
        bp2 = axes.boxplot(data[data["train method"] == "DRAT"]["ACC"]*100, positions=[0.7 + i],widths=0.3, manage_ticks=False, \
                           patch_artist=True, boxprops=dict(facecolor="C2"),medianprops={'color':'black', 'linewidth': 1.0})
    # sns.boxplot(x="pretrained_model", y="ACC", data=data, ax=axes, hue="train method")
    plt.vlines(1, 50, 77, color="k", linestyle=':')  # 竖线
    plt.vlines(2, 50, 77, color="k", linestyle=':')  # 竖线
    plt.vlines(3, 50, 77, color="k", linestyle=':')  # 竖线
    plt.vlines(4, 50, 77, color="k", linestyle=':')  # 竖线
    plt.vlines(5, 50, 77, color="k", linestyle=':')  # 竖线
    plt.vlines(6, 50, 77, color="k", linestyle=':')  # 竖线
    axes.legend([bp1["boxes"][0], bp2['boxes'][0]], ["STL", "DRTA"], loc="upper right")
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], pretrained_models)
    plt.xlabel("pretrained models")
    plt.ylabel("accuracy(%)")
    fig.tight_layout()
    plt.show()

def show_boxes_of_scores_when_task_changes(task_name_model_socres_map):
    task_name_scores = task_name_model_socres_map.items()
    task_name_scores = list(sorted(task_name_scores, key=lambda x: x[0]))
    subfigure_num = len(task_name_scores)

    fig, axes = plt.subplots()

    for i in range(subfigure_num):
        task_name, scores = task_name_scores[i]
        data = []
        for model_name in scores:
            if "baseline" in model_name or "dynamically" in model_name:
                print("任务", task_name, "分数", "模型", model_name, scores[model_name])
                for acc in scores[model_name]:
                    if "baseline" in model_name:
                        model_type = 'STL'
                    else:
                        model_type = "DRAT"
                    data.append({"train method": model_type, "ACC": acc, \
                                 "task": task_name})
        data = pd.DataFrame(data)
        axes = plt.subplot(1, subfigure_num, i+1)
        print(data[data["train method"]=="STL"]["ACC"])
        if task_name == "masr": rate = 1
        else: rate=100
        bp1 = axes.boxplot(data[data["train method"]=="STL"]["ACC"]*rate, positions=[1], widths=0.45, patch_artist=True,\
                           boxprops=dict(facecolor="C0"),medianprops={'color':'black', 'linewidth': 1.0})
        bp2 = axes.boxplot(data[data["train method"] == "DRAT"]["ACC"]*rate, positions=[2],widths=0.45,  patch_artist=True, \
                           boxprops=dict(facecolor="C2"),medianprops={'color':'black', 'linewidth': 1.0})
        plt.xticks([1.5], [task_name])
        if task_name=="masr":
            plt.ylabel("F1 score")
        else:
            plt.ylabel("accuracy(%)")
        # plt.setp(axes.)
    axes.legend([bp1["boxes"][0], bp2['boxes'][0]], ["STL", "DRAT"], loc="upper right", bbox_to_anchor=[0.1, 1.2])
    fig.text(0.5, 0.01, 'task', ha='center')
    fig.tight_layout()
    plt.show()

#分析每个模型的f1值，与基线模型的差距delta，以及delta与任务权重的相关性
def f1score_deltas_of_each_models(model_scores_map, task_weight_map):
    model_scores_map = copy.deepcopy(model_scores_map)
    baseline_model_f1 = model_scores_map['test']["baseline_100_smaples_with_BiLSTM"]["F1score_mean"]
    del model_scores_map['test']["baseline_100_smaples_with_BiLSTM"]
    data = []
    for model_name in model_scores_map['test']:
        if "all" in model_name: continue#多任务的不包括
        print(model_name)
        this_model_f1 = model_scores_map['test'][model_name]["F1score_mean"]
        data.append({"model": model_name, "delta":this_model_f1 -  baseline_model_f1, \
                     "task_weight": task_weight_map.get(model_name.split("[")[-1][:-1], 0)})
    data = pd.DataFrame(data)
    data.to_excel("任务权重计算结果.xlsx")
    print(data)
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.boxplot(data["delta"], patch_artist=True,\
                           boxprops=dict(facecolor="C0"),medianprops={'color':'black', 'linewidth': 1.0})

    plt.ylabel("increase(%)")
    plt.xlabel("all tasks")
    plt.xticks([], [])
    plt.title("(a)")
    plt.subplot(1,2,2)
    plt.scatter(data["task_weight"], data["delta"], marker="*")
    plt.xlabel("task influence")
    plt.ylabel("increase(%)")
    plt.vlines(0.003, -1, 0.5, color = "r", linestyle = ':')  # 竖线
    plt.title("(b)")
    fig.tight_layout()
    plt.show()
    print("任务权重与主任务模型提升幅度的相关性：", pearsonr(data["task_weight"], data["delta"]))

def get_task_infuence_scores(training_result_map_100):
    #基于100训练样本的训练结果计算任务影响力得分
    data = training_result_map_100['stage_II_100_smaples_all']
#     print(data['dev']["task_weights"])
    
    influence_scores = np.array(data['dev']["task_weights"][0])
    for line in data['dev']["task_weights"][1:]:
        influence_scores += np.array(line)
    return influence_scores/len(data['dev']["task_weights"])
    
    
