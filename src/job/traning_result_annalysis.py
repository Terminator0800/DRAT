#分析
import json
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from scipy.stats import pearsonr, kendalltau
from config import run_time
def readlines(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = list(map(lambda x: x[:-1], lines))
    return lines

def read_json_lines(file_name):
    lines = readlines(file_name)
    data_list = list(map(lambda x: json.loads(x), lines))
    return data_list

def find_best(evaluation_records):
    best_record = evaluation_records[0]
    for record in evaluation_records[1:]:
       # if record["f1score"] + record["accuracy"] > best_record["f1score"] + best_record["accuracy"]:
        if record["f1score"] > best_record["f1score"]:
            best_record = record
    f1score_list = list(map(lambda x: float(x['f1score']) if "f1score" in x else float(x[" f1score"]), evaluation_records))
    return best_record, f1score_list

def find_best_and_worst_training_process(evaluation_record_of_this_model):
    best_f1score_of_each_training = list(map(lambda x: x["f1score"], evaluation_record_of_this_model["best_records"]))
    best_of_best_index =  np.argmax(best_f1score_of_each_training)
    worst_of_best_index =np.argmin(best_f1score_of_each_training)
    # print(evaluation_record_of_this_model["f1score_list"])
    best_f1score_list = evaluation_record_of_this_model["f1score_list"][best_of_best_index]
    worst_f1score_list = evaluation_record_of_this_model["f1score_list"][worst_of_best_index]

    #寻找上下界
    upper_limit, lower_limit = [0]*len(best_f1score_list), [1]*len(best_f1score_list)
    for i in range(len(evaluation_record_of_this_model["f1score_list"])):
        for j in range(len(best_f1score_list)):
            f1score = evaluation_record_of_this_model["f1score_list"][i][j]
            if f1score>upper_limit[j]: upper_limit[j] = f1score
            if f1score < lower_limit[j]: lower_limit[j] = f1score
    return best_f1score_list, worst_f1score_list, upper_limit, lower_limit

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

task_weights = [ 0.060859293694607755, 0.058656594083555266, 0.060919967194239165, 0.05928557021381478, 0.059972745068042524, 0.05849676494177433, 0.06405425184469843, 0.06158459209874372, 0.05937589280178543, 0.05873616913284458, 0.060658238760546324, 0.053736176076666795, 0.051796096844469834, 0.060171375008242586, 0.054260209041521575, 0.05831216306841963]
task_weights = [0.05726450189165514, 0.05470747454261585, 0.05981391854602835, 0.054545454642796594, 0.06732121540052063, 0.05516467146294846, 0.06515819356710463, 0.05485635643574852, 0.060190274276007025, 0.05457315208461723, 0.05725471665083351, 0.05371345848089612, 0.05558951276901961, 0.05949811930337341, 0.055550648939912656, 0.05666407896497732]
task_weights = [0.025771077540346146, 0.026332157600739085, 0.025828886479571638, 0.02681416009284454, 0.026175997669633688, 0.01893328254857739, 0.019144815031096357, 0.021168098073718524, 0.01958109503550995, 0.025574449585901014, 0.02471712862147017, 0.0271014168714739, 0.01434760242361509, 0.018154532068103003, 0.02013805966065801, 0.018581384486735388, 0.023774539649582673, 0.02081860165200705, 0.0157528401357835, 0.015995141713862463, 0.015944856360228094, 0.02543874720955274, 0.02544756067936781, 0.026143927474457193, 0.025067026946326287, 0.025851214008701225, 0.026363038828335815, 0.02568473736482505
                ]
task_names =  ["text_classification_THUCNews", "text_smililarity_atec", "text_classification_toutiao","text_similarity_CCKS2018",
                       "text_classification_tansongbo_hotel_comment", "NER_rmrb", "NER_CLUENER_public", "MRC_DRCD", "MRC_cmrc2018", \
                           "text_classification_chnsenticorp", "text_similarity_ChineseSTS", "text_classification_simplifyweibo_4_moods",
                           "NER_MSAR", "NER_rmrb_2014", "NER_boson", "NER_weibo", "MRC_chinese_SQuAD", "MRC_CAIL2019", "NER_CMNER",
                           "NER_CCKS2019_task1_Yidu_S4K", "NER_baidu2020_event", "NLI_cmnli_public", "NLI_ocnli_public", "sentiment_analysis_dmsc_v2",
                             "sentiment_analysis_online_shopping_10_cats", "sentiment_analysis_waimai_10k", "sentiment_analysis_weibo_senti_100k",
                           "sentiment_analysis_yf_dianping"]
task_name_weight_map =  {'text_similarity_LCQMC': 90, 'text_classification_THUCNews': 96, 'text_smililarity_atec': 86, 'text_classification_toutiao': 87, 'text_similarity_CCKS2018': 88, 'text_classification_tansongbo_hotel_comment': 90, 'NER_rmrb': 88, 'NER_CLUENER_public': 91, 'MRC_DRCD': 92, 'MRC_cmrc2018': 98, 'text_classification_chnsenticorp': 94, 'text_similarity_ChineseSTS': 87, 'text_classification_simplifyweibo_4_moods': 87, 'NER_MSAR': 91, 'NER_rmrb_2014': 84, 'NER_boson': 86, 'NER_weibo': 91, 'MRC_chinese_SQuAD': 85, 'MRC_CAIL2019': 90, 'NER_CMNER': 92, 'NER_CCKS2019_task1_Yidu_S4K': 88, 'NER_baidu2020_event': 88, 'NLI_cmnli_public': 91, 'NLI_ocnli_public': 92, 'sentiment_analysis_dmsc_v2': 89, 'sentiment_analysis_online_shopping_10_cats': 94, 'sentiment_analysis_waimai_10k': 86, 'sentiment_analysis_weibo_senti_100k': 90, 'sentiment_analysis_yf_dianping': 86}

task_name_weight = list(zip(task_weights, task_names))
task_name_weight = sorted(task_name_weight, key=lambda x: x[0], reverse=False)
print(task_name_weight)
task_name_weight = list(map(lambda x: x[1], task_name_weight))
print(task_name_weight)
# task_names = ["text_classification_THUCNews", "text_classification_toutiao",
#                        "text_classification_tansongbo_hotel_comment", \
#                            "text_classification_chnsenticorp", "text_classification_simplifyweibo_4_moods"]
# task_weights = [1 for i in range(len(task_names))]
model_version_task_name_map = {"stage_II_100_smaples_[text_classification_THUCNews]": "text_classification_THUCNews",
                                 "stage_II_100_smaples_[text_smililarity_atec]": "text_smililarity_atec",
                                 "stage_II_100_smaples_[text_classification_toutiao]": "text_classification_toutiao",
                                 "stage_II_100_smaples_[text_similarity_CCKS2018]": "text_similarity_CCKS2018",
                                 "stage_II_100_smaples_[text_classification_tansongbo_hotel_comment]": "text_classification_tansongbo_hotel_comment",
                                "stage_II_100_smaples_[NER_rmrb]": "NER_rmrb",
                               "stage_II_100_smaples_[NER_rmrb_2014]": "NER_rmrb_2014",
                                "stage_II_100_smaples_[NER_CLUENER_public]": "NER_CLUENER_public",
                                "stage_II_100_smaples_[MRC_DRCD]": "MRC_DRCD",
                                "stage_II_100_smaples_[MRC_cmrc2018]": "MRC_cmrc2018",
                                "stage_II_100_smaples_[text_classification_chnsenticorp]": "text_classification_chnsenticorp",
                                 "stage_II_100_smaples_[text_classification_simplifyweibo_4_moods]": "text_classification_simplifyweibo_4_moods",
                                 "stage_II_100_smaples_[text_similarity_ChineseSTS]": "text_similarity_ChineseSTS",
                               "stage_II_100_smaples_[NER_boson]": "NER_boson",
                               "stage_II_100_smaples_[NER_weibo]": "NER_weibo",
                               "stage_II_100_smaples_[NER_MSAR]": "NER_MSAR",
                               "stage_II_100_smaples_[MRC_DRCD-NER_boson-NER_MSAR-NER_rmrb]": "MRC_DRCD-NER_boson-NER_MSAR-NER_rmrb",
                                 "stage_II_100_smaples_[text_smililarity_atec-text_similarity_CCKS2018-text_classification_toutiao-text_classification_THUCNews]":
                                    "text_smililarity_atec-text_similarity_CCKS2018-text_classification_toutiao-text_classification_THUCNews",
                               "stage_II_100_smaples_[THUCNews-atec-toutiao-CCKS2018-comment-rmrb-public-DRCD-cmrc2018-chnsenticorp-ChineseSTS-moods-MSAR-2014-boson-weibo]": "all",
                               "stage_II_100_smaples_[MRC_CAIL2019]": "MRC_CAIL2019",
                               "stage_II_100_smaples_[MRC_chinese_SQuAD]": 'MRC_chinese_SQuAD',
                               "stage_II_100_smaples_[NER_CMNER]":"NER_CMNER",
                               "stage_II_100_smaples_[NER_CCKS2019_task1_Yidu_S4K]":"NER_CCKS2019_task1_Yidu_S4K",
                                "stage_II_100_smaples_[NER_baidu2020_event]":"NER_baidu2020_event",
                                "stage_II_100_smaples_[NLI_cmnli_public]":"NLI_cmnli_public",
                                "stage_II_100_smaples_[NLI_ocnli_public]":"NLI_ocnli_public",
                                "stage_II_100_smaples_[sentiment_analysis_dmsc_v2]":"sentiment_analysis_dmsc_v2",
                               "stage_II_100_smaples_[sentiment_analysis_online_shopping_10_cats]":
                                   "sentiment_analysis_online_shopping_10_cats",
                               "stage_II_100_smaples_[sentiment_analysis_waimai_10k]":"sentiment_analysis_waimai_10k",
                                         "stage_II_100_smaples_[sentiment_analysis_weibo_senti_100k]": "sentiment_analysis_weibo_senti_100k",
                               "stage_II_100_smaples_[sentiment_analysis_yf_dianping]":"sentiment_analysis_yf_dianping"
                               }
model_version_task_name_map = {v: k for k, v in model_version_task_name_map.items()}
print(model_version_task_name_map)
model_version_task_weight_map = {}
for i in range(len(task_weights)):
    #model_version_task_weight_map[model_version_task_name_map[task_names[i]]] = task_weights[i]
    model_version_task_weight_map[model_version_task_name_map[task_names[i]]] = task_name_weight_map[task_names[i]]
model_version_task_type = {}
for i in range(len(task_weights)):
    model_version = model_version_task_name_map[task_names[i]]
    model_version_task_type[model_version] = run_time.AUXILIARY_TASK_TYPE_MAP.get(task_names[i], "other")
    print(task_names[i], model_version)
model_version_task_type["baseline_100_smaples_with_BiLSTM"] = "CLS"

def annalysis_model_evaluation_records():
    record_dir = "../../data/results/"
    result_map = {}
    for model_version in os.listdir(record_dir):
        if model_version=="good": continue
        model_dir = record_dir + model_version + "/"
        file_names = os.listdir(model_dir)
        print(model_version, 'asd',file_names)
        if len(file_names)==0 or model_version not in model_version_task_type:
            print("@@@@@@@@@@@@", model_version)
           # continue
        for file_name in file_names:
            file_name = model_dir + file_name
            if model_version not in result_map:
                result_map[model_version] = {
                                            "dev": {"f1score_list": [], "best_records": [] , "worst_training_process": [], "best_training_process": [], \
                                                           "upper_limit": [], "lower_limit": [], "task_weights": []},\
                                             'test': {"f1score_list": [], "best_records": [], "worst_training_process": [], "best_training_process": [], \
                                                      "upper_limit": [], "lower_limit": [],"task_weights": [] }
                                             }
            record_list = read_json_lines(file_name)
            model_plan = record_list[0]
            train_records, dev_records, test_records = [], [], []
            dev_task_weight, test_task_weight = 0, 0
            for record in record_list:
                if " f1score" in record: record["f1score"] = record[" f1score"]
                if ("top_or_base_model" in record and record['top_or_base_model']=="基层模型") or "stage_II" in file_name or 'baseline' in file_name:
                    #continue
                    if "stage" in record  and record["stage"]=="dev": dev_records.append(record)
                    if "stage" in record and record["stage"] == "test":
                        test_records.append(record)
                        #print("record", record)
                if 'task_weight' in record:
                    dev_task_weight, test_task_weight = record["task_weight"][1], record["task_weight"][1]
            result_map[model_version]['dev']['task_weights'].append(dev_task_weight)
            result_map[model_version]['test']['task_weights'].append(test_task_weight)
            # print(len(record_list), "dev_records len", len(dev_records), model_version)
            for records in [train_records, dev_records, test_records]:
                if len(records)>0:
                    stage = records[0]['stage']
                    evaluation_records = list(filter(lambda x: "accuracy" in x, records))
                    best_record, f1score_list = find_best(evaluation_records)
                    result_map[model_version][stage]["f1score_list"].append(f1score_list)
                    result_map[model_version][stage]["best_records"].append(best_record)

            for key in ["dev"]:#result_map[model_version]:
                if len(result_map[model_version][key]["f1score_list"])==0: continue
                best_f1score_list, worst_f1score_list, upper_limit, lower_limit = find_best_and_worst_training_process(result_map[model_version][key])
                result_map[model_version][key]["worst_training_process"] = worst_f1score_list
                result_map[model_version][key]["best_training_process"] = best_f1score_list
                result_map[model_version][key]["upper_limit"] = upper_limit
                result_map[model_version][key]["lower_limit"] = lower_limit

    trainingset_sizes = list(range(1, 1000))
    from numpy import random, linspace
    colors = plt.cm.plasma(linspace(0, 1, len(result_map)*2))
    random.shuffle(colors)
    model_no = 0
    model_socres = {"test": {},  'dev': {}}
    f1score_list_map = {"test": {},  'dev': {}}
    task_weight_f1score_pairs = []
    print("result_map", result_map.keys())
    for model_version in result_map:

        for mode in ['dev', "test"]:

            color = colors[model_no]
            # print(mode, result_map[model_version])
            print("model_version", model_version)#, result_map[model_version]["dev"]["f1score_list"][0])
            best_record_index = np.argmax(list(map(lambda x: x["f1score"] + x["accuracy"], result_map[model_version][mode]["best_records"])))

            best_record = result_map[model_version][mode]["best_training_process"]
            worst_record = result_map[model_version][mode]["worst_training_process"]
            upper_limit = result_map[model_version][mode]["upper_limit"]
            lower_limit = result_map[model_version][mode]["lower_limit"]
            best_model_score = result_map[model_version][mode]["best_records"][best_record_index]
            f1scores_of_this_model = [x['f1score'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            accuracy_of_this_model = [x['accuracy'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            precision_of_this_model = [x['precision'] for x in result_map[model_version][mode]["best_records"] if x['epoch']>=0]
            recall_of_this_model = [x['recall'] for x in result_map[model_version][mode]["best_records"] if x['epoch'] >= 0]

            # print("f1scores_of_this_model", f1scores_of_this_model)
            print(mode, result_map[model_version][mode]["f1score_list"])
            mean_f1scores = mean_of_lists(result_map[model_version][mode]["f1score_list"])
            print("f1: max ", np.max(f1scores_of_this_model), "mean ", np.mean(f1scores_of_this_model), "std ", np.std(f1scores_of_this_model))
            print("accuracy: max ", np.max(accuracy_of_this_model), "mean ", np.mean(accuracy_of_this_model), "std ",np.std(accuracy_of_this_model))
            print("precision: max ", np.max(precision_of_this_model), "mean ", np.mean(precision_of_this_model), "std ",np.std(precision_of_this_model))
            print("recall: max ", np.max(recall_of_this_model), "mean ", np.mean(recall_of_this_model), "std ", np.std(recall_of_this_model))
            print("f1scores_of_this_model", f1scores_of_this_model)
            # print("best_model_score", best_model_score)
            socres_of_this_model = {"model_name": model_version, "accuracy": best_model_score['accuracy']*100, "recall": best_model_score['recall']*100,
                                    "precision": best_model_score['precision']*100, "F1score": best_model_score['f1score']*100,
                                    "F1score_mean": np.mean(f1scores_of_this_model)*100, "F1socre_std": np.std(f1scores_of_this_model)*100,
                                    "task_weight_std": np.std(result_map[model_version][mode].get("task_weights", [])),
                                    "task_type": model_version_task_type.get(model_version, "UNK"),
                                    "F1score_median": np.median(f1scores_of_this_model)*100,
                                    # "task_weight": np.mean(result_map[model_version][mode].get("task_weights", []))}
                                    "task_weight": model_version_task_weight_map.get(model_version, 0)}  #
            model_socres[mode][model_version] = socres_of_this_model
            f1score_list_map[mode][model_version] = f1scores_of_this_model
            if mode=="test" and model_version!="baseline_100_smaples_with_BiLSTM":
                task_weight_f1score_pairs.append([result_map[model_version][mode].get("task_weights", []), f1scores_of_this_model])
            x_labels = trainingset_sizes[: len(mean_f1scores)]
            plt.plot(x_labels, mean_f1scores, label=model_version + "_" + mode, color=color, mfc=color)
            # print(worst_record)
            print("############################")
            if len(lower_limit)>0:
                plt.plot(x_labels, lower_limit, color=color, mfc=color, linestyle="dashdot")
            if len(upper_limit)>0:
                plt.plot(x_labels, upper_limit, color=color, mfc=color, linestyle="dashdot")
            if len(x_labels)==len(lower_limit):
                plt.fill_between(x_labels, lower_limit, upper_limit, color=color, alpha=0.3)#, facecolor='green'
            model_no += 1
    plt.ylim([0.65, 0.85])
    plt.legend()
    plt.show()

    #增加豫baseline差异以及显著性
    model_name_list = []
    print("f1score_list_map", f1score_list_map["test"].keys())
    for model_name in f1score_list_map["test"]:
        p = ttest_ind(f1score_list_map["test"]["baseline_100_smaples_with_BiLSTM"], f1score_list_map["test"][model_name])
        delta = model_socres['test'][model_name]['F1score_median'] - model_socres['test']['baseline_100_smaples_with_BiLSTM']["F1score_median"]
        model_socres['test'][model_name]["p"] = p[1]
        model_socres['test'][model_name]["delta"] = delta
    model_socre_list = list(model_socres['test'].values())
    df_test = pd.DataFrame(model_socre_list)
    df_test = df_test.round(2)
    df_test.to_csv("在测试集中的表现.csv")
    del model_socres['test']["baseline_100_smaples_with_BiLSTM"]
    # del model_socres['test']["baseline_100_smaples_with_BiLSTM_large_lr"]
    # del model_socres['test']["stage_II_100_smaples_[text_similarity_CCKS2018]"]
    # del model_socres['test']["stage_II_100_smaples_[text_similarity_ChineseSTS]"]

    df_test = pd.DataFrame(model_socres['test'].values())
    delta = list(df_test["delta"])# + list(df_test["delta"])
    task_weight = list(df_test["task_weight"]) #+ list(df_test["task_weight"])
    delta_new, task_weight_new = [], []
    for d, w in zip(delta, task_weight):
        if w>0:
            delta_new.append(d)
            task_weight_new.append(w)
    delta, task_weight = delta_new, task_weight_new
    print(task_weight)
    print(pearsonr(delta,task_weight))
    print(kendalltau(delta, task_weight))
    plt.plot(task_weight, delta, marker="*")
    plt.legend()
    plt.show()

    # #show the relation between task weight and f1score
    # task_weights, fsscores = [], []
    # for line in task_weight_f1score_pairs:
    #     task_weights.extend(line[0])
    #     fsscores.extend(line[1])
    # print("relation od task weight and f1", pearsonr(task_weights, fsscores))
    # plt.plot(task_weights, fsscores, marker="*")
    # plt.legend()
    # plt.show()

    #plot box of each task's f1s-scores
    import seaborn as sns
    data = []
    for model_name in f1score_list_map["test"]:
        for f1_score in f1score_list_map["test"][model_name]:
            data.append({"model": model_name, "f1_score":f1_score})
    data = pd.DataFrame(data)
    fig, axes = plt.subplots()
    sns.boxplot(x="model", y="f1_score", data=data, ax=axes)
    plt.show()



if __name__ == '__main__':
    annalysis_model_evaluation_records()#全体训练的chuqi
