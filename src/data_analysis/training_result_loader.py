'''
Created on 2021年9月17日

@author: Administrator
'''
import os, json

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
        if record["f1score"] + record["accuracy"] > best_record["f1score"] + best_record["accuracy"]:
        # if record["accuracy"] > best_record["accuracy"]:

            best_record = record
    f1score_list = list(map(lambda x: float(x['f1score']) if "f1score" in x else float(x[" f1score"]), evaluation_records))
    return best_record, f1score_list

#从训练过程记录中，把各个模型在dev和test集中的测试结果找到，并为各次训练找到early stoping点和对应的最佳指标
def load_training_result_map(training_size_name):
    record_dir = "../../data/results/" + training_size_name + "/"
    result_map = {}
    for model_version in os.listdir(record_dir):
        model_dir = record_dir + model_version + "/"
        file_names = os.listdir(model_dir)
        if len(file_names)==0:
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
            # print("file_name", file_name)
            record_list = read_json_lines(file_name)
            # model_plan = record_list[0]
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
                    dev_task_weight, test_task_weight = record["task_weight"], record["task_weight"]
            result_map[model_version]['dev']['task_weights'].append(dev_task_weight)
            result_map[model_version]['test']['task_weights'].append(test_task_weight)
            print(len(record_list), "dev_records len", len(dev_records), model_version)
            for records in [train_records, dev_records, test_records]:
                if len(records)>0:
                    stage = records[0]['stage']
                    evaluation_records = list(filter(lambda x: "accuracy" in x, records))
                    best_record, f1score_list = find_best(evaluation_records)
                    result_map[model_version][stage]["f1score_list"].append(f1score_list)
                    result_map[model_version][stage]["best_records"].append(best_record)
                    
    return result_map

