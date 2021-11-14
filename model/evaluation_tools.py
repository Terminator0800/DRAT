'''
Created on 2021年9月26日

@author: Administrator
'''
import torch
import numpy as np
def evaluate_SIM(epoch, dev_data_batches, model, trainng_records, trainer, mode="dev", task_weight=None):
    
    def get_confusion_matrix(confusion_matrix, predict_logits, real_labels):
        predict_outputs = list(map(lambda x: 1 if x[1]>x[0] else 0, predict_logits))
        for i in range(len(predict_outputs)):
            prediction, real_label = predict_outputs[i], real_labels[i]
            confusion_matrix[prediction][real_label] += 1
    
    def get_evaluation_metrics(confusion_matrix, part_name, stage, task_weight=None):
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
        task_weight = task_weight.cpu().numpy().tolist() if type(task_weight)!=list else task_weight
        trainng_progress = {"mode": mode, "epoch": epoch, "top_or_base_model": part_name, "loss": np.mean(losses), "accuracy":accuracy, \
               "recall": recall,  'precision': precision, "f1score": f1_score, "stage": stage, \
                            "task_weight": "none" if task_weight==None else task_weight ,\
                            "confusion_matrix": confusion_matrix}

        trainng_records.append(trainng_progress)
        print({k:v for k, v in trainng_progress.items() if k in ["epoch", "accuracy", "recall", "precision", "f1score"]})
        return f1_score, accuracy
    
    losses = []
    with torch.no_grad():
        model.eval()
        confusion_matrix_main = [[0, 0], [0, 0]]
        confusion_matrix_aux = [[0, 0], [0, 0]]
        for a_batch_ori in dev_data_batches:
            a_batch = trainer.process_a_batch(a_batch_ori)
            if len(a_batch_ori["outputs"][0])==0:#如果这个batch没有主任务数据，跳过
                continue
            outputs, main_outputs, task_weight, share_rate = model.forward(a_batch[0])

            # print("a_batch_ori[][0]", a_batch_ori["outputs"][0])
            # print("outputs", outputs)
            if trainer.supervised:#监督模式下，才有顶层任务中的混淆矩阵
                get_confusion_matrix(confusion_matrix_main, main_outputs, a_batch_ori["outputs"][0][0])
            get_confusion_matrix(confusion_matrix_aux, outputs[0][0], a_batch_ori["outputs"][0][0])
#                 predict_logits = outputs[self.task_name_id_map["text_similarity_LCQMC"]][0]
        print(mode, "辅助模型的结果:")
        f1_score, accuracy = get_evaluation_metrics(confusion_matrix_aux, "基层模型", mode, task_weight=task_weight)
    return accuracy#f1_score + accuracy


#计算span预测型阅读理解任务的EM正确率
def evaluate_MRC(epoch, dev_data_batches, model, trainng_records, trainer, mode="dev"):
    right_count = 0
    total_num = 0
    with torch.no_grad():
        model.eval()
        main_task_sample_count = 0
        for a_batch_ori in dev_data_batches:
            a_batch = trainer.process_a_batch(a_batch_ori)
            outputs, main_outputs, task_weight, share_rate = model.forward(a_batch[0])
            if 0 not in outputs: continue

            prediction = outputs[0][0]
            main_task_sample_count += prediction[0].size()[0]
            start_logits, end_logits = prediction[0], prediction[1]
            pred_start_index, pred_end_index = np.argmax(start_logits.cpu().detach().numpy(), axis=1), np.argmax(end_logits.cpu().detach().numpy(), axis=1)
            real_span = a_batch_ori["outputs"][0][0].cpu().detach().numpy()
            real_start_index, real_end_index = real_span[:, 0], real_span[:, 1]
            for i in range(len(pred_start_index)):
                if pred_start_index[i]==real_start_index[i] and pred_end_index[i]==real_end_index[i]:
                    right_count += 1
                total_num += 1
        # print("smaple number is ", main_task_sample_count)

    EM_accuracy = right_count / total_num

    trainng_progress = {"mode": mode, "epoch": epoch, "top_or_base_model": "", "loss": 0,
                        "accuracy": EM_accuracy, \
                        "recall": 0, 'precision': 0, "f1score": 0, "stage": "dev", \
                        "task_weight": "none" if task_weight == None else list(
                            map(lambda x: x, task_weight.cpu().numpy().tolist())), \
                        "confusion_matrix": []}
    trainng_records.append(trainng_progress)

    return EM_accuracy


#计算span预测型阅读理解任务的EM正确率
def evaluate_NER(epoch, dev_data_batches, model, trainng_records, trainer, mode="dev"):

    def compare_real_and_predition(real_tag_ids, prediction_tag_ids):
        right_found = 0
        found = 0
        real = 0
        for i in range(len(real_tag_ids)):
            if real_tag_ids[i]!=0:
                real += 1
                if real_tag_ids[i]==prediction_tag_ids[i]:
                    right_found += 1

            if prediction_tag_ids[i]!=0:
                found += 1
        return right_found, found, real

    right_count = 0
    total_num = 0
    total_recall_num = 0
    with torch.no_grad():
        model.eval()
        total_right_found, total_found, total_real = 0, 0, 0
        for a_batch_ori in dev_data_batches:
            a_batch = trainer.process_a_batch(a_batch_ori)
            outputs, main_outputs, task_weight, share_rate = model.forward(a_batch[0])
            if 0 not in outputs: continue
            prediction = outputs[0][0]
            pred_entity_types_list = np.argmax(prediction.cpu().detach().numpy(), axis=2)
            real_entity_types_list = a_batch_ori["outputs"][0][0].cpu().detach().numpy()
           # print("real_entity_type", pred_entity_types_list)

            for i in range(0, len(pred_entity_types_list)):
                right_found, found, real = compare_real_and_predition(real_entity_types_list[i], pred_entity_types_list[i])
                total_right_found += right_found
                total_found += found
                total_real += real

    precision = total_right_found / total_found if total_found>0 else 0
    recall = total_right_found/total_real
    f1score = 2*precision*recall/(recall + precision) if recall + precision>0 else 0
    trainng_progress = {"mode": mode, "epoch": epoch, "top_or_base_model": "", "loss": 0,
                        "accuracy": f1score, \
                        "recall": recall, 'precision': precision, "f1score": f1score, "stage": "dev", \
                        "task_weight": "none" if task_weight == None else list(
                            map(lambda x: x, task_weight.cpu().numpy().tolist())), \
                        "confusion_matrix": []}
    trainng_records.append(trainng_progress)

    return f1score


#计算span预测型阅读理解任务的EM正确率
def evaluate_CLS(epoch, dev_data_batches, model, trainng_records, trainer, mode="dev"):
    right_count = 0
    total_num = 0
    total_recall_num = 0
    main_task_sample_count = 0
    with torch.no_grad():
        model.eval()
        for a_batch_ori in dev_data_batches:
            a_batch = trainer.process_a_batch(a_batch_ori)
            outputs, main_outputs, task_weight, share_rate = model.forward(a_batch[0])
            if 0 not in outputs: continue
            prediction = outputs[0][0]
            main_task_sample_count += prediction.size()[0]
            pred_entity_types_list = np.argmax(prediction.cpu().detach().numpy(), axis=1)
            real_entity_types_list = a_batch_ori["outputs"][0][0].cpu().detach().numpy()
            for i in range(0, len(pred_entity_types_list)):
                if pred_entity_types_list[i]==real_entity_types_list[i]:
                    right_count += 1
                total_num += 1
    # print("main_task_sample_count", main_task_sample_count, total_num)
    accuracy = right_count / total_num if total_num>0 else 1.0
    trainng_progress = {"mode": mode, "epoch": epoch, "top_or_base_model": "", "loss": 0,
                        "accuracy": accuracy, \
                        "recall": 0, 'precision': 0, "f1score": 0, "stage": "dev", \
                        "task_weight": "none" if task_weight == None else list(
                            map(lambda x: x, task_weight.cpu().numpy().tolist())), \
                        "confusion_matrix": [], "total_num": total_num}
    trainng_records.append(trainng_progress)
    return accuracy