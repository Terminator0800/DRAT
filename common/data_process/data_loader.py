'''
Created on 2021年2月10日

@author: Administrator
'''
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from common.data_process import base_processor, data_process_NER, data_process_event_extraction, \
                        data_process_text_classification, data_process_text_similarity, data_process_MRC_span
from  config import run_time

def upsampling(samples):
    postive_samples, negative_samples = [], []
    for sample in samples:
        if sample[-1]==1: postive_samples.append(sample)
        else: negative_samples.append(sample)
    import random
    random.seed(666)
    random.shuffle(negative_samples)
    left_num_each_class = max(len(postive_samples), len(negative_samples))
    supp_sample_num = int((left_num_each_class - len(negative_samples))/2)#上采样规模控制一下
    new_negative_samples = negative_samples + negative_samples[: supp_sample_num]
    print("new_lnegative_samples", len(new_negative_samples), "new_postive_samples", len(postive_samples))
    lines = new_negative_samples + postive_samples
    random.shuffle(lines)
    return lines

def add_data_source_label(data_list, label):
    new_data_list = []
    for data in data_list: 
        data = data[:4] + [label] + data[4:]
        new_data_list.append(data)
    return new_data_list

#人民日报NER
def load_data_for_NER_rmrb(task_name, data_source_label, train_set_size, devset_size):
    print("加载人民日报NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/train_data", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

#CLUE的NER
def load_data_for_NER_CLUENER_public(task_name, data_source_label, train_set_size, devset_size):
    print("加载CLUE的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/cluener_public_in_CRFpp_style.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list


def load_data_for_event_extraction(task_name, if_demo, data_source_label, devset_size):#加载事件抽取的数据
    dp_event_extraction = data_process_event_extraction.DataProcessor4EventExraction()
    event_extraction_train_data_list, text_list = dp_event_extraction.get_train_data(task_name, if_demo=if_demo)
    event_extraction_train_data_list = add_data_source_label(event_extraction_train_data_list, data_source_label)

    event_extraction_dev_data_list, _ = dp_event_extraction.get_dev_data(task_name, if_demo=if_demo)
    event_extraction_dev_data_list = add_data_source_label(event_extraction_dev_data_list, data_source_label)
    return task_name, event_extraction_train_data_list, event_extraction_dev_data_list

def load_data_for_text_classification(task_name, data_source_label, train_set_size, devset_size):#加载文本分类数据
    print("任务", task_name)
    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    text_classification_train_data_list, text_list = dp_text_classification.get_train_data(task_name, train_set_size=train_set_size)
    text_classification_train_data_list = add_data_source_label(text_classification_train_data_list, data_source_label)
    print("头条数据量是", len(text_classification_train_data_list))
    return task_name, text_classification_train_data_list, None, None, text_list

def load_data_for_text_classification_THUCNews(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    text_classification_train_data_list, text_list = dp_text_classification.get_train_data(task_name,
              file_name="../../data/corpus/text_classification/THUCNews.txt", train_set_size=train_set_size, standard_format=True)
    text_classification_train_data_list = add_data_source_label(text_classification_train_data_list, data_source_label)
    print("THUCNews数据量是", len(text_classification_train_data_list))
    return task_name, text_classification_train_data_list, None, None, text_list

def load_data_for_text_similarity(task_name, data_source_label, train_set_size, devset_size):#加载文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/text_similarity_atec_nlp_2018/train_part_SF.txt", train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    text_similarity_dev_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/text_similarity_atec_nlp_2018/dev_part_SF.txt", train_set_size=devset_size)
    text_similarity_dev_data_list = add_data_source_label(text_similarity_dev_data_list, data_source_label)
    print("蚂蚁金服数据量是", len(text_similarity_train_data_list))
    return task_name, text_similarity_train_data_list, text_similarity_dev_data_list, None, text_list

def load_data_for_text_similarity_afqmc(task_name, data_source_label, train_set_size, devset_size):#加载文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/afqmc_public/train_SF.txt", train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    text_similarity_dev_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/afqmc_public/dev_SF.txt", train_set_size=devset_size)
    text_similarity_dev_data_list = add_data_source_label(text_similarity_dev_data_list, data_source_label)
    print("afqmc数据量是", len(text_similarity_train_data_list))
    return task_name, text_similarity_train_data_list, text_similarity_dev_data_list, None, text_list

def load_data_for_text_similarity_LCQMC(task_name, data_source_label, train_set_size, devset_size):#加载LCQMC文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/text_similarity_LCQMC/train.txt", \
                                                                  train_set_size=train_set_size)

    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    #text_similarity_train_data_list = upsampling(text_similarity_train_data_list)

    text_similarity_dev_data_list, _ = dp_text_similarity.get_data(task_name, run_time.PATH_TEXT_SIMILARITY_LCQMC_DEV_DATA, train_set_size=devset_size)#, if_demo=if_demo)
    text_similarity_dev_data_list = add_data_source_label(text_similarity_dev_data_list, data_source_label)

    text_similarity_test_data_list, _ = dp_text_similarity.get_data(task_name, run_time.PATH_TEXT_SIMILARITY_LCQMC_TEST_DATA, train_set_size=devset_size)#, if_demo=if_demo)
    text_similarity_test_data_list = add_data_source_label(text_similarity_test_data_list, data_source_label)
    print("LCQMC", len(text_similarity_train_data_list))
    return task_name, text_similarity_train_data_list, text_similarity_dev_data_list, text_similarity_test_data_list, text_list

def load_data_for_text_similarity_CCKS2018(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, run_time.PATH_TEXT_SIMILARITY_CCKS2018_TRAIN_DATA, train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    print("CCKS2018", len(text_similarity_train_data_list))

    return task_name, text_similarity_train_data_list, None, None, text_list

def load_data_for_text_classification_tansongbo_hotel_comment_sentiment(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    text_classification_train_data_list, text_list = \
             dp_text_classification.get_train_data(task_name, file_name="../../data/corpus/sentiment_analysis/tansongbo_hotel_comment_sentiment/tansongbo_hotel_comment_sentiment_10000.txt", \
                                                   train_set_size=train_set_size,\
                                                   standard_format=True)
    text_classification_train_data_list = add_data_source_label(text_classification_train_data_list, data_source_label)
    print("谭松波酒店评论数据", len(text_classification_train_data_list))
    return task_name, text_classification_train_data_list, None, None, text_list

def load_data_for_DRCD_or(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp = data_process_MRC_span.DataPreprocess4Corpus()

    train_data_list, text_list = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/DRCD/DRCD_training.json", \
                                                   train_set_size=train_set_size)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("DRCD数据", len(train_data_list))
    dev_data_list, _ = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/DRCD/DRCD_dev.json", \
                                                   train_set_size=devset_size)
    print("DRCD dev数据量", len(dev_data_list))
    dev_data_list = add_data_source_label(dev_data_list, data_source_label)
    return task_name, train_data_list, dev_data_list, None, text_list

def load_data_for_DRCD(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp = data_process_MRC_span.DataPreprocess4Corpus()

    train_data_list, text_list = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/ibudda_train.txt", \
                                                   train_set_size=train_set_size)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("DRCD数据", len(train_data_list))
    dev_data_list, _ = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/ibudda_dev.txt", \
                                                   train_set_size=devset_size)
    print("DRCD dev数据量", len(dev_data_list))
    dev_data_list = add_data_source_label(dev_data_list, data_source_label)
    return task_name, train_data_list, dev_data_list, None, text_list

def load_data_for_cmrc2018(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)
    dp = data_process_MRC_span.DataPreprocess4Corpus()
    train_data_list, text_list = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/cmrc2018-master/squad-style-data/cmrc2018_train.json", \
                                                   train_set_size=train_set_size)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("cmrc2018数据", len(train_data_list))

    return task_name, train_data_list, None, None, text_list

def load_data_for_MRC_chinese_SQuAD(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    dp = data_process_MRC_span.DataPreprocess4Corpus()
    train_data_list, text_list = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/chinese_SQuAD/dev-zen-v1.0 (1).json", \
                                                   train_set_size=train_set_size)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("MRC_chinese_SQuAD数据", len(train_data_list))

    return task_name, train_data_list, None, None, text_list

def load_data_for_MRC_CAIL2019(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    dp = data_process_MRC_span.DataPreprocess4Corpus()
    train_data_list, text_list = \
             dp.get_data(task_name, file_name=r"../../data/corpus/MRC/CAIL2019/dev_ground_truth.json", \
                                                   train_set_size=train_set_size)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("cmrc2018数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list

def load_data_for_chnsenticorp(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name=r"../../data/corpus/sentiment_analysis/chnsenticorp/chnsenticorp_train.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)

    dev_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name=r"../../data/corpus/sentiment_analysis/chnsenticorp/chnsenticorp_dev.txt", \
                                                   train_set_size=devset_size, standard_format=True)
    dev_data_list = add_data_source_label(dev_data_list, data_source_label)
    print("chnsenticorp数据", len(train_data_list))

    return task_name, train_data_list, dev_data_list, None, text_list

def load_data_for_text_similarity_ChineseSTS(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = \
          dp_text_similarity.get_data(task_name, "../../data/corpus/text_similarity/text_similarity_ChineseSTS//simtrain_to05sts_normal_format.txt",\
                                                                                                             train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    print("ChineseSTS数据", len(text_similarity_train_data_list))

    return task_name, text_similarity_train_data_list, None, None, text_list

def load_data_for_implifyweibo_4_moods(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name=r"../../data/corpus/text_classification/simplifyweibo_4_mood_normal_format.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("text_classification_simplifyweibo_4_moods数据", len(train_data_list))

    return task_name, train_data_list, None, None, text_list

def load_data_for_NER_MSAR(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    print("加载MSAR的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/MSAR_NER/train.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)

    NER_dev_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/MSAR_NER/val.txt",
                                                           train_set_size=devset_size)
    NER_dev_data_list = add_data_source_label(NER_dev_data_list, data_source_label)

    return task_name, NER_train_data_list, NER_dev_data_list, None, text_list

def load_data_for_NER_rmrb_2014(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    print("加载rmrb_2014的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/pd2014/train.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NER_boson(task_name, data_source_label, train_set_size, devset_size):
    print("任务", task_name)

    print("加载boson的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/boson/boson_in_crfpp_format.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NER_weibo(task_name, data_source_label, train_set_size, devset_size):
    print("加载weibo的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/weibo/train.txt", train_set_size=train_set_size)
    # _, _ = dp_NER.get_train_data(task_name, "../../data/corpus/NER/weibo/weiboNER.conll.dev", train_set_size=train_set_size)

    # _, _ = dp_NER.get_train_data(task_name, "../../data/corpus/NER/weibo/weiboNER.conll.test", train_set_size=train_set_size)

    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NER_CMNER(task_name, data_source_label, train_set_size, devset_size):
    print("加载CMNER的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/CMNER/example_train.crfpp", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NER_CCKS2019_task1_Yidu_S4K(task_name, data_source_label, train_set_size, devset_size):
    print("加载CCKS2019_task1_Yidu_S4K的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/NER/CCKS2019_task1_Yidu_S4K/subtask1_training_SF.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NER_baidu2020_event(task_name, data_source_label, train_set_size, devset_size):
    print("加载NER_baidu2020_event的NER数据", task_name, data_source_label)
    dp_NER = data_process_NER.DataProcessorNER()
    NER_train_data_list, text_list = dp_NER.get_train_data(task_name, "../../data/corpus/event_extraction/event_extraction_baidu2020/train_triger_tag_crfpp.txt", train_set_size=train_set_size)
    NER_train_data_list = add_data_source_label(NER_train_data_list, data_source_label)
    return task_name, NER_train_data_list, None, None, text_list

def load_data_for_NLI_cmnli_public(task_name, data_source_label, train_set_size, devset_size):#加载文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/NLI/cmnli_public/train_SF_mini.txt", train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    print("NLI_cmnli_public数据量是", len(text_similarity_train_data_list))
    return task_name, text_similarity_train_data_list, None, None, text_list

def load_data_for_NLI_ocnli_public(task_name, data_source_label, train_set_size, devset_size):#加载文本相似度数据
    print("任务", task_name)

    dp_text_similarity = data_process_text_similarity.DataProcessorTextSimilarity()
    text_similarity_train_data_list, text_list = dp_text_similarity.get_data(task_name, "../../data/corpus/NLI/ocnli_public/train.50k_SF.txt", train_set_size=train_set_size)
    text_similarity_train_data_list = add_data_source_label(text_similarity_train_data_list, data_source_label)
    print("NLI_ocnli_public数据量是", len(text_similarity_train_data_list))
    return task_name, text_similarity_train_data_list, None, None, text_list

def load_data_for_sentiment_analysis_dmsc_v2(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name="../../data/corpus/sentiment_analysis/dmsc_v2/ratings_SF.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("dmsc_v2数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list

def load_data_for_sentiment_analysis_online_shopping_10_cats(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name="../../data/corpus/sentiment_analysis/online_shopping_10_cats/online_shopping_10_cats_SF.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("online_shopping_10_cats数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list

def load_data_for_sentiment_analysis_waimai_10k(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name="../../data/corpus/sentiment_analysis/waimai_10k/waimai_10k_SF.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("waimai_10k数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list

def load_data_for_sentiment_analysis_weibo_senti_100k(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name="../../data/corpus/sentiment_analysis/weibo_senti_100k/weibo_senti_100k_SF.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("weibo_senti_100k数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list

def load_data_for_sentiment_analysis_yf_dianping(task_name, data_source_label, train_set_size, devset_size):#加载CCKS2018文本相似度数据
    print("任务", task_name)

    dp_text_classification = data_process_text_classification.DataProcessorTextClassification()
    train_data_list, text_list = \
             dp_text_classification.get_data(task_name, file_name="../../data/corpus/sentiment_analysis/yf_dianping/ratings_mini_SF.txt", \
                                                   train_set_size=train_set_size, standard_format=True)
    import random
    random.shuffle(train_data_list)
    train_data_list = add_data_source_label(train_data_list, data_source_label)
    print("yf_dianping数据", len(train_data_list))
    return task_name, train_data_list, None, None, text_list


def load_data_for_all_tasks(train_set_size, devset_size=100):
    print("正在加载训练数据")
    import multiprocessing

    data_for_all_tasks_map = {"train": {run_time.ALL_AUXILIARY_TASK_LIST[0]: {"samples": [], "batches": None}}, \
                              "dev": {"samples": [], "batches": None}, \
                              "test": {"samples": [], "batches": None}, \
                              "train_step": 0, "text_list": []}

    loaders = {
        "text_classification_toutiao": load_data_for_text_classification,
        "text_classification_tansongbo_hotel_comment": load_data_for_text_classification_tansongbo_hotel_comment_sentiment,
        "event_extraction_baidu2020": load_data_for_event_extraction,
        "text_smililarity_atec": load_data_for_text_similarity,
        "text_similarity_afqmc": load_data_for_text_similarity_afqmc,
        "text_similarity_LCQMC": load_data_for_text_similarity_LCQMC,
        "text_similarity_CCKS2018": load_data_for_text_similarity_CCKS2018,
        "NER_rmrb": load_data_for_NER_rmrb,
        "NER_CLUENER_public": load_data_for_NER_CLUENER_public,
        "text_classification_THUCNews": load_data_for_text_classification_THUCNews,
        "MRC_DRCD": load_data_for_DRCD,
        "MRC_cmrc2018": load_data_for_cmrc2018,
        "text_classification_chnsenticorp": load_data_for_chnsenticorp,
        "text_similarity_ChineseSTS": load_data_for_text_similarity_ChineseSTS,
        "text_classification_simplifyweibo_4_moods": load_data_for_implifyweibo_4_moods,
        "NER_MSAR": load_data_for_NER_MSAR,
        "NER_rmrb_2014": load_data_for_NER_rmrb_2014,
        "NER_boson": load_data_for_NER_boson,
        "NER_weibo": load_data_for_NER_weibo,
        "MRC_chinese_SQuAD": load_data_for_MRC_chinese_SQuAD,
        "MRC_CAIL2019": load_data_for_MRC_CAIL2019,
        "NER_CMNER": load_data_for_NER_CMNER,
        "NER_CCKS2019_task1_Yidu_S4K": load_data_for_NER_CCKS2019_task1_Yidu_S4K,
        "NER_baidu2020_event": load_data_for_NER_baidu2020_event,
        "NLI_cmnli_public":load_data_for_NLI_cmnli_public,
        "NLI_ocnli_public":load_data_for_NLI_ocnli_public,
        "sentiment_analysis_dmsc_v2":load_data_for_sentiment_analysis_dmsc_v2,
         "sentiment_analysis_online_shopping_10_cats" :    load_data_for_sentiment_analysis_online_shopping_10_cats,
          "sentiment_analysis_waimai_10k" :   load_data_for_sentiment_analysis_waimai_10k,
          "sentiment_analysis_weibo_senti_100k"  :  load_data_for_sentiment_analysis_weibo_senti_100k,
          "sentiment_analysis_yf_dianping"  :  load_data_for_sentiment_analysis_yf_dianping,
    }
    results = []
    pool = multiprocessing.Pool(processes=8)  # 创建4个进程
    for task_name in run_time.ALL_AUXILIARY_TASK_LIST:
        # print("数据源id", task_name, run_time.ALL_AUXILIARY_TASK_LIST.index(task_name), run_time.ALL_AUXILIARY_TASK_LIST)
        # print(task_info.keys())
        results.append(pool.apply_async(func=loaders[task_name], \
                                        args=(task_name, \
                                              run_time.ALL_AUXILIARY_TASK_LIST.index(task_name), \
                                              train_set_size, devset_size)))
    pool.close()
    pool.join()
    main_task_class_count = {}
    for r in results:
        r = r.get()
        task_name, train_data, dev_data, test_data, text_list = r
        data_for_all_tasks_map["train"][task_name] = {"samples": train_data, "batches": None}
        if dev_data != None:
            data_for_all_tasks_map["dev"]["samples"].extend(dev_data)
        if test_data != None:
            data_for_all_tasks_map["test"]["samples"].extend(test_data)
        print("加载得到数据的例子", task_name, train_data[0][0])
        # 统计各种样本的数量分布，计算类别权重
        if task_name == run_time.ALL_AUXILIARY_TASK_LIST[0]:
            for sample in train_data:
                if type(sample[-1])==int:#分类任务才类加权
                    main_task_class_count[sample[-1]] = main_task_class_count.get(sample[-1], 0) + 1
    main_task_sample_num = 0
    for label in main_task_class_count: main_task_sample_num += main_task_class_count[label]
    class_weight = [1 / main_task_class_count[label] for label in range(len(main_task_class_count))]
    class_sum = sum(class_weight)
    class_weight = [x / class_sum for x in class_weight]
    print("class_weight", class_weight)
    return data_for_all_tasks_map, class_weight

def all_data_into_batches(data_for_all_tasks_map, batch_size, mini_batch_size=2, this_rander=None):
    dp_base = base_processor.DataProcessor()        #训练数据的分组需要特殊处理：（1）将各个类别的数据分为GPU_num大小的batch；(2)将所有的小batch收集起来shuffle，然后构造大batch
    mini_batches = []
    for task_name in data_for_all_tasks_map["train"]:
        if task_name not in run_time.ALL_AUXILIARY_TASK_LIST: continue
        #print(task_name, "(data_for_all_tasks_map[train]", (data_for_all_tasks_map["train"]["text_similarity_LCQMC"]["samples"][:10]))
        mini_batches += dp_base.data_into_batches(data_for_all_tasks_map["train"][task_name]["samples"],\
                                rander=this_rander, \
                                batch_size=mini_batch_size)#训练数据需要经常shuffle
    batches, batch = [], []
    this_rander.shuffle(mini_batches)
    for mini_batch in mini_batches:
        for sample in mini_batch:
            batch.append(sample)
            if len(batch)==batch_size:
                batches.append(batch)
                batch = []
    if len(batch)>0: batches.append(batch)
    data_for_all_tasks_map["train"]["batches"] = batches
    if data_for_all_tasks_map["dev"]["samples"]!=None:#如果有开发集
        data_for_all_tasks_map["dev"]["batches"] = \
                dp_base.data_into_batches(data_for_all_tasks_map["dev"]["samples"], batch_size=100)
    if data_for_all_tasks_map["test"]["samples"]!=None:#如果有开发集
        data_for_all_tasks_map["test"]["batches"] = \
                dp_base.data_into_batches(data_for_all_tasks_map["test"]["samples"], batch_size=100)

def distribute_task_samples(feature_data, ori_task_ids):
    task_data_list = [[] for _ in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
    new_feature_data = []
    for sample_index in range(len(ori_task_ids)):
        task_data_list[ori_task_ids[sample_index]].append(feature_data[sample_index])

    task_batch_size_list = [None for _ in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
    for i in range(len(task_batch_size_list)):
        task_batch_size_list[i] = int(len(task_data_list[i]) / run_time.GPU_NUM)

    for gpu_id in range(run_time.GPU_NUM):
        for task_index in range(len(task_batch_size_list)):
            this_batch_size = task_batch_size_list[task_index]
            new_feature_data.extend(
                task_data_list[task_index][gpu_id * this_batch_size: (gpu_id + 1) * this_batch_size])
    return new_feature_data

def samples_into_features(samples):
    data_list = [[] for i in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
    for task_id in range(len(samples)):
        sample_in_this_task = samples[task_id]
        # print(task_id, sample_in_this_task)
        if len(sample_in_this_task)>0:
            feature_num = len(sample_in_this_task[0])
            data_list[task_id] = [[] for _ in range(feature_num)]
            for sample in sample_in_this_task:
                for i in range(feature_num):
                    data_list[task_id][i].append(sample[i])
    return data_list

# 把各个子任务的数据batch合起来，形成训练数据
def combine_all_task_samples(main_task, data_for_all_tasks_map, \
                             job_stage="train", batch_size=30):
    data_batches = []

    # 事件抽取任务数据处理
    a_batch = {}
    a_batch["inputs"] = {"token_ids": [], "seq_lengths": [], "segment_ids": [],
                         "input_mask": [], "task_ids": []}
    a_batch["outputs"] = [[] for _ in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
    for i in range(len(data_for_all_tasks_map[job_stage]["batches"])):
        task_batch = data_for_all_tasks_map[job_stage]["batches"][i]
        a_batch["inputs"]['token_ids'].extend(list(map(lambda x: x[0], task_batch)))
        a_batch["inputs"]['seq_lengths'].extend(list(map(lambda x: x[1], task_batch)))
        a_batch["inputs"]['segment_ids'].extend(list(map(lambda x: x[2], task_batch)))
        a_batch["inputs"]['input_mask'].extend(list(map(lambda x: x[3], task_batch)))
        a_batch["inputs"]['task_ids'].extend(list(map(lambda x: x[4], task_batch)))

        # 按照任务对output
        task_ids = list(map(lambda x: x[4], task_batch))
        for i in range(len(task_ids)):
            task_id = task_ids[i]
            # print("task_id", task_id)
            a_batch["outputs"][task_id].append(task_batch[i][5:])
        # outputs = list(map(lambda x: x[5:], task_batch))
        # a_batch["outputs"].extend(outputs)

        a_batch["outputs"] = samples_into_features(a_batch["outputs"])
        # print("a_batch[outputs]", a_batch["outputs"])
        for key in a_batch["inputs"]:
            a_batch["inputs"][key] = distribute_task_samples(a_batch["inputs"][key], task_ids)
        data_batches.append(a_batch)
        a_batch = {}
        a_batch["inputs"] = {"token_ids": [], "seq_lengths": [], "segment_ids": [],
                             "input_mask": [], "task_ids": []}
        a_batch["outputs"] = [[] for _ in range(len(run_time.ALL_AUXILIARY_TASK_LIST))]
    return data_batches



if __name__ == '__main__':
    a, b, c, d = load_data_for_text_classification_tansongbo_hotel_comment_sentiment("asd", True, 0)
#     for line in b:
#         print(line)
