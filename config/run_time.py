'''
Created on 2020年7月25日

@author: Administrator
'''
GPU_NUM = None
TAG_ID_FILE = "../../data/model/tag_id_maps.pkl"
CHAR_ID_FILE = "../../data/model/char_id_map.pkl"

PATH_HANZI_EMBEDDING_VECTORS = "../../data/pretrained_models/other/pretrained_char_vec.txt"

#基础模型参数
PATH_DIR_MODEL = "../../data/models/base/"
PATH_MODEL_TOKEN_ID = "../../data/models/base/char_id_map.pkl"
PATH_MODEL_PRETRAINED_TOKEN_EMBEDDINGS = "../../data/models/base/pretrained_char_vec.pkl"
PATH_MODEL_TOKEN_EMBEDDINGS = "../../data/models/base/embeddings.pkl"
MAX_TEXT_LEN = 90
MAX_DOC_LEN = 70
MAX_QUESTION_LEN = 20
PATH_MODEL_CKPTS = "../../data/models/ckpts/"
PATH_MODEL_PB_FORMAT = "../../data/models/PB/"
AUXILIARY_TASK_NUMBER = 2


ALL_AUXILIARY_TASK_LIST = ["text_similarity_LCQMC" , "text_classification_THUCNews", "text_smililarity_atec", "text_classification_toutiao","text_similarity_CCKS2018",
                       "text_classification_tansongbo_hotel_comment", "NER_rmrb", "NER_CLUENER_public", "MRC_DRCD", "MRC_cmrc2018", \
                           "text_classification_chnsenticorp", "text_similarity_ChineseSTS", "text_classification_simplifyweibo_4_moods",
                           "NER_MSAR", "NER_rmrb_2014", "NER_boson", "NER_weibo", "MRC_chinese_SQuAD", "MRC_CAIL2019", "NER_CMNER",
                           "NER_CCKS2019_task1_Yidu_S4K", "NER_baidu2020_event", "NLI_cmnli_public", "NLI_ocnli_public", "sentiment_analysis_dmsc_v2",
                             "sentiment_analysis_online_shopping_10_cats", "sentiment_analysis_waimai_10k", "sentiment_analysis_weibo_senti_100k",
                           "sentiment_analysis_yf_dianping"]#任务列表
AUXILIARY_TASK_OUTPUT_NUM_MAP = {"event_extraction_baidu2020": 2, \
                        "text_classification_toutiao": 1, "text_smililarity_atec": 1, \
                    "text_similarity_LCQMC": 1, 'text_similarity_CCKS2018': 1, \
                    "text_classification_tansongbo_hotel_comment": 1, "NER_rmrb": 1, \
                    "NER_CLUENER_public": 1, "text_classification_THUCNews": 1, "MRC_DRCD": 2, "MRC_cmrc2018": 2, "text_classification_chnsenticorp": 1,
                                 "text_similarity_ChineseSTS": 1, "text_classification_simplifyweibo_4_moods": 1,
                                 "NER_MSAR": 1, "NER_rmrb_2014": 1, "NER_boson": 1, "NER_weibo": 1, "MRC_chinese_SQuAD": 1,
                                 "MRC_CAIL2019": 1,                          "NER_CMNER": 1,
                           "NER_CCKS2019_task1_Yidu_S4K":  1,
                           "NER_baidu2020_event":  1,
                           "NLI_cmnli_public": 1,
                           "NLI_ocnli_public": 1,
                           "sentiment_analysis_dmsc_v2": 1,
                           "sentiment_analysis_online_shopping_10_cats": 1,
                           "sentiment_analysis_waimai_10k": 1,
                           "sentiment_analysis_weibo_senti_100k": 1,
                           "sentiment_analysis_yf_dianping": 1,
                                    }
AUXILIARY_TASK_TYPE_MAP = {"event_extraction_baidu2020": "CLS", \
                        "text_classification_toutiao": "CLS", "text_smililarity_atec": "SIM", \
                    "text_similarity_LCQMC": "SIM", 'text_similarity_CCKS2018': "SIM", \
                    "text_classification_tansongbo_hotel_comment": "SEN", "NER_rmrb": "NER", \
                    "NER_CLUENER_public": "NER", "text_classification_THUCNews": "CLS", "MRC_DRCD": "MRC", "MRC_cmrc2018": "MRC",
                           "text_classification_chnsenticorp": "SEN", "text_similarity_ChineseSTS": "SIM",
                           "text_classification_simplifyweibo_4_moods": "CLS",
                           "NER_MSAR": "NER", "NER_rmrb_2014": "NER", "NER_boson": "NER", "NER_weibo": "NER", "MRC_chinese_SQuAD": "MRC",
                           "MRC_CAIL2019": "MRC",
                           "NER_CMNER": "NER",
                           "NER_CCKS2019_task1_Yidu_S4K":  "NER",
                           "NER_baidu2020_event":  "NER",
                           "NLI_cmnli_public": "MAT",
                           "NLI_ocnli_public": "MAT",
                           "sentiment_analysis_dmsc_v2": "SEN",
                           "sentiment_analysis_online_shopping_10_cats": "SEN",
                           "sentiment_analysis_waimai_10k": "SEN",
                           "sentiment_analysis_weibo_senti_100k": "SEN",
                           "sentiment_analysis_yf_dianping": "SEN",
                           "text_similarity_afqmc": "SIM",
                           }
#############NER模型参数##########
PATH_NER_TRAIN_CORPS = "../../data/corpus/NER/train_data"
PATH_NER_TEST_CORPS = "../../data/corpus/NER/test_data"

LSTM_UNIT_NUM = 50

##################事件抽取模型参数###################
PATH_MOEL_EVENT_EXTRACTION_NER_TAG_ID = "../../data/models/event_extraction_ner_tag_id_maps.pkl"
EVENT_MODEL_NER_TAG_NUM = 745
IF_FILTER_EVENT_TYPES_IN_TRAINING = False
PATH_EVENT_EXTRACTION_TAG_ID_MAP = "../../data/models/event_extraction_tag_id_maps.pkl"
PATH_EVENT_TYPE_STATUS = "../../data/corpus/event_extraction/event_extraction_baidu2020/事件类型与角色对应表.xls"
#语料
PATH_EVENT_EXTRACTION_TRAIN_CORPUS = "../../data/corpus/event_extraction/event_extraction_baidu2020/train.json"
PATH_EVENT_EXTRACTION_TEST_CORPUS = "../../data/corpus/event_extraction/event_extraction_baidu2020/test.json"
PATH_EVENT_EXTRACTION_DEV_CORPUS = "../../data/corpus/event_extraction/event_extraction_baidu2020/dev.json"
PATH_EVENT_EXTRACTION_SAMPLE_CORPUS = "../../data/corpus/event_extraction/event_extraction_baidu2020/sample_data.json"


#文本分类模型参数
TEXT_CLASS_NUMBER = 15
PATH_TEXT_CLASS_TAG_IDS = "../../data/models/text_classification_tag_id_maps.pkl"
PATH_TEXT_CLASSIFICATION_TRAIN_DATA = "../../data/corpus/text_classification/toutiao_new_classification_random.txt"

##################文本相似度模型参数###########
#蚂蚁金服语料
PATH_TEXT_SIMILARITY_TRAIN_DATA = "../../data/corpus/text_similarity/text_similarity_atec_nlp_2018/atec_nlp_sim_train.txt"
#LCQMC语料
PATH_TEXT_SIMILARITY_LCQMC_TRAIN_DATA =  "../../data/corpus/text_similarity/text_similarity_LCQMC/train.txt"
PATH_TEXT_SIMILARITY_LCQMC_DEV_DATA = "../../data/corpus/text_similarity/text_similarity_LCQMC/dev.txt"
PATH_TEXT_SIMILARITY_LCQMC_TEST_DATA = "../../data/corpus/text_similarity/text_similarity_LCQMC/test.txt"
SIMILARITY_CLASS_NUM = 2
#text_similarity_CCKS2018
PATH_TEXT_SIMILARITY_CCKS2018_TRAIN_DATA = "../../data/corpus/text_similarity/text_similarity_CCKS2018/task3_train.txt"

#谭松波酒店评论情感数据集
PATH_TEXT_CLASSIFICATION_TANSONGBO_HOTEL_COMMENT_SENTIMENT_TRAIN_DATA = \
                           "../../data/corpus/text_classification/tansongbo_hotel_comment_sentiment/tansongbo_hotel_comment_sentiment_10000.txt"

#################机器阅读理解模型参数################
#DRCD数据
#DuReader数据

#预训练语言模型参数
PATH_BERT_VOCAB = "../../data/pretrained_models/chinese_L-12_H-768_A-12/vocab.txt"
PATH_BERT_CONFIG = "../../data/pretrained_models/chinese_L-12_H-768_A-12/bert_config.json"

PATH_BERT_BASE_FILE = "../../data/pretrained_models/chinese_bert_for_pytorch/pytorch_model.bin"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/model-ernie-gram-zh.1_torch"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/RoBERTa_zh_L12_PyTorch"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/ZEN_pretrain_base_v0.1.0"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/chinese_bert_for_pytorch"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/chinese_macbert_base"
PATH_BERT_BASE_DIR = "../../data/pretrained_models/model-ernie1.0.1"



