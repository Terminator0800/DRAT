from common.models import text_classification_model, NER_model, MRC_span

def get_model_base():
    model_map = {"text_similarity_LCQMC": \
                     text_classification_model.TextClassification("text_similarity_LCQMC", highway=False,
                                                                  class_num=2),

                 "text_classification_toutiao": \
                     text_classification_model.TextClassification("text_classification_toutiao", highway=False,
                                                                  class_num=15),

                 "text_smililarity_atec": \
                     text_classification_model.TextClassification("text_smililarity_atec", highway=False,
                                                                  class_num=2),
                 "text_similarity_afqmc": \
                     text_classification_model.TextClassification("text_smililarity_atec", highway=False,
                                                                  class_num=2),
                 "text_similarity_CCKS2018": \
                     text_classification_model.TextClassification("text_similarity_CCKS2018", highway=False,
                                                                  class_num=2),

                 "text_classification_tansongbo_hotel_comment": \
                     text_classification_model.TextClassification("text_classification_tansongbo_hotel_comment",
                                                                  highway=False,
                                                                  class_num=2),
                 "NER_rmrb": NER_model.NERModel("NER_rmrb", class_num=13),

                 "NER_CLUENER_public": NER_model.NERModel("NER_CLUENER_public", class_num=31),

                 "text_classification_THUCNews": \
                     text_classification_model.TextClassification("text_classification_THUCNews", highway=False,
                                                                  class_num=14),
                 "MRC_DRCD": MRC_span.MRCSpan("MRC_DRCD"),
                 "MRC_cmrc2018": MRC_span.MRCSpan("MRC_cmrc2018"),
                 "text_classification_chnsenticorp": text_classification_model.TextClassification(
                     "text_classification_chnsenticorp", highway=False,
                     class_num=2),
                 "text_similarity_ChineseSTS": text_classification_model.TextClassification(
                     "text_similarity_ChineseSTS", highway=False,
                     class_num=2),
                 "text_classification_simplifyweibo_4_moods": \
                     text_classification_model.TextClassification("text_classification_simplifyweibo_4_moods",
                                                                  highway=False,
                                                                  class_num=4),
                 "NER_MSAR": NER_model.NERModel("NER_MSAR", class_num=7),
                 "NER_rmrb_2014": NER_model.NERModel("NER_rmrb_2014", class_num=9),
                 "NER_boson": NER_model.NERModel("NER_boson", class_num=15),
                 "NER_weibo": NER_model.NERModel("NER_weibo", class_num=15),
                 "MRC_chinese_SQuAD": MRC_span.MRCSpan("MRC_chinese_SQuAD"),
                 "MRC_CAIL2019": MRC_span.MRCSpan("MRC_CAIL2019"),
                 "NER_CMNER": NER_model.NERModel("NER_CMNER", class_num=45),
                 "NER_CCKS2019_task1_Yidu_S4K": NER_model.NERModel("NER_CCKS2019_task1_Yidu_S4K", class_num=25),
                 "NER_baidu2020_event": NER_model.NERModel("NER_baidu2020_event", class_num=65*4+1),
                 "NLI_cmnli_public": text_classification_model.TextClassification("NLI_cmnli_public",
                                                                  highway=False, class_num=3),
                 "NLI_ocnli_public": text_classification_model.TextClassification("NLI_ocnli_public",
                                                                                  highway=False, class_num=3),
                 "sentiment_analysis_dmsc_v2": text_classification_model.TextClassification("sentiment_analysis_dmsc_v2",
                                                                  highway=False, class_num=5),
                 "sentiment_analysis_online_shopping_10_cats": text_classification_model.TextClassification("sentiment_analysis_online_shopping_10_cats",
                                                                                  highway=False, class_num=2),

            "sentiment_analysis_waimai_10k": text_classification_model.TextClassification( "sentiment_analysis_waimai_10k",
                             highway=False, class_num=2),
                 "sentiment_analysis_weibo_senti_100k": text_classification_model.TextClassification("sentiment_analysis_weibo_senti_100k",
                                                                                  highway=False, class_num=2),

                "sentiment_analysis_yf_dianping": text_classification_model.TextClassification(
                        "sentiment_analysis_yf_dianping",
                        highway=False, class_num=5),

                 }
    return model_map

def get_supervisor_model(task_name, model_map):
    if task_name in ["text_similarity_LCQMC", "text_smililarity_atec", "text_similarity_afqmc"]:
        supervisor = text_classification_model.TextClassification(task_name, supervisor=True,
                                                                   class_num=2)
        model_map[task_name] = text_classification_model.TextClassification(task_name, highway=True,
                                                                  class_num=2)
    if task_name in ["MRC_DRCD"]:
        supervisor = MRC_span.MRCSpan(task_name, supervisor=True)
        model_map[task_name] = MRC_span.MRCSpan(task_name, highway=True)

    if task_name in ["NER_MSAR"]:
        supervisor = NER_model.NERModel(task_name, class_num=7, supervisor=True)
        model_map[task_name] = NER_model.NERModel(task_name, class_num=7, highway=True)

    if task_name in ["text_classification_chnsenticorp"]:
        supervisor = text_classification_model.TextClassification(
            task_name, highway=False, class_num=2, supervisor=True)
        model_map[task_name] = text_classification_model.TextClassification(task_name, highway=True,
                                                                  class_num=2)
    return supervisor, model_map