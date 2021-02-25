from split_words import * 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import scipy.sparse as sp

from datetime import datetime 
from collections import Counter 
import os 
import sys 


def load_idf(idf_path): 
    # 导入提前计算好的 idf 数据 
    idf_df = pd.read_csv(idf_path) 
    idf_vec = np.array(idf_df["IDF"].values)
    id2word = idf_df["WORD"].values
    word2id = {} 
    for i, word in enumerate(id2word):
        word2id[word] = i 
    return id2word, word2id, idf_vec


def norm(vec):
    sum_ = np.sum(vec)
    if sum_ > 0:
        return vec / sum_
    return vec 


def get_tf_vec(word_list, word2id):
    # 对于一条新闻，计算词频向量 
    tf_vec = np.zeros(len(word2id)) 
    for word in word_list: 
        try:
            tf_vec[word2id[word]] += 1.0 
        except:
            pass 
    return norm(tf_vec) 


def cal_tf_idf_vec(tf_vec, idf_vec):
    # 计算 tf-idf 
    tf_idf_vec = tf_vec * idf_vec 
    return tf_idf_vec 


def increment_tf_vec_by_time_period(tp_tf_idf_vec, new_tf_idf_vec): 
    # 累积月度/周度词频向量 （这样可以尽量减少计算 tf-idf 中乘法的次数，提高效率）
    # 每条新闻的权重一样 
    return tp_tf_idf_vec + new_tf_idf_vec 


def get_topN_word(tf_idf_vec, id2word, N=5): 
    desc_topN_id = np.argsort(tf_idf_vec)[::-1][:N] 
    return ["{} {}".format(id2word[id_].upper(), tf_idf_vec[id_]) for id_ in desc_topN_id]


def str2week(s): 
    dt = datetime.strptime(s, "%Y%m%d") 
    return int(dt.timestamp() - datetime(2014,12,28,0,0,0).timestamp()) // (86400*7)


def week_name(week):
    start = datetime.fromtimestamp(week*86400*7 + datetime(2014,12,28,0,0,0).timestamp())
    return "%04d%02d%02d" % (start.year, start.month, start.day) 



if __name__ == '__main__':
    text_type = sys.argv[1] 
    time_period = sys.argv[2] 
    certain_start = certain_end = None 
    if len(sys.argv) >= 5:
        certain_start = sys.argv[3] 
        certain_end = sys.argv[4] 
        print("calculating for time {} - {}".format(certain_start, certain_end)) 
    assert time_period in ["week", "month"]
    # assert time_period == "month"

    topN = 200

    # stopwords_set = merge_stopwords("../stopwords/our_stopwords") 
    words_path = "../intermediate_results/fenci/{}/".format(text_type) 

    # load idf 
    id2word, word2id, idf_vec = load_idf(os.path.join(words_path, "word_idf", "word_idf.csv")) 

    if time_period == "month": 
        print("#########################   Calculating tf-idf by month   #########################")
        month_tf_vec = np.zeros(len(idf_vec)) 
        cur_month = None 

        file_list = sorted(os.listdir(words_path))[::-1] 
        for file in tqdm(file_list): 
            if file.split(".")[-1] != "csv": 
                continue 

            file_path = os.path.join(words_path, file) 
            df = pd.read_csv(file_path) 

            if not cur_month:
                cur_month = file[20:26] 

            elif cur_month != file[20:26]: 
                # calculate tf-idf and save current 
                month_tf_idf_vec = cal_tf_idf_vec(month_tf_vec, idf_vec)
                month_topN = get_topN_word(month_tf_idf_vec, id2word, N=topN) 
                save_file = "{}.txt".format(cur_month)
                with open(os.path.join(words_path, "tf_idf", "month", save_file), "w") as f:
                    for word in month_topN: 
                        f.write(word+"\n") 
                np.save(os.path.join(words_path, "tf_idf", "month", "vec", "{}.npy".format(cur_month)), month_tf_idf_vec) 

                # renew 
                cur_month = file[20:26] 
                month_tf_vec = np.zeros(len(idf_vec)) 

            for obj_id, words in df[["OBJECT_ID", "WORDS"]].values:
                if not words or words != words:
                    continue 
                # word_list = filter_news(words, stopwords_set=stopwords_set)
                word_list = words.strip().split(" ")
                if not word_list:
                    continue 

                doc_tf_vec = get_tf_vec(word_list, word2id) 
                month_tf_vec = increment_tf_vec_by_time_period(month_tf_vec, doc_tf_vec) 

        month_tf_idf_vec = cal_tf_idf_vec(month_tf_vec, idf_vec)
        month_topN = get_topN_word(month_tf_idf_vec, id2word, N=topN) 
        save_file = "{}.txt".format(cur_month)
        with open(os.path.join(words_path, "tf_idf", "month", save_file), "w") as f:
            for word in month_topN: 
                f.write(word+"\n") 
        np.save(os.path.join(words_path, "tf_idf", "month", "vec", "{}.npy".format(cur_month)), month_tf_idf_vec) 

    elif time_period == "week":
        print("#########################   Calculating tf-idf by week   #########################")
        week_tf_vec = np.zeros(len(idf_vec)) 
        cur_week = None 
        
        file_list = sorted(os.listdir(words_path)) 
        if certain_start and certain_end: 
            file_list = [file for file in file_list if file[20:28]>=certain_start and file[20:28]<certain_end]
        print("{} - {}".format(file_list[0], file_list[-1]))
        for file in tqdm(file_list): 
            if file.split(".")[-1] != "csv": 
                continue 

            file_path = os.path.join(words_path, file) 
            df = pd.read_csv(file_path) 

            if not cur_week: 
                cur_week = str2week(file[20:28]) 

            elif cur_week != str2week(file[20:28]): 
                # calculate tf-idf and save current 
                week_tf_idf_vec = cal_tf_idf_vec(week_tf_vec, idf_vec) 
                week_topN = get_topN_word(week_tf_idf_vec, id2word, N=topN) 
                save_file = "{}.txt".format(week_name(cur_week)) 
                with open(os.path.join(words_path, "tf_idf", "week", save_file), "w") as f: 
                    for word in week_topN: 
                        f.write(word+"\n") 
                np.save(os.path.join(words_path, "tf_idf", "week", "vec", "{}.npy".format(week_name(cur_week))), week_tf_idf_vec) 

                # renew 
                cur_week = str2week(file[20:28]) 
                week_tf_vec = np.zeros(len(idf_vec)) 

            for obj_id, words in df[["OBJECT_ID", "WORDS"]].values:
                if not words or words != words:
                    continue 
                # word_list = filter_news(words, stopwords_set=stopwords_set)
                word_list = words.strip().split(" ")
                if not word_list:
                    continue 

                doc_tf_vec = get_tf_vec(word_list, word2id) 
                week_tf_vec = increment_tf_vec_by_time_period(week_tf_vec, doc_tf_vec) 

        week_tf_idf_vec = cal_tf_idf_vec(week_tf_vec, idf_vec)
        week_topN = get_topN_word(week_tf_idf_vec, id2word, N=topN) 
        save_file = "{}.txt".format(week_name(cur_week))
        with open(os.path.join(words_path, "tf_idf", "week", save_file), "w") as f:
            for word in week_topN: 
                f.write(word+"\n") 
        np.save(os.path.join(words_path, "tf_idf", "week", "vec", "{}.npy".format(week_name(cur_week))), week_tf_idf_vec) 



















