from tqdm import tqdm
import pandas as pd 
import numpy as np 
from tf_idf_vec import * 
import sys 
import os 


def indus_vec(df, word2id):

    # 获得代表每个行业专属词的词频向量（利用在 industry_related_word_freq.py 中获得的每个行业的专属词及频率来计算）

    tf_vec = np.zeros(len(word2id)) 
    scores = np.array(df["IND_TF"].values) / np.sum(df["IND_TF"].values) 
    for i in range(len(scores)):
        w = df["WORD"].values[i] 
        s = scores[i] 
        tf_vec[word2id[w]] = s 
    return tf_vec 


def indus_vec_dict(path, word2id):

    # 行业词频向量的字典，即通过行业名称来获取该行业专属词的词频向量 

    ind_vec_dict = {} 
    for file in tqdm(os.listdir(path), desc="indus vec dict"):
        if file.split(".")[-1] != "csv":
            continue 
        ind = file.split(".")[0] 
        df = pd.read_csv(os.path.join(path, file)) 
        ind_vec_dict[ind] = indus_vec(df, word2id) # 对于该行业 ind，计算其行业专属词词频向量 
    print(len(ind_vec_dict))
    return ind_vec_dict


# 计算 月度/周度 行业热度
# 月度 or 周度的 tf-idf * 行业专属词的词频向量

if __name__ == '__main__':
    text_type = sys.argv[1]     # 新闻标题 title，或者新闻内容 content 
    time_period = sys.argv[2]   # 月度 month，或者周度 week 

    words_path = "../intermediate_results/fenci/{}/".format(text_type)
    id2word, word2id, idf_vec = load_idf(os.path.join(words_path, "word_idf", "word_idf.csv"))    

    save_path = os.path.join(words_path, "industry_to_tf")

    ind1_vec_dict = indus_vec_dict(os.path.join(save_path, "industry1"), word2id) 
    ind2_vec_dict = indus_vec_dict(os.path.join(save_path, "industry2"), word2id)

    # 导入事先计算好的月度/周度 tf-idf 
    tf_idf_path = os.path.join(words_path, "tf_idf", time_period, "vec") 

    for file in tqdm(os.listdir(tf_idf_path), desc="by {}".format(time_period)):
        tf_idf = np.load(os.path.join(tf_idf_path, file)) 

        # tf-idf * 一级行业词频向量，获得行业热度得分 
        indus1_scores = [(k,tf_idf.dot(v)) for k,v in ind1_vec_dict.items()]   
        indus1_sort_scores = sorted(indus1_scores, key=lambda x:(-x[1], x[0])) 

        # tf-idf * 二级行业词频向量，获得行业热度得分 
        indus2_scores = [(k,tf_idf.dot(v)) for k,v in ind2_vec_dict.items()]   
        indus2_sort_scores = sorted(indus2_scores, key=lambda x:(-x[1], x[0]))


        pd.DataFrame(data=indus1_sort_scores, columns=["INDUSTRY", "HEAT"]).to_csv(
                os.path.join(save_path, "industry1", "indus_heat", time_period, file.split(".")[0]+".csv"), index=False, header=True) 
        pd.DataFrame(data=indus2_sort_scores, columns=["INDUSTRY", "HEAT"]).to_csv(
                os.path.join(save_path, "industry2", "indus_heat", time_period, file.split(".")[0]+".csv"), index=False, header=True) 
        

