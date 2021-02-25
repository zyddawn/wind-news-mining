from split_words import * 
from collections import Counter
from tqdm import tqdm 
import pandas as pd 
import os 
import sys 
import numpy as np 


def merge_word_count(word_count, cur_count):
    for k,v in cur_count.items():
        word_count[k] = word_count.get(k,0) + v 
    return word_count 

def get_word_count_df(word_count):
    word_cnt_lst = sorted([(k,v) for k,v in word_count.items()], key=lambda x: (-x[1], x[0])) 
    df = pd.DataFrame(data=word_cnt_lst, columns=["WORD", "COUNT"]) 
    return df 


# 统计数据库中所有词的词频 

if __name__ == '__main__':
    np.random.seed(1111)

    text_type = sys.argv[1] 
    # stopwords_set = merge_stopwords("../stopwords/our_stopwords") 
    stopwords_set = set() 
    words_path = "../intermediate_results/fenci/{}/".format(text_type) 

    word_count = {} 
    for file in tqdm(os.listdir(words_path)): 
        if file.split(".")[-1] != "csv": 
            continue 

        file_path = os.path.join(words_path, file) 
        df = pd.read_csv(file_path) 

        for obj_id, words in df[["OBJECT_ID", "WORDS"]].values:          
            if not words or words != words:
                continue 
            word_list = words.strip().split(" ")
            if not word_list:
                continue 
            word_count = merge_word_count(word_count, Counter(word_list)) 

    print("Saving word count in all docs...")
    words_path = "../intermediate_results/fenci/{}/word_count/".format(text_type) 
    word_df = get_word_count_df(word_count) 
    save_path = os.path.join(words_path, "word_count.csv")
    word_df.to_csv(save_path, index=False, header=True) 

    with open(os.path.join(words_path, "word_list.txt"), "w") as f:
        for word in word_df["WORD"].values:
            f.write(word+"\n")
    print("Saved.")







