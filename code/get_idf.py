from split_words import * 
from tqdm import tqdm 
import pandas as pd 
import os 
import sys 
import numpy as np 



if __name__ == '__main__':
    print("Getting IDF values...")
    text_type = sys.argv[1] 

    # stopwords_set = merge_stopwords("../stopwords/our_stopwords") 
    words_path = "../intermediate_results/fenci/{}/".format(text_type) 

    word_docs = {} 
    cnt_docs = 0 # 总新闻数 

    for file in tqdm(os.listdir(words_path)): 
        if file.split(".")[-1] != "csv": 
            continue 

        file_path = os.path.join(words_path, file) 
        df = pd.read_csv(file_path) 

        for obj_id, words in df[["OBJECT_ID", "WORDS"]].values: 
            # word_list = filter_stopwords(words, stopwords_set=stopwords_set) 
            if not words or words != words:
                continue 
            word_list = words.strip().split(" ") 
            if not word_list: 
                continue 

            cnt_docs += 1 
            uniq_word_list = list(set(word_list)) 
            for word in uniq_word_list: 
                # 一条新闻中出现的每个词的 idf + 1
                word_docs[word] = word_docs.get(word, 0) + 1 

    word_idf = {} 
    word_large_idf = {} 
    for w, c in word_docs.items(): 
        if is_nan(word) or not word:
            continue 
        word_large_idf[w] = float(cnt_docs / c)         # 尝试扩大 idf （不加 log）
        word_idf[w] = float(np.log10(cnt_docs / c))     # 原始的 idf （ log(出现该词的新闻数 / 总新闻数) ）
        
    print("Sort word idf...") 
    word_large_idf_list = sorted([(w,idf) for w,idf in word_large_idf.items()], key=lambda x: (-x[1], x[0]))
    word_idf_list = sorted([(w,idf) for w,idf in word_idf.items()], key=lambda x: (-x[1], x[0])) 
    print(len(word_idf)) 


    # 存下两个版本的 idf 
    idf_df = pd.DataFrame(data=word_idf_list, columns=["WORD", "IDF"])
    idf_df.to_csv(os.path.join(words_path, "word_idf", "word_idf.csv"), index=False, header=True) 
    large_idf_df = pd.DataFrame(data=word_large_idf_list, columns=["WORD", "IDF"])
    large_idf_df.to_csv(os.path.join(words_path, "word_idf", "word_large_idf.csv"), index=False, header=True) 


    lst = [w for w,_ in word_idf_list] 
    with open(os.path.join(words_path, "word_idf", "word_list.txt"), "w") as f:
        for w in lst: 
            f.write(w+"\n") 

