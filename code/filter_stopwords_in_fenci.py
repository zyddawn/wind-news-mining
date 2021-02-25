from split_words import * 
from get_word_count_in_whole_corpus import *
from collections import Counter 
import pandas as pd 


# 在分词结果中过滤掉停用词 
# 当停用词更新时，可用这个脚本来更新分词结果，而不用重新分词（过于耗时）


if __name__ == "__main__":
    from tqdm import tqdm 
    import pandas as pd 
    import os 
    import sys 

    folder = sys.argv[1] 

    # stopwords_set = merge_stopwords("../stopwords/our_stopwords/") 
    stopwords_set = merge_stopwords("../stopwords/our_stopwords/", ["additional_1.txt", "content_round_4.txt"])

    save_paths = "../intermediate_results/fenci/{}/".format(folder)
    df_path = save_paths

    word_count = {} 
    file_lst = sorted(os.listdir(df_path))
    for file in tqdm(file_lst, desc=folder):
        if file.split(".")[-1] != "csv":
            continue 

        file_path = os.path.join(df_path, file) 
        df = pd.read_csv(file_path) 
        
        words_list = [] 
        # print("{}: {}".format(file, len(df)))
        for i, idx, words in df.values:
            ch_lst = filter_stopwords(words, stopwords_set=stopwords_set)
            ch_words = " ".join(ch_lst)
            if not ch_words:
                continue 
            words_list.append((i, idx, ch_words)) 
            word_count = merge_word_count(word_count, Counter(ch_lst)) 

        word_file = file
        new_df = pd.DataFrame(data=words_list, columns=["INDEX", "OBJECT_ID", "WORDS"]) 
        new_df.to_csv(os.path.join(save_paths, word_file), header=True, index=False) 

    word_count_df = get_word_count_df(word_count) 
    word_count_df.to_csv(os.path.join(save_paths, "word_count", "word_count.csv"), index=False, header=True) 
    print("Saved word count.") 

