from split_words import * 
from collections import Counter 
import pandas as pd 



if __name__ == "__main__":
    from tqdm import tqdm 
    import pandas as pd 
    import os 
    import sys 

    folder = sys.argv[1] 

    # stopwords_set = merge_stopwords("../stopwords/our_stopwords/") 
    stopwords_set = set() 
    if not stopwords_set:
        print("Do not filter stopwords.") 
    else:
        print("filter stopwords.") 

    save_paths = "../intermediate_results/fenci/{}/".format(folder)
    df_path = save_paths 

    file_lst = sorted(os.listdir(df_path))
    for file in tqdm(file_lst, desc=folder):
        if file.split(".")[-1] != "csv":
            continue 

        file_path = os.path.join(df_path, file) 
        df = pd.read_csv(file_path) 
        
        words_list = [] 
        # print("{}: {}".format(file, len(df)))
        for i, idx, words in df.values:
            ch_lst = filter_news(words)
            ch_words = " ".join(ch_lst)
            if not ch_words:
                continue 
            words_list.append((i, idx, ch_words)) 

        word_file = file
        new_df = pd.DataFrame(data=words_list, columns=["INDEX", "OBJECT_ID", "WORDS"]) 
        new_df.to_csv(os.path.join(save_paths, word_file), header=True, index=False) 


