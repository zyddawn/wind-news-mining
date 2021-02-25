import pandas as pd 
from tqdm import tqdm 


# 把整个新闻语料库中过于高频常见的词，或者过于低频的词加入停用词 


def load_word_count(df_path): 
    df = pd.read_csv(df_path) 
    print("get word count...")
    word_count = {} 
    for word, cnt in tqdm(df.values, desc="word count"):
        word_count[word] = cnt 
    return word_count 


def get_word_freq(word_count): 
    print("get word freq...")
    tot = 0 
    for v in word_count.values():
        tot += v 

    word_freq = {} 
    for word, cnt in tqdm(word_count.items(), desc="word freq"):
        word_freq[word] = cnt / tot 
    return word_freq 


def filter_min_max(word_count, min_cnt=6, max_freq=0.5): 
    word_freq = get_word_freq(word_count) 
    min_words = [] 
    for word, cnt in word_count.items(): 
        if cnt < min_cnt: 
            min_words.append((word, cnt)) 
    max_words = [] 
    for word, freq in word_freq.items(): 
        if freq > max_freq: 
            max_words.append(word) 

        
    # write stop word list 
    
    min_words = sorted(min_words, key=lambda x: (-x[1], x[0])) 
    with open("../stopwords/our_stopwords/content_min_cnt.txt", "w") as f: 
        for word, _ in min_words:
            f.write(word+"\n") 
    with open("../stopwords/our_stopwords/content_max_freq.txt", "w") as f: 
        for word in max_words: 
            f.write(word+"\n") 
    print("Saved.") 


if __name__ == '__main__':
    import sys 
    text_type = sys.argv[1] 

    word_count = load_word_count("../intermediate_results/fenci/{}/word_count/word_count.csv".format(text_type)) 
    filter_min_max(word_count, min_cnt=100, max_freq=0.1)   
    # 把整个新闻语料库中出现次数小于 100 的词加入停用词
    # 把整个新闻语料库中出现频率大于 0.1 的词加入停用词



