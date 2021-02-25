from pyhanlp import *
import string 
import jieba
import re 

CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')

def is_nan(text):
    # 判断文本是否为空
    return text != text or text == "nan" 


def remove_special_marks(text):
    # 去掉链接
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+\.com', '', text)

    # 去掉百分比的数字
    text = re.sub(r'(\d+(\.\d+)?)%', ' ', text)
    # 去掉价格数字
    text = re.sub(r'[$￥€£](\d+(\.\d+)?)', ' ', text) 
    # 去掉数字
#     text = re.sub(r'\b\d+(\.\d+)?\b', ' ', text) 
    text = re.sub(r'\d+(\.\d+)?', ' ', text) 
    # 去掉 html & javascript 代码片段
    text = re.sub(r'&lt;.+&gt;', " ", text) 
    
    # 清理其他噪音 
    text = re.sub(r"本文参考以下来源.+", " ", text) 
    text = re.sub(r'<br>', ' ', text) 
    text = re.sub(r'\u3000', ' ', text) 
    text = re.sub(r'&amp;', ' ', text)
    
    # 去掉英文里的标点符号
    text = "".join([ch if ch not in set(string.punctuation) else " " for ch in text ])
    # 去掉中文里的标点符号 
    text = re.sub(r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：∶·；《）《》“”()»〔〕-]+', " ", text)
    
    return text 


def normalization(content): 
    # 使用 hanlp 库函数对文本进行清理 
    # CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    if is_nan(content):
        return ""
    return CharTable.convert(content)  


def clean_text(text):
    # 清理文本 
    text = normalization(text)
    text = remove_special_marks(text) 
    return text


def merge_stopwords(stopwords_path, stopwords_list=None): 
    # 合并不同来源的停用词 
    stopwords_set = set() 
    if not stopwords_list:
        stopwords_list = [f for f in os.listdir(stopwords_path) if f[-3:]=="txt"]
    print("stopwords list: {}".format(stopwords_list))
        
    for stopwords in stopwords_list:
        with open(os.path.join(stopwords_path, stopwords), "r") as f:
            for line in f.readlines():
                stopwords_set.add(line.strip()) 
    return stopwords_set


def split_words(text, stop_words=None):
    res = [] 
    for word in list(jieba.cut(text)):
        if len(word) == 1:   # including " "
            continue 
        elif stop_words and word in stop_words:
            continue 
        res.append(word)
    return res 


def clean_and_split(text, stop_words=None):
    return split_words(clean_text(text), stop_words=stop_words) 


def has_chinese(text):
    # 文本中是否包含中文（用于去除纯英文新闻）
    for ch in text[:200]:       # 为了效率，只看前 200 个字符 
        if ch >= u'\u4e00' and ch <= u'\u9fa5':
            return True 
    return False 


def filter_news(words, stopwords_set=set()):
    if not words or words == " " or words != words:
        return []
    if stopwords_set:
        word_list = filter_stopwords(words, stopwords_set=stopwords_set)
    else:
        word_list = [word for word in words.split(" ") if not is_nan(word)] 
    if not word_list:
        return [] 
    # skip english news 
    if not has_chinese(words): 
        return [] 
    return word_list


def filter_stopwords(words, stopwords_set=set()):
    if not words or words != words:
        return [] 
    return [word for word in words.split(" ") if word and not is_nan(word) and word not in stopwords_set]




if __name__ == "__main__":
    from tqdm import tqdm 
    import pandas as pd 
    import os 
    import sys 

    text_type = sys.argv[1] # 新闻标题 title，或新闻内容 content
    selected_folder = sys.argv[2]
    col_name = text_type.upper() 

    stopwords_set = merge_stopwords("../stopwords", ["baidu_stopwords.txt", "cn_stopwords.txt", "nltk_en_stopwords.txt"]) 

    csv_path = "../data/"
    save_paths = "../intermediate_results/fenci/{}/".format(text_type)

    
    if selected_folder == '*':
        folder_lst = sorted(os.listdir(csv_path))
    else:
        folder_lst = [selected_folder, ]

    # 统计已经处理过的新闻 (csv)，当程序中断时，可以随时恢复进度 
    processed_set = set() 
    for file in os.listdir(save_paths): 
        if file.split(".")[-1] == 'csv': 
            processed_set.add(file[6:]) 


    print(folder_lst) 
    for folder in folder_lst: 
        file_lst = sorted(os.listdir(os.path.join(csv_path, folder))) 
        for file in tqdm(file_lst, desc=folder): 
            if file.split(".")[-1] != "csv" or file in processed_set: 
                continue 

            file_path = os.path.join(csv_path, folder, file) 
            df = pd.read_csv(file_path) 
            
            df[col_name] = df[col_name].apply(lambda x: x[:50000] if type(x) is str else " ")   # 截断部分太长的新闻 
            df[col_name] = df[col_name].apply(lambda x: x.lower() if type(x) is str else " ") 

            words_list = [] 
            # print("{}: {}".format(file, len(df))) 
            for i, (idx, txt) in tqdm(enumerate(df[["OBJECT_ID", col_name]].values)): 
                words = " ".join(clean_and_split(txt, stop_words=stopwords_set))    # 清洗文本 
                if not words:
                    words = " "
                words_list.append((i, idx, words)) 
            
            word_file = "fenci_"+file
            new_df = pd.DataFrame(data=words_list, columns=["INDEX", "OBJECT_ID", "WORDS"]) 
            new_df.to_csv(os.path.join(save_paths, word_file), header=True, index=False) 


