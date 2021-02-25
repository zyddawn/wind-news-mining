from collections import Counter
from tqdm import tqdm 


def remove_dup(dict_): 

    # 统计每个词的词频在每个行业中的排名情况，只保留最高排名的行业，认为这个词是该行业的专属词之一
    # 尽可能消除行业之间的共同信息，保留当前行业特有的信息 

    word_rank = {} 
    dlist = {} 
    # get word rank
    for ind, ind_dict in tqdm(dict_.items(), desc="word rank"): 
        dlist[ind] = sorted([(k,v) for k,v in ind_dict.items()], key=lambda x: (-x[1], x[0])) 
        for i, (word, _) in enumerate(dlist[ind]):
            if word not in word_rank:
                word_rank[word] = i 
            else:
                word_rank[word] = min(i, word_rank[word]) 

    new_dlist = {} 
    for ind, ind_lst in tqdm(dlist.items(), desc="filter dup"):
        new_dlist[ind] = [] 
        for i, (word, v) in enumerate(dlist[ind]):
            if i > word_rank[word]:
                continue 
            new_dlist[ind].append((word, v)) 
    return new_dlist 


# 获得每个行业的高频词 

if __name__ == "__main__":
    from tqdm import tqdm 
    import pandas as pd 
    import os 
    import sys 

    folder = sys.argv[1] 

    ind1_df = pd.read_csv("../field_ref/indus1_belong.csv") 
    ind1_df["date"] = ind1_df["date"].apply(lambda x: str(x))  
    ind1_df.set_index("date", inplace=True) 
    ind2_df = pd.read_csv("../field_ref/indus2_belong.csv") 
    ind2_df["date"] = ind2_df["date"].apply(lambda x: str(x))
    ind2_df.set_index("date", inplace=True) 

    df_path = "../intermediate_results/fenci/{}/".format(folder) 

    industry1_to_tf = {}    # 一级行业 
    industry2_to_tf = {}    # 二级行业 

    print("\n")
    file_lst = sorted(os.listdir(df_path))
    for file in tqdm(file_lst, desc=folder):
        if file.split(".")[-1] != "csv":
            continue 

        date = file[20:28] # 比如 “20190101“
        ind1_dict = {} 
        ind2_dict = {} 
        ind1 = ind1_df.loc[date] 
        for c, i1 in zip(ind1_df.columns, ind1): 
            if i1 and i1==i1:
                ind1_dict[c] = i1       # c 为公司代码，i1 为对应的一级行业 
        ind2 = ind2_df.loc[date]
        for c, i2 in zip(ind2_df.columns, ind2):
            if i2 and i2==i2:
                ind2_dict[c] = i2       # c 为公司代码，i2 为对应的二级行业 

        if not ind1_dict and not ind2_dict:
            continue 

        file_path = os.path.join(df_path, file) 
        df = pd.read_csv(file_path) 
        
        # print("{}: {}".format(file, len(df)))
        for i, idx, words, wcode, _ in df.values: 
            if wcode and wcode == wcode: 
                wcode_lst = [w.split(":")[0] for w in wcode.split("|")] 
            else: 
                continue 
            word_cnt = Counter(words.split(" ")) 
            tot = sum([v for v in word_cnt.values()]) 
            tf = {k:v*1.0/tot for k,v in word_cnt.items()} 

            for wc in wcode_lst:        # wind code 
                if wc in ind1_dict:
                    i1 = ind1_dict[wc]  # get industry
                    if i1 not in industry1_to_tf:
                        industry1_to_tf[i1] = {} 
                    # industry related word freq 
                    for w,f in tf.items():
                        # 一级行业 i1 相关的新闻中词语 w 出现的词频增加 f 
                        industry1_to_tf[i1][w] = industry1_to_tf[i1].get(w, 0.0) + f        
                if wc in ind2_dict:
                    i2 = ind2_dict[wc] 
                    if i2 not in industry2_to_tf:
                        industry2_to_tf[i2] = {} 
                    for w,f in tf.items():
                        # 二级行业 i2 相关的新闻中词语 w 出现的词频增加 f 
                        industry2_to_tf[i2][w] = industry2_to_tf[i2].get(w, 0.0) + f 


    print("Saving...")
    folder1_path = os.path.join(df_path, "industry_to_tf", "industry1")
    if not os.path.isdir(folder1_path):
        os.makedirs(folder1_path) 

    print(industry1_to_tf)
    industry1_to_tf_uniq = remove_dup(industry1_to_tf)
    for i1, wdict in tqdm(industry1_to_tf_uniq.items(), desc="saving ind1"):
        pd.DataFrame(data=wdict, columns=["WORD", "IND_TF"]).to_csv(
			os.path.join(folder1_path, "{}.csv".format(i1)), index=False, header=True) 

    folder2_path = os.path.join(df_path, "industry_to_tf", "industry2")
    if not os.path.isdir(folder2_path):
        os.makedirs(folder2_path)


    # 去掉不同行业之间的共同词，保留当前行业特有的词 
    industry2_to_tf_uniq = remove_dup(industry2_to_tf)
    for i2, wdict in tqdm(industry2_to_tf_uniq.items(), desc="saving ind2"):
        pd.DataFrame(data=wdict, columns=["WORD", "IND_TF"]).to_csv(
			os.path.join(folder2_path, "{}.csv".format(i2)), index=False, header=True) 

    print("Saved.")
    

