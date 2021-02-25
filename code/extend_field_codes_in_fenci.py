

# 把新闻的行业归属代号和新闻内容进行匹配 

if __name__ == "__main__":
    from tqdm import tqdm 
    import pandas as pd 
    import os 
    import sys 

    folder = sys.argv[1] 

    save_paths = "../intermediate_results/fenci/{}/".format(folder)
    df_path = save_paths

    file_lst = sorted(os.listdir(df_path))
    for file in tqdm(file_lst, desc=folder):
        if file.split(".")[-1] != "csv":
            continue 

        file_path = os.path.join(df_path, file) 
        df = pd.read_csv(file_path) 
        
        code_dict = {} 
        code_df = pd.read_csv("../field_ref/CODES/codes_{}".format(file.split("_", 1)[-1])) 
        for oid, wcode, icode in code_df.values:
            code_dict[oid] = (wcode, icode) 

        new_df_data = [] 
        # print("{}: {}".format(file, len(df)))
        for i, idx, words in df.values:
            (wcode, icode) = code_dict[idx] 
            new_df_data.append((i, idx, words, wcode, icode)) 

        extended_df_file = file
        new_df = pd.DataFrame(data=new_df_data, columns=["INDEX", "OBJECT_ID", "WORDS", "WINDCODES", "INDUSTRYCODES"]) 
        new_df.to_csv(os.path.join(save_paths, extended_df_file), header=True, index=False) 

