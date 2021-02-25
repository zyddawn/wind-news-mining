from split_words import * 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import scipy.sparse as sp

from collections import Counter 
import os 
import sys 


# 统计并存下月度词频向量
# 这个脚本是从 tf_idf_vec.py 魔改过来的  


def load_idf(idf_path):
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
	tf_vec = np.zeros(len(word2id)) 
	for word in word_list: 
		try:
			tf_vec[word2id[word]] += 1.0 
		except:
			pass 
	return norm(tf_vec) 


def cal_tf_idf_vec(tf_vec, idf_vec):
	tf_idf_vec = tf_vec * idf_vec 
	return tf_idf_vec 


def increment_tf_vec_by_time_period(tp_tf_idf_vec, new_tf_idf_vec): 
	# Each doc was given the same weight 
	return tp_tf_idf_vec + new_tf_idf_vec 


def get_topN_word(tf_idf_vec, id2word, N=5): 
	desc_topN_id = np.argsort(tf_idf_vec)[::-1][:N] 
	return ["{} {}".format(id2word[id_].upper(), tf_idf_vec[id_]) for id_ in desc_topN_id]


def save_monthly_freq(arr, path):
    np.save(path, arr) 



if __name__ == '__main__':
	text_type = sys.argv[1] 
	time_period = "month" 
	# assert time_period in ["day", "month"]
	assert time_period == "month"

	topN = 150

	# stopwords_set = merge_stopwords("../stopwords/our_stopwords") 
	words_path = "../intermediate_results/fenci/{}/".format(text_type) 

	# load idf 
	id2word, word2id, idf_vec = load_idf(os.path.join(words_path, "word_idf", "word_idf.csv")) 

	if time_period == "month": 
		print("#########################   Calculating tf by month   #########################")
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
				# calculate tf and save current 
				save_file = "{}.npy".format(cur_month)
				save_monthly_freq(month_tf_vec, os.path.join(words_path, "word_monthly_freq", save_file))

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
		save_file = "{}.npy".format(cur_month)
		save_monthly_freq(month_tf_vec, os.path.join(words_path, "word_monthly_freq", save_file))
