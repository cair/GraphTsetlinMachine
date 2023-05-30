# -*- coding: utf-8 -*-

import os
import glob
import random
random.seed(42)
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from tmu.tsetlin_machine import TMClassifier
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from PySparseCoalescedTsetlinMachineCUDA.tm import AutoEncoderTsetlinMachine
from sklearn.metrics.pairwise import cosine_similarity
import codecs

target_word_weight=defaultdict(list)
target_similarity=defaultdict(list)

import re
import nltk
from nltk.corpus import brown
nltk.download('brown')
nltk.download('punkt')

with open('word1_ws.pkl', 'rb') as f:
    word1 = pickle.load(f)
with open('word2_ws.pkl', 'rb') as f:
    word2 = pickle.load(f)

def evaluate(profile_dict_word1):
    language="english"
    pair_list = []
    fread_simlex=codecs.open("wordsim353-" + language + ".txt", 'r', 'utf-8')
    pair_list = []

    line_number = 0
    for line in fread_simlex:
        if line_number > 0:
            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])
            pair_list.append( ((word_i, word_j), score) )
        line_number += 1
    print(len(pair_list))

    calculated_score=[]
    extracted_list = []
    original_score=[]
    word_pairs=[]
    for (x,y) in pair_list:
        if x in profile_dict_word1:
            word1, word2=x
            word1_prof=profile_dict_word1[x]
            extracted_list.append((x, word1_prof))
            calculated_score.append(word1_prof)
            original_score.append(y)
            word_pairs.append(x)

    from scipy.stats import spearmanr
    from sklearn import preprocessing
    spearman_TM = spearmanr(original_score, calculated_score)
    spearman_TM = round(spearman_TM[0], 3)
    print(f'spearman TM {spearman_TM}')

    total_list=[]
    total_list.append(original_score)
    total_list.append(calculated_score)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(total_list)
    print(f'Cosine TM {similarity}')

    TM_corr= np.corrcoef(original_score, calculated_score)
    print(f'pearson TM {TM_corr}')

    from scipy.stats import kendalltau
    kendal_TM, _ = kendalltau(original_score, calculated_score)
    print(f'kendal TM {kendal_TM}')

    import pandas as pd
    data = pd.DataFrame([original_score,calculated_score])
    data=data.transpose()
    data.columns=['original','TM']
    correlation = data.corr()
    print("pearson corr", correlation)

num = 0
word_total= list(set(word1[num:num+800] + word2[num:num+800]))

clause_weight_threshold = 0

number_of_examples = 2000*10
accumulation = 25 

type_i_ii_ratio = 1.0

clause_drop_p = 0.0

factor = 20
clauses = int(factor*20/(1.0 - clause_drop_p))
T = factor*40
s = 5.0
q = 10.0

print("Loading Vectorizer")
f_vectorizer_X = open("./vectorizer_X.pickle", "rb")
vectorizer_X = pickle.load(f_vectorizer_X)
f_vectorizer_X.close()

print("Loading Data")
f_X = open("./X.pickle", "rb")
X_csr = pickle.load(f_X)
f_X.close()

X_train = X_csr

feature_names = vectorizer_X.get_feature_names_out()

number_of_features = vectorizer_X.get_feature_names_out().shape[0]

target_words=[]
for i in word_total:
    if i in vectorizer_X.vocabulary_:
        target_words.append(i)

print("target word length", len(target_words))
output= open('target_ws_' + str(num) + '.pkl', "wb")
pickle.dump(target_words, output)
output.close()


print(len(target_words))
print("feature name", feature_names)

output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
	target_word = target_words[i]
	target_id = vectorizer_X.vocabulary_[target_word]
	output_active[i] = target_id
    
tm = AutoEncoderTsetlinMachine(clauses, T, s, output_active, q=q, max_included_literals=3, accumulation=accumulation, append_negated=False)

print("\nAccuracy Over 40 Epochs:")
for e in range(1000):
    start_training = time()
    tm.fit(X_train, epochs=1, number_of_examples=number_of_examples, incremental=True)
    stop_training = time()
    total_time= stop_training-start_training
    print("\nEpoch #%d\n" % (e+1))
    print(f'epoch per time: {total_time}')

    weights = tm.get_state()[1].reshape((len(target_words), -1))

    profile = np.empty((len(target_words), clauses))
    for i in range(len(target_words)):
        profile[i,:] = np.where(weights[i,:] >= clause_weight_threshold, weights[i,:], 0)

    similarity = cosine_similarity(profile)
    for i in range(len(target_words)):
        sorted_index = np.argsort(-1*similarity[i,:])
        for j in range(1, len(target_words)):
            target_similarity[(target_words[i], target_words[sorted_index[j]])]  = similarity[i,sorted_index[j]]

    evaluate(target_similarity)
