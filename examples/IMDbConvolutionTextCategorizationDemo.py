import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

maxlen = 100

epochs = 100

clauses = 10000
T = 8000
s = 10.0

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, maxlen=maxlen, index_from=INDEX_FROM)

train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

print("Producing bit representation...")

print(train_y.shape[0])
X_train = lil_matrix((train_y.shape[0], maxlen*(NUM_WORDS+INDEX_FROM)), dtype=np.uint32)
for e in range(train_y.shape[0]):
	position = 0
	for word_id in train_x[e]:
		X_train[e, position*(NUM_WORDS+INDEX_FROM) + word_id] = 1
		position += 1
X_train = X_train.tocsr()
Y_train = train_y.astype(np.uint32)

print(test_y.shape[0])
X_test = lil_matrix((test_y.shape[0], maxlen*(NUM_WORDS+INDEX_FROM)), dtype=np.uint32)
for e in range(test_y.shape[0]):
	position = 0
	for word_id in test_x[e]:
		X_test[e, position*(NUM_WORDS+INDEX_FROM) + word_id] = 1
		position += 1
X_test = X_test.tocsr()
Y_test = test_y.astype(np.uint32)

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (1, maxlen, (NUM_WORDS+INDEX_FROM)), (1, 1))
for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
