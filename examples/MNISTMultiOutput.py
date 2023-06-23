from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

factor = 1.25

groups = 10

clauses_1 = int(256)
clauses_2 = int(factor*2000)
s = 10.0
T_1 = int(clauses_1*0.8)
T_2 = int(factor*25*100)

epochs = 250

patch_size = 10

(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

Y_train = np.empty((Y_train_org.shape[0], groups), dtype=np.uint32)
random_grouping = []
for group in range(groups):
	random_grouping.append(np.random.choice(10, size=5, replace=False))
	Y_train[:, group] = np.where(np.isin(Y_train_org, random_grouping[-1]), 1, 0)

Y_test = np.empty((Y_test_org.shape[0], groups), dtype=np.uint32)
for group in range(groups):
	Y_test[:, group] = np.where(np.isin(Y_test_org, random_grouping[group]), 1, 0)

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, clauses_1, T_1,  patch_size), "w+")

tm = MultiOutputConvolutionalTsetlinMachine2D(clauses_1, T_1, s, (28, 28, 1), (patch_size, patch_size))

for i in range(10):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
    f.flush()
f.close()

X_train_transformed = tm.transform(X_train)
X_test_transformed = tm.transform(X_test)

tm = MultiClassTsetlinMachine(clauses_2, T_2, s)
for i in range(epochs):
	start_training = time()
	tm.fit(X_train_transformed, Y_train_org, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result_test = 100*(tm.predict(X_test_transformed) == Y_test_org).mean()
	stop_testing = time()

	result_train = 100*(tm.predict(X_train_transformed) == Y_train_org).mean()

	print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))


