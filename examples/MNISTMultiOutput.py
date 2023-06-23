from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

factor = 1.25

groups = 10

s = 10.0

T = int(factor*25*100)

epochs = 250

patch_size = 10

(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

Y_train = np.empty((Y_train_org.shape[0], groups), dtype=np.uint32)
for group in range(groups):
	random_grouping = np.random.choice(10, size=5, replace=False)
	Y_train[:, group] = np.where(np.isin(Y_train_org, random_grouping), 1, 0)

Y_test = np.empty((Y_test_org.shape[0], groups), dtype=np.uint32)
for group in range(groups):
	random_grouping = np.random.choice(10, size=5, replace=False)
	Y_test[:, group] = np.where(np.isin(Y_test_org, random_grouping), 1, 0)

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, int(factor*2000), T,  patch_size), "w+")

tm = MultiOutputConvolutionalTsetlinMachine2D(int(factor*2000), T, s, (28, 28, 1), (patch_size, patch_size))

for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
    print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
    f.flush()
f.close()
