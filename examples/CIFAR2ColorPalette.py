import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0

resolution = 8

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

examples = 5000

max_included_literals = 32
clauses = 2000
T = 5000
s = 1.5

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org[0:examples]
X_test_org = X_test_org[0:examples]

Y_train = Y_train.reshape(Y_train.shape[0])[0:examples]
Y_test = Y_test.reshape(Y_test.shape[0])[0:examples]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

X_train = np.zeros((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], resolution**3), dtype=np.uint32)
for i in range(X_train.shape[0]):
        for x in range(X_train_org.shape[1]):
                for y in range(X_train_org.shape[2]):
                        index = (X_train_org[i, x, y, 0] // (256//resolution))*(resolution**2) + (X_train_org[i, x, y, 1] // (256//resolution))*resolution + (X_train_org[i, x, y, 2] // (256//resolution))
                        X_train[i, x, y, index] = 1 

X_test = np.zeros((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], resolution**3), dtype=np.uint32)
for i in range(X_test.shape[0]):
        for x in range(X_test_org.shape[1]):
                for y in range(X_test_org.shape[2]):
                        index = (X_test_org[i, x, y, 0] // (256//resolution))*(resolution**2) + (X_test_org[i, x, y, 1] // (256//resolution))*resolution + (X_test_org[i, x, y, 2] // (256//resolution))
                        X_test[i, x, y, index] = 1 

print(X_test.shape, X_test.shape)

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

print(X_test.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d.txt" % (s, clauses, T, scaling), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, resolution**3), (3, 3), max_included_literals=max_included_literals)

        for epoch in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train, epochs=1, incremental=True)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()

                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f" % (ensemble, epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
