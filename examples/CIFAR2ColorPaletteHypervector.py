import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0

hypervector_size = 256

bits = 5

resolution = 8

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

examples = 5000

max_included_literals = 32

clauses = 8000*2
T = int(clauses * 0.75)

#clauses = 2000
#T = 5000

s = 10.0

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = np.zeros((resolution, bits), dtype=np.uint32)
for i in range(resolution):
        encoding[i] = np.random.choice(indexes, size=(bits))

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org#[0:examples]
X_test_org = X_test_org#[0:examples]
Y_train = Y_train.reshape(Y_train.shape[0])#[0:examples]
Y_test = Y_test.reshape(Y_test.shape[0])#[0:examples]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

X_train = lil_matrix((X_train_org.shape[0], X_train_org.shape[1] * X_train_org.shape[2] * hypervector_size), dtype=np.uint32)
for i in range(X_train.shape[0]):
        for x in range(X_train_org.shape[1]):
                for y in range(X_train_org.shape[2]):
                        roll = 0
                        for c in range(X_train_org.shape[3]):
                                if c == 0:
                                        roll += (X_train_org[i, x, y, c] // 32)
                                elif c == 1:
                                        roll += 11 * (X_train_org[i, x, y, c] // 32)
                                else:
                                        code = encoding[(X_train_org[i, x, y, c] // 32)]
                                        for bit in code:
                                                X_train[i, x * X_train_org.shape[2] * hypervector_size + y * hypervector_size + ((bit + roll) % hypervector_size)] = 1

X_test = lil_matrix((X_test_org.shape[0], X_test_org.shape[1] * X_test_org.shape[2] * hypervector_size), dtype=np.uint32)
for i in range(X_test.shape[0]):
        for x in range(X_test_org.shape[1]):
                for y in range(X_test_org.shape[2]):
                        roll = 0
                        for c in range(X_test_org.shape[3]):
                                if c == 0:
                                        roll += (X_test_org[i, x, y, c] // 32)
                                elif c == 1:
                                        roll += 11 * (X_test_org[i, x, y, c] // 32)
                                else:
                                        code = encoding[(X_test_org[i, x, y, c] // 32)]
                                        for bit in code:
                                                X_test[i, x * X_test_org.shape[2] * hypervector_size + y * hypervector_size + ((bit + roll) % hypervector_size)] = 1

print(X_test.shape, X_test.shape)

#X_train = X_train.reshape((X_train.shape[0], -1))
#X_test = X_test.reshape((X_test.shape[0], -1))

print(X_test.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d.txt" % (s, clauses, T, scaling), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, hypervector_size), (3, 3), max_included_literals=max_included_literals)

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
