import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from numba import jit

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0

hypervector_size = 64

bits = 5

resolution = 8

animals = np.array([2, 3, 4, 5, 6, 7])

ensembles = 5
epochs = 250

examples = 1000

max_included_literals = 32

#clauses = 8000*2
#T = int(clauses * 0.75)

clauses = 2000
T = 5000

s = 1.5

@jit(nopython=True)
def count_nonzero_hypervector(hypervector, encoding, X):
        nonzero_count = 0
        for i in range(X.shape[0]):
                for x in range(X.shape[1]):
                        for y in range(X.shape[2]):
                                hypervector[:] = 0
                                roll = 0
                                for c in range(3):
                                        if c == 0:
                                                roll += (X[i, x, y, c] // 32)
                                        elif c == 1:
                                                roll += 11 * (X[i, x, y, c] // 32)
                                        else:
                                                code = encoding[(X[i, x, y, c] // 32)]
                                                for bit in code:
                                                        hypervector[((bit + roll) % hypervector_size)] = 1
                                nonzero_count += hypervector.sum()
        return nonzero_count

@jit(nopython=True)
def produce_hypervectors(hypervector, hypervector_size, encoding, X, indptr, indices, data):
        nonzero_count = 0
        for i in range(X.shape[0]):
                indptr[i] = nonzero_count
                for x in range(X.shape[1]):
                        for y in range(X.shape[2]):
                                hypervector[:] = 0
                                roll = 0
                                for c in range(3):
                                        if c == 0:
                                                roll += (X[i, x, y, c] // 32)
                                        elif c == 1:
                                                roll += 11 * (X[i, x, y, c] // 32)
                                        else:
                                                code = encoding[(X[i, x, y, c] // 32)]
                                                for bit in code:
                                                        hypervector[((bit + roll) % hypervector_size)] = 1
                                nonzero = hypervector.nonzero()[0]

                                for bit in nonzero:
                                        indices[nonzero_count] = x * X.shape[2] * hypervector_size + y * hypervector_size + bit
                                        data[nonzero_count] = 1
                                        nonzero_count += 1
        indptr[X.shape[0]] = nonzero_count

        return nonzero_count

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = np.zeros((resolution, bits), dtype=np.uint32)
for i in range(resolution):
        encoding[i] = np.random.choice(indexes, size=(bits))

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org[0:examples]
X_test_org = X_test_org[0:examples]
Y_train = Y_train.reshape(Y_train.shape[0])[0:examples]
Y_test = Y_test.reshape(Y_test.shape[0])[0:examples]

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

print("Preprocessing")

hypervector = np.zeros(hypervector_size, dtype=np.uint32)

start_preprocessing = time()
nonzero_count = count_nonzero_hypervector(hypervector, encoding, X_train_org)
stop_preprocessing = time()

print(stop_preprocessing-start_preprocessing, nonzero_count)

print("Training Data")

X_train_indptr = np.empty(X_train_org.shape[0]+1, dtype=np.uint32)
X_train_indices = np.empty(nonzero_count, dtype=np.uint32)
X_train_data = np.empty(nonzero_count, dtype=np.uint32)

produce_hypervectors(hypervector, hypervector_size, encoding, X_train_org, X_train_indptr, X_train_indices, X_train_data)

X_train = csr_matrix((X_train_data, X_train_indices, X_train_indptr))

print("Testing Data")

X_test_indptr = np.empty(X_test_org.shape[0]+1, dtype=np.uint32)
X_test_indices = np.empty(nonzero_count, dtype=np.uint32)
X_test_data = np.empty(nonzero_count, dtype=np.uint32)

produce_hypervectors(hypervector, hypervector_size, encoding, X_test_org, X_test_indptr, X_test_indices, X_test_data)

X_test = csr_matrix((X_test_data, X_test_indices, X_test_indptr))

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
