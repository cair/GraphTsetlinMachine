import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import argparse

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

animals = np.array([2, 3, 4, 5, 6, 7])

parser = argparse.ArgumentParser()
parser.add_argument("--clauses", default=100, type=int)
parser.add_argument("--T", default=1000, type=int)
parser.add_argument("--s", default=1.0, type=float)
parser.add_argument("--epochs", default=250, type=int)
parser.add_argument("--ensembles", default=5, type=int)
parser.add_argument("--resolution", default=8, type=int)
parser.add_argument("--max_included_literals", default=32, type=int)
parser.add_argument("--convolution_size", default=1, type=int)
parser.add_argument("--number_of_examples", default=5000, type=int)
parser.add_argument("--animals",  action='store_true')

args = parser.parse_args()

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
X_train_org = X_train_org[0:args.number_of_examples]
X_test_org = X_test_org[0:args.number_of_examples]

Y_train = Y_train.reshape(Y_train.shape[0])[0:args.number_of_examples]
Y_test = Y_test.reshape(Y_test.shape[0])[0:args.number_of_examples]

if args.animals:
        Y_train = np.where(np.isin(Y_train, animals), 1, 0)
        Y_test = np.where(np.isin(Y_test, animals), 1, 0)

X_train = np.zeros((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], args.resolution**3), dtype=np.uint32)
for i in range(X_train.shape[0]):
        for x in range(X_train_org.shape[1]):
                for y in range(X_train_org.shape[2]):
                        index = (X_train_org[i, x, y, 0] // (256//args.resolution))*(args.resolution**2) + (X_train_org[i, x, y, 1] // (256//args.resolution))*args.resolution + (X_train_org[i, x, y, 2] // (256//args.resolution))
                        X_train[i, x, y, index] = 1 

X_test = np.zeros((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], args.resolution**3), dtype=np.uint32)
for i in range(X_test.shape[0]):
        for x in range(X_test_org.shape[1]):
                for y in range(X_test_org.shape[2]):
                        index = (X_test_org[i, x, y, 0] // (256//args.resolution))*(args.resolution**2) + (X_test_org[i, x, y, 1] // (256//args.resolution))*args.resolution + (X_test_org[i, x, y, 2] // (256//args.resolution))
                        X_test[i, x, y, index] = 1 

print(X_test.shape, X_test.shape)

X_train = csr_matrix(X_train.reshape((X_train.shape[0], -1)))
X_test = csr_matrix(X_test.reshape((X_test.shape[0], -1)))

print(X_test.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d_%d_%d.txt" % (args.s, args.clauses, args.T, args.resolution, args.convolution_size, args.max_included_literals), "w+")
for ensemble in range(args.ensembles):
        print("\nAccuracy over %d epochs:\n" % (args.epochs))

        tm = MultiClassConvolutionalTsetlinMachine2D(args.clauses, args.T, args.s, (32, 32, args.resolution**3), (args.convolution_size, args.convolution_size), max_included_literals=args.max_included_literals)

        for epoch in range(args.epochs):
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
