import numpy as np
from time import time
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer
from skimage.transform import pyramid_gaussian, pyramid_laplacian, downscale_local_mean
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import cifar10

scaling = 1.0

resolution = 8

animals = np.array([2, 3, 4, 5, 6, 7])
random_grouping = np.random.choice(10, size=5, replace=False)

ensembles = 5
epochs = 250

max_included_literals = 32
clauses = 2000
T = 5000
s = 1.5

(X_train_org, Y_train_org), (X_test_org, Y_test_org) = cifar10.load_data()
X_train_org = X_train_org[0:5000]
X_test_org = X_test_org[0:5000]

Y_train_org = Y_train_org.reshape(Y_train_org.shape[0])[0:5000]
Y_test_org = Y_test_org.reshape(Y_test_org.shape[0])[0:5000]

Y_train = np.empty((Y_train_org.shape[0], 2), dtype=np.uint32)
Y_train[:, 0] = np.where(np.isin(Y_train_org, animals), 1, 0)
Y_train[:, 1] = np.where(np.isin(Y_train_org, random_grouping), 1, 0)

Y_test = np.empty((Y_test_org.shape[0], 2), dtype=np.uint32)
Y_test[:, 0] = np.where(np.isin(Y_test_org, animals), 1, 0)
Y_test[:, 1] = np.where(np.isin(Y_test_org, random_grouping), 1, 0)

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
                   dtype=np.uint8)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                  dtype=np.uint8)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], -1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], -1))

print(X_train.shape, X_test.shape)

f = open("cifar2_%.1f_%d_%d_%d.txt" % (s, clauses, T, scaling), "w+")
for ensemble in range(ensembles):
        print("\nAccuracy over %d epochs:\n" % (epochs))

        tm = MultiOutputConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, X_train_org.shape[3] * resolution), (3, 3), max_included_literals=max_included_literals)

        for epoch in range(epochs):
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()

                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()

                result_train = 100*(tm.predict(X_train) == Y_train).mean()

                number_of_includes = 0
                for i in range(2):
                        for j in range(clauses):
                                number_of_includes += tm.number_of_include_actions(i, j)
                number_of_includes /= 2*clauses

                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
                print("%d %d %.2f %.2f %.2f %.2f %.2f" % (ensemble, epoch, number_of_includes, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
                f.flush()
f.close()
