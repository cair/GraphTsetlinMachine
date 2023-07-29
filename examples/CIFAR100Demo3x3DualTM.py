from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar100

max_included_literals = 32
clauses = 4000*40
T = int(clauses * 0.75)
s = 10.0 # 5.0
q = 16 # q = 32
patch_size = 3
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0

epochs = 1000
ensembles = 5

(X_train_org, Y_train), (X_test_org, Y_test) = cifar100.load_data()

X_train_org = X_train_org[0:250]
X_test_org = X_test_org[0:250]
Y_train = Y_train.reshape(Y_train.shape[0])[0:250]
Y_test = Y_test.reshape(Y_test.shape[0])[0:250]

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution),
                   dtype=np.uint32)
for z in range(resolution):
    X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution),
                  dtype=np.uint32)
for z in range(resolution):
    X_test[:, :, :, :, z] = X_test_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution)).reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution)).reshape((X_test.shape[0], -1))


f = open("cifar100_%.1f_%d_%d_%d_%.2f_%d_%d.txt" % (
s, clauses, T, patch_size, literal_drop_p, resolution, max_included_literals), "w+")
for ensemble in range(ensembles):
    tm_positive = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, 3*resolution), (patch_size, patch_size), max_included_literals=max_included_literals, q=q)
    tm_negative = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (32, 32, 3*resolution), (patch_size, patch_size), max_included_literals=max_included_literals, q=-1*q)

    for epoch in range(epochs):
        start_training = time()
        tm_positive.fit(X_train, Y_train, incremental=True, epochs=1)
        tm_negative.fit(X_train, Y_train, incremental=True, epochs=1)
        stop_training = time()

        start_testing = time()
        result_test_positive = 100 * (tm_positive.predict(X_test) == Y_test).mean()
        result_test_negative = 100 #* (tm_negative.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train_positive = 100 #* (tm_positive.predict(X_train) == Y_train).mean()
        result_train_negative = 100 #* (tm_negative.predict(X_train) == Y_train).mean()

        #joint_Y_predicted_test = np.argmax(tm_positive.score(X_test) + tm_negative.score(X_test), axis=1)
        #joint_Y_predicted_train = np.argmax(tm_positive.score(X_train) + tm_negative.score(X_train), axis=1)

        result_train_joint = 100 #* (joint_Y_predicted_train == Y_train).mean()
        result_test_joint = 100 #* (joint_Y_predicted_test == Y_test).mean()

        print("%d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, result_train_joint, result_test_joint, result_train_positive, result_test_positive, result_train_negative, result_test_negative, stop_training - start_training, stop_testing - start_testing))
        print("%d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, result_train_joint, result_test_joint, result_train_positive, result_test_positive, result_train_negative, result_test_negative, stop_training - start_training, stop_testing - start_testing),
              file=f)
        f.flush()
f.close()
