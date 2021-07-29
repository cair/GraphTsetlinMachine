from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import cv2
from keras.datasets import fashion_mnist

factor = 20.0

s = 10.0

T = int(factor*25*10)

patch_size = 10

ensembles = 10
epochs = 250

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

f = open("fashion_%.1f_%d_%d_%d.txt" % (s, int(factor*2000), T,  patch_size), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(int(factor*2000), T, s, (10, 10))

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