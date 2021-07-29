from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

factor = 10

s = 10.0

T = int(factor*50*10)

ensembles = 10
epochs = 250

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(int(factor*2000), T, s)

print("\nAccuracy over 100 epochs:\n")
for i in range(100):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
