import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from keras.datasets import mnist
from skimage import io
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine

noise = 0.0

number_of_features = 28*28

number_of_examples = 1000

number_of_clauses = 1024
T = int(number_of_clauses*0.75)
s = 10.0
max_included_literals = 2*number_of_features
accumulation = 1
clause_weight_threshold = 0
upsampling=1

print("Number of clauses:", number_of_clauses)
print("T:", T)
print("Number of features:", number_of_features)

output_active = np.arange(number_of_features, dtype=np.uint32)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_train = csr_matrix(X_train)
X_test = csr_matrix(X_test)

tm = MultiOutputConvolutionalTsetlinMachine2D(number_of_clauses, T, s, (28, 28, 1), (10, 10), max_included_literals=max_included_literals)

print("\nAccuracy Over 40 Epochs:")
for e in range(100):
	start_training = time()
	tm.fit(X_train, X_train.toarray(), epochs=1, incremental=True)
	stop_training = time()

	print("\nEpoch #%d\n" % (e+1))

	X_test_predicted = tm.predict(X_test)
	X_train_predicted = tm.predict(X_train)

	print("Test accuracy", 100*(X_test_predicted == X_test).mean())

	print("Train accuracy", 100*(X_train_predicted == X_train).mean())

	print("\nTraining Time: %.2f" % (stop_training - start_training))

	tm_2 = MultiClassTsetlinMachine(2000, 50*100, 10.0)

	print("\n\tAccuracy over 10 epochs:\n")
	for i in range(3):
		start_training = time()
		tm_2.fit(X_train_predicted, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result = 100*(tm_2.predict(X_test_predicted) == Y_test).mean()
		stop_testing = time()

		print("\t#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

