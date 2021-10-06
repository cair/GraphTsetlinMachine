from PyCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine
import numpy as np

noise = 0.1
number_of_features = 12

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.zeros((X_train.shape[0], 3), dtype=np.uint32)
Y_train[:,0] = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train[:,1] = np.logical_and(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train[:,2] = np.logical_or(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000,3) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)

Y_test = np.zeros((X_test.shape[0], 3), dtype=np.uint32)
Y_test[:,0] = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)
Y_test[:,1] = np.logical_and(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)
Y_test[:,2] = np.logical_or(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

average_accuracy = 0.0

for i in range(100):
	tm = MultiOutputTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

	tm.fit(X_train, Y_train, epochs=200)

	print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

	average_accuracy += 100*(tm.predict(X_test) == Y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))