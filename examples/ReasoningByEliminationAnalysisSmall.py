from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from time import time
import random

number_of_training_examples = 10000
number_of_testing_examples = 10000

noise = [0.0, 0.05, 0.1]

number_of_features = 1000

number_of_characterizing_features = 100 # Each class gets this many unique features in total

number_of_low_precision_features = 1

number_of_characterizing_features_per_example = 1 # Each example consists of this number of unique features
number_of_common_features_per_example = 10

number_of_clauses = 100
T = number_of_clauses*10
s = 100.0
number_of_blocks = 2

a = 1.1
b = 2.7

low_precision_features_p = 0.5

#characterizing_features = np.random.choice(number_of_features, size=(2, number_of_characterizing_features), replace=False).astype(np.uint32)
characterizing_features = np.arange(number_of_characterizing_features*2).reshape((2, number_of_characterizing_features)).astype(np.uint32)
low_precision_features = np.arange(number_of_characterizing_features*2, number_of_characterizing_features*2 + number_of_low_precision_features*2).reshape((2, number_of_low_precision_features)).astype(np.uint32)
common_features = np.arange(number_of_characterizing_features*2 + number_of_low_precision_features*2, number_of_features)

p_common_feature = np.empty(common_features.shape[0])
for k in range(common_features.shape[0]):
	p_common_feature[k] = (k + b)**(-a)
p_common_feature = p_common_feature / p_common_feature.sum()

symbols = []
for i in range(number_of_features):
    symbols.append("x%d" % (i))

graphs_train = Graphs(
    number_of_training_examples,
    symbols=symbols,
    one_hot_encoding = True
)

for graph_id in range(number_of_training_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, 1)

graphs_train.prepare_node_configuration()

for graph_id in range(number_of_training_examples):
    graphs_train.add_graph_node(graph_id, 'X', 0)

graphs_train.prepare_edge_configuration()

Y_train = np.zeros(number_of_training_examples, dtype=np.uint32)
for graph_id in range(number_of_training_examples):
	Y_train[graph_id] = np.random.choice(2)

	indexes = np.random.choice(characterizing_features[Y_train[graph_id]], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		graphs_train.add_graph_node_property(graph_id, "X", "x%d" % (j))

#	for j in low_precision_features[Y_train[i]]:
#		if random.random() <= low_precision_features_p:
#			graphs_train.add_graph_node_property(graph_id, "X", "x%d" % (j))

#	for j in low_precision_features[1-Y_train[i]]:
#		if random.random() <= 1.0 - low_precision_features_p:
#			graphs_train.add_graph_node_property(graph_id, "X", "x%d" % (j))

#	indexes = np.random.choice(characterizing_features[1 - Y_train[i]], number_of_characterizing_features_per_example, replace=False)
#	for j in indexes:
#		if random.random() <= noise[j%len(noise)]:
#			graphs_train.add_graph_node_property(graph_id, "X", "x%d" % (j))

	indexes = np.random.choice(common_features, number_of_common_features_per_example, replace=False, p=p_common_feature)
	for j in indexes:
		graphs_train.add_graph_node_property(graph_id, "X", "x%d" % (j))

graphs_train.encode()

print("Training data produced")

graphs_test = Graphs(
	number_of_testing_examples,
	init_with=graphs_train
)

for graph_id in range(number_of_testing_examples):
    graphs_test.set_number_of_graph_nodes(graph_id, 1)

graphs_test.prepare_node_configuration()

for graph_id in range(number_of_testing_examples):
    graphs_test.add_graph_node(graph_id, 'X', 0)

graphs_test.prepare_edge_configuration()

Y_test = np.zeros(number_of_testing_examples, dtype=np.uint32)
for graph_id in range(number_of_testing_examples):
	Y_test[graph_id] = np.random.choice(2)

	indexes = np.random.choice(characterizing_features[Y_test[graph_id]], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		graphs_test.add_graph_node_property(graph_id, "X", "x%d" % (j))

#	for j in low_precision_features[Y_test[i]]:
#		if random.random() <= low_precision_features_p:
#			graphs_test.add_graph_node_property(graph_id, "X", "x%d" % (j))

#	for j in low_precision_features[1-Y_test[i]]:
#		if random.random() <= 1.0 - low_precision_features_p:
#			graphs_test.add_graph_node_property(graph_id, "X", "x%d" % (j))

	indexes = np.random.choice(common_features, number_of_common_features_per_example, replace=False, p=p_common_feature)
	for j in indexes:
		graphs_test.add_graph_node_property(graph_id, "X", "x%d" % (j))

graphs_test.encode()

print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(
    number_of_clauses,
    T,
    s,
    number_of_state_bits = 8,
    depth=1,
    max_included_literals=100,
    double_hashing = False,
    one_hot_encoding = True,
    number_of_blocks=number_of_blocks
)

start = time()
for epoch in range(100):
	tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop = time()

print(stop-start)

print("Accuracy:", 100*(tm.predict(graphs_test) == Y_test).mean())

np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

print("\nClass 0 Positive Clauses:\n")

precision = tm.clause_precision(0, 0, graphs_test, Y_test)
recall = tm.clause_recall(0, 0, graphs_test, Y_test)

for j in range(number_of_clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k, the_class = 0, polarity = 0):
			if k < number_of_features:
				l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 0, polarity = 0)))
			else:
				l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 0, polarity = 0)))
	print(" ∧ ".join(l))

print("\nClass 0 Negative Clauses:\n")

precision = tm.clause_precision(0, 1, X_test, Y_test)
recall = tm.clause_recall(0, 1, X_test, Y_test)

for j in range(number_of_clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k, the_class = 0, polarity = 1):
			if k < number_of_features:
				l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
			else:
				l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
	print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")

precision = tm.clause_precision(1, 0, X_test, Y_test)
recall = tm.clause_recall(1, 0, X_test, Y_test)

for j in range(number_of_clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
			if k < number_of_features:
				l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
			else:
				l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
	print(" ∧ ".join(l))

print("\nClass 1 Negative Clauses:\n")

precision = tm.clause_precision(1, 1, X_test, Y_test)
recall = tm.clause_recall(1, 1, X_test, Y_test)

for j in range(number_of_clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.get_ta_action(j, k, the_class = 1, polarity = 1):
			if k < number_of_features:
				l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
			else:
				l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
	print(" ∧ ".join(l))

print("\nLiteral Clause Frequency:", end=' ')
literal_frequency = tm.literal_clause_frequency()
sorted_literals = np.argsort(-1*literal_frequency)
for k in sorted_literals:
		if literal_frequency[k] == 0:
			break

		if k < number_of_features:
			print("%d(%.2f,%d,%d)" % (k, 1.0*literal_frequency[k]/number_of_clauses, (k - number_of_features) % len(noise), k in common_features), end=' ')
		else:
			print("¬%d(%.2f,%d,%d)" % (k - number_of_features, 1.0*literal_frequency[k]/number_of_clauses, (k - number_of_features) % len(noise), k in common_features), end=' ')
print()

for i in range(2):
	print("\nLiteral Importance Class #%d:" % (i), end=' ')

	literal_importance = tm.literal_importance(i, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print(k, end=' ')

	literal_importance = tm.literal_importance(i, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print("¬" + str(k - number_of_features), end=' ')
	print()

print(characterizing_features, common_features.shape)

