import argparse
import pickle
import random
from time import time

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=2, type=int)
	parser.add_argument("--number-of-clauses", default=4, type=int)
	parser.add_argument("--T", default=100, type=int)
	parser.add_argument("--s", default=2.0, type=float)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--depth", default=2, type=int)
	parser.add_argument("--hypervector-size", default=8, type=int)
	parser.add_argument("--hypervector-bits", default=2, type=int)
	parser.add_argument("--message-size", default=16, type=int)
	parser.add_argument("--message-bits", default=2, type=int)
	parser.add_argument("--double-hashing", dest="double_hashing", default=False, action="store_true")
	parser.add_argument("--noise", default=0.01, type=float)
	parser.add_argument("--number-of-examples", default=10000, type=int)
	parser.add_argument("--max-included-literals", default=4, type=int)

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args


args = default_args()

############ Training Graphs ##############
print("Creating training data")
graphs_train = Graphs(
	args.number_of_examples,
	symbols=["A", "B"],
	hypervector_size=args.hypervector_size,
	hypervector_bits=args.hypervector_bits,
)

for graph_id in range(args.number_of_examples):
	graphs_train.set_number_of_graph_nodes(graph_id, 2)

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
	number_of_outgoing_edges = 1
	graphs_train.add_graph_node(graph_id, "Node 1", number_of_outgoing_edges)
	graphs_train.add_graph_node(graph_id, "Node 2", number_of_outgoing_edges)

graphs_train.prepare_edge_configuration()

for graph_id in range(args.number_of_examples):
	edge_type = "Plain"
	graphs_train.add_graph_node_edge(graph_id, "Node 1", "Node 2", edge_type)
	graphs_train.add_graph_node_edge(graph_id, "Node 2", "Node 1", edge_type)

X_train = np.empty((args.number_of_examples, 2), dtype=str)
Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
	x1 = random.choice(["A", "B"])
	x2 = random.choice(["A", "B"])

	X_train[graph_id, 0] = x1
	X_train[graph_id, 1] = x2

	graphs_train.add_graph_node_property(graph_id, "Node 1", x1)
	graphs_train.add_graph_node_property(graph_id, "Node 2", x2)

	if x1 == x2:
		Y_train[graph_id] = 0
	else:
		Y_train[graph_id] = 1

	if np.random.rand() <= args.noise:
		Y_train[graph_id] = 1 - Y_train[graph_id]

graphs_train.encode()
##########################################

############ Testing Graphs ##############
print("Creating testing data")

graphs_test = Graphs(args.number_of_examples, init_with=graphs_train)

for graph_id in range(args.number_of_examples):
	graphs_test.set_number_of_graph_nodes(graph_id, 2)

graphs_test.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
	number_of_outgoing_edges = 1
	graphs_test.add_graph_node(graph_id, "Node 1", number_of_outgoing_edges)
	graphs_test.add_graph_node(graph_id, "Node 2", number_of_outgoing_edges)

graphs_test.prepare_edge_configuration()

for graph_id in range(args.number_of_examples):
	edge_type = "Plain"
	graphs_test.add_graph_node_edge(graph_id, "Node 1", "Node 2", edge_type)
	graphs_test.add_graph_node_edge(graph_id, "Node 2", "Node 1", edge_type)

X_test = np.empty((args.number_of_examples, 2), dtype=str)
Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
	x1 = random.choice(["A", "B"])
	x2 = random.choice(["A", "B"])

	X_test[graph_id, 0] = x1
	X_test[graph_id, 1] = x2
	graphs_test.add_graph_node_property(graph_id, "Node 1", x1)
	graphs_test.add_graph_node_property(graph_id, "Node 2", x2)

	if x1 == x2:
		Y_test[graph_id] = 0
	else:
		Y_test[graph_id] = 1

graphs_test.encode()
##########################################

#############Model Train#################
tm = MultiClassGraphTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=args.number_of_state_bits,
	depth=args.depth,
	message_size=args.message_size,
	message_bits=args.message_bits,
	# max_included_literals=args.max_included_literals,
	double_hashing=args.double_hashing,
	grid=(1, 1, 1),
	block=(1, 1, 1),
)

for i in range(args.epochs):
	start_training = time()
	tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
	stop_testing = time()

	result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

	print(
		"%d %.2f %.2f %.2f %.2f"
		% (
			i,
			result_train,
			result_test,
			stop_training - start_training,
			stop_testing - start_testing,
		)
	)
##########################################
print("\nTA States and weights before save")
weights = tm.get_weights()
clause_states = tm.get_ta_states(depth=0)
message_states = tm.get_ta_states(depth=1)

print("Clause in Hyperliterals format:")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in clause_states[clause]])
print()

print("Messages as hypervectors:")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in message_states[clause]])

#############Model Save#################
state_dict = tm.save()

fname = "model.tm"
with open(fname, "wb") as f:
	pickle.dump(state_dict, f)

print(f"\nModel saved to {fname}.")


######Create new model and load#########
new_tm = MultiClassGraphTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=args.number_of_state_bits,
	depth=args.depth,
	message_size=args.message_size,
	message_bits=args.message_bits,
	# max_included_literals=args.max_included_literals,
	double_hashing=args.double_hashing,
	grid=(1, 1, 1),
	block=(1, 1, 1),
)

with open(fname, "rb") as f:
	loaded_state = pickle.load(f)

new_tm.load(loaded_state)
print("\n Created and Loaded new model.")

print("\nTA States and weights after load, before training")
weights = tm.get_weights()
clause_states = tm.get_ta_states(depth=0)
message_states = tm.get_ta_states(depth=1)

print("Clause in Hyperliterals format:")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in clause_states[clause]])
print()

print("Messages as hypervectors:")
for clause in range(tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in message_states[clause]])

######Coontinue training new model from the same states###############
print("\nTraining new model from the loaded states")
for i in range(args.epochs):
	start_training = time()
	new_tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result_test = 100 * (new_tm.predict(graphs_test) == Y_test).mean()
	stop_testing = time()

	result_train = 100 * (new_tm.predict(graphs_train) == Y_train).mean()

	print(
		"%d %.2f %.2f %.2f %.2f"
		% (
			i,
			result_train,
			result_test,
			stop_training - start_training,
			stop_testing - start_testing,
		)
	)


print("\nTA States and weights after training new model")
weights = new_tm.get_weights()
clause_states = new_tm.get_ta_states(depth=0)
message_states = new_tm.get_ta_states(depth=1)

print("Clause in Hyperliterals format:")
for clause in range(new_tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in clause_states[clause]])
print()

print("Messages as hypervectors:")
for clause in range(new_tm.number_of_clauses):
	print(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]", end=": ")
	print(*[state for state in message_states[clause]])
