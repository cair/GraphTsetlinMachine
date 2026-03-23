import random
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs

random.seed(42)

def generate_graphs(symbols, noise, graph_args: dict):
	graphs = Graphs(**graph_args)
	number_of_examples = graph_args["number_of_graphs"]

	for graph_id in range(number_of_examples):
		graphs.set_number_of_graph_nodes(graph_id, 2)

	graphs.prepare_node_configuration()

	for graph_id in range(number_of_examples):
		for node_id in range(graphs.number_of_graph_nodes[graph_id]):
			number_of_edges = 1
			graphs.add_graph_node(graph_id, node_id, number_of_edges)

	graphs.prepare_edge_configuration()

	X = np.empty((number_of_examples, 2))
	Y = np.empty(number_of_examples, dtype=np.uint32)

	for graph_id in range(number_of_examples):
		edge_type = "Plain"
		source_node_id = 0
		destination_node_id = 1
		graphs.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

		source_node_id = 1
		destination_node_id = 0
		graphs.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

		x1 = random.choice(symbols)
		x2 = random.choice(symbols)
		X[graph_id] = np.array([x1, x2])
		if (x1 % 2) == (x2 % 2):
			Y[graph_id] = 0
		else:
			Y[graph_id] = 1

		graphs.add_graph_node_property(graph_id, 0, x1)
		graphs.add_graph_node_property(graph_id, 1, x2)

		if np.random.rand() <= noise:
			Y[graph_id] = 1 - Y[graph_id]

	graphs.encode()

	return graphs, X, Y


if __name__ == "__main__":
	tm_params = {
		"number_of_clauses": 1000,
		"T": 2000,
		"s": 1,
		"message_size": 2048,
		"message_bits": 2,
		"double_hashing": True,
		"depth": 2,
		"grid": (16 * 13, 1, 1),
		"block": (128, 1, 1),
	}

	epochs = 10
	noise = 0.1
	num_value = 100
	symbols = [i for i in range(num_value)]
	graph_params = {
		"number_of_graphs": 50000,
		"hypervector_size": 2048,
		"hypervector_bits": 2,
		"double_hashing": True,
		"symbols": symbols,
	}
	graphs_train, X_train, y_train = generate_graphs(symbols, noise, graph_params)

	graphs_test, X_test, y_test = generate_graphs(
		symbols,
		0.0,
		{
			"number_of_graphs": 2000,
			"init_with": graphs_train,
		},
	)

	print("====================Training with graph splits====================")
	tm = MultiClassGraphTsetlinMachine(**tm_params)
	for i in range(epochs):
		print(f"Epoch {i} ---------------------")
		fit_time = 0.0
		for b in range(0, y_train.shape[0], 10000):
			gsub = graphs_train[b : b + 10000]
			ysub = y_train[b : b + 10000]
			tm.fit(gsub, ysub, epochs=1, incremental=True)
			result_sub = 100 * (tm.predict(gsub) == ysub).mean()
			print(f"  [Batch {b}-{b + 10000}] Train Acc: {result_sub:.4f}")

		pred_test = tm.predict(graphs_test)
		result_test = 100 * (pred_test == y_test).mean()
		result_train = 100 * (tm.predict(graphs_train) == y_train).mean()
		print(f"[Graph Splits] Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f}")

	print("====================Training with original graphs====================")
	tm2 = MultiClassGraphTsetlinMachine(**tm_params)
	for i in range(epochs):
		tm2.fit(graphs_train, y_train, epochs=1, incremental=True)

		pred_test = tm2.predict(graphs_test)

		result_test = 100 * (pred_test == y_test).mean()
		result_train = 100 * (tm2.predict(graphs_train) == y_train).mean()

		print(
			f"[Original Graphs] Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f}"
		)

