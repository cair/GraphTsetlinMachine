from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--number-of-clauses", default=20, type=int)
    parser.add_argument("--T", default=200, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--hypervector_size", default=16, type=int)
    parser.add_argument("--hypervector_bits", default=1, type=int)
    parser.add_argument("--noise", default=0.2, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--max-sequence-length", default=1000, type=int)
    parser.add_argument("--number-of-classes", default=2, type=int)
    parser.add_argument("--max-included-literals", default=2, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

print("Creating training data")

# Create train data

graphs_train = Graphs(args.number_of_examples, symbol_names=['A', 'B'], hypervector_size=16, hypervector_bits=1)
for graph_id in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, np.random.randint(1, args.max_sequence_length))

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_train.number_of_graph_nodes[graph_id]-1 else 1
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)

graphs_train.prepare_edge_configuration()

Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        if node_id > 0:
            destination_node_id = node_id - 1
            edge_type = 0
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < graphs_train.number_of_graph_nodes[graph_id]-1:
            destination_node_id = node_id + 1
            edge_type = 0
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    Y_train[graph_id] = np.random.randint(args.number_of_classes)
    node_id = 0
    if Y_train[graph_id] == 0:
        graphs_train.add_graph_node_feature(graph_id, node_id, 'A')
    else:
        graphs_train.add_graph_node_feature(graph_id, node_id, 'B')
        
Y_train = np.where(np.random.rand(args.number_of_examples) < args.noise, 1 - Y_train, Y_train)  # Add noise

graphs_train.encode()

# Create test data

print("Creating testing data")

graphs_test = Graphs(args.number_of_examples, init_with = graphs_train)
for graph_id in range(args.number_of_examples):
    graphs_test.set_number_of_graph_nodes(graph_id, np.random.randint(1, args.max_sequence_length))

graphs_test.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_test.number_of_graph_nodes[graph_id]-1 else 1
        graphs_test.add_graph_node(graph_id, node_id, number_of_edges)

graphs_test.prepare_edge_configuration()

Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        if node_id > 0:
            destination_node_id = node_id - 1
            edge_type = 0
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < graphs_test.number_of_graph_nodes[graph_id]-1:
            destination_node_id = node_id + 1
            edge_type = 0
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    Y_test[graph_id] = np.random.randint(args.number_of_classes)
    node_id = 0
    if Y_test[graph_id] == 0:
        graphs_test.add_graph_node_feature(graph_id, node_id, 'A')
    else:
        graphs_test.add_graph_node_feature(graph_id, node_id, 'B')
        
graphs_test.encode()

tm = MultiClassGraphTsetlinMachine(args.number_of_clauses, args.T, args.s, max_included_literals=args.max_included_literals)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))
        print(" AND ".join(l))

print(graphs_train.hypervectors)
print(graphs_train.symbol_id)
print(graphs_test.hypervectors)
print(graphs_test.symbol_id)
