from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
#from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
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

graphs_train = Graphs(args.number_of_examples, hypervector_size=16, hypervector_bits=1)
for i in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(i, 4)

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_train.number_of_graph_nodes[graph_id]-1 else 1
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)

graphs_train.prepare_edge_configuration()

for graph_id in range(args.number_of_examples):
    for source_node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        if source_node_id > 0:
            destination_node_id = source_node_id - 1
            edge_type = 0
            graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

        if source_node_id < graphs_train.number_of_graph_nodes[graph_id]-1:
            destination_node_id = source_node_id + 1
            edge_type = 0
            graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

print(graphs_train.number_of_graph_node_edges)
print(graphs_train.edge_index)
print(graphs_train.graph_node_edge.shape)
print(graphs_train.graph_node_edge)
freer

Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
    graph_name = "G%d" % (i)
    graphs_train.add_graph(graph_name)
    
    # Create nodes

    number_of_nodes = 4#np.random.randint(1, args.max_sequence_length)
    for j in range(number_of_nodes):
        node_name = "N%d" % (j)
        graphs_train.add_graph_node(graph_name, node_name)

    # Add node features

    Y_train[i] = np.random.randint(args.number_of_classes)

    j = np.random.randint(number_of_nodes)
    node_name = "N%d" % (j)
    if Y_train[i] == 0:
        graphs_train.add_graph_node_feature(graph_name, node_name, 'A')
    else:
        graphs_train.add_graph_node_feature(graph_name, node_name, 'B')
   
    # Add node edges

    for j in range(number_of_nodes):
        node_name = "N%d" % (j)

        if j > 0:
            previous_node_name = "N%d" % (j-1)
            graphs_train.add_graph_node_edge(graph_name, node_name, 'Plain', previous_node_name)
        
        if j < number_of_nodes-1:
            next_node_name = "N%d" % (j+1)
            graphs_train.add_graph_node_edge(graph_name, node_name, 'Plain', next_node_name)

Y_train = np.where(np.random.rand(args.number_of_examples) < args.noise, 1 - Y_train, Y_train)  # Add noise

graphs_train.encode(hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)

graphs_test = Graphs(args.number_of_examples, init_with=graphs_train)
Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
    graph_name = "G%d" % (i)
    graphs_test.add_graph(graph_name)
    
    # Create nodes

    number_of_nodes = 4#np.random.randint(1, args.max_sequence_length)
    for j in range(number_of_nodes):
        node_name = "N%d" % (j)
        graphs_test.add_graph_node(graph_name, node_name)

    # Add node features

    Y_test[i] = np.random.randint(args.number_of_classes)

    j = np.random.randint(number_of_nodes)
    node_name = "N%d" % (j)
    if Y_test[i] == 0:
        graphs_test.add_graph_node_feature(graph_name, node_name, 'A')
    else:
        graphs_test.add_graph_node_feature(graph_name, node_name, 'B')

    # Add node edges

    for j in range(number_of_nodes):
        node_name = "N%d" % (j)

        if j > 0:
            previous_node_name = "N%d" % (j-1)
            graphs_test.add_graph_node_edge(graph_name, node_name, 'Plain', previous_node_name)
        
        if j < number_of_nodes-1:
            next_node_name = "N%d" % (j+1)
            graphs_test.add_graph_node_edge(graph_name, node_name, 'Plain', next_node_name)

graphs_test.encode()

print(graphs_test.edge_target)

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