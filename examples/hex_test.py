import argparse
import numpy as np
import pandas as pd
from time import time
# Graph Tsetlin Machine stuff
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=6000, type=int)
    parser.add_argument("--T", default=7000, type=int)
    parser.add_argument("--s", default=1.2, type=float)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=2056, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=2, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

# Load data
data = pd.read_csv('3x3_small.csv')

board_size = 3
subset_size = 1000
test_size = 100
X = data.iloc[:subset_size, 0].values
X_test = data.iloc[subset_size:subset_size + test_size, 0].values
y = data.iloc[:subset_size, 1].values
y_test = data.iloc[subset_size:subset_size + test_size, 1].values

print(np.sum(y))

symbol_names = []
for i in range(board_size**2):
    symbol_names.append(i)

    
print("Creating training data")
graphs_train = Graphs(
    number_of_graphs=subset_size,
    symbol_names=symbol_names,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing,
)

# Prepare nodes
for graph_id in range(X.shape[0]):
    graphs_train.set_number_of_graph_nodes(
        graph_id=graph_id,
        number_of_graph_nodes=1,
    )
graphs_train.prepare_node_configuration()

# Prepare edges
for graph_id in range(X.shape[0]):
    graphs_train.add_graph_node(graph_id, 0, 0)
graphs_train.prepare_edge_configuration()

# Create the graph
for graph_id in range(X.shape[0]):
    for k in range(board_size**2):
        sym = X[graph_id][k]
        if sym == 'O':
            graphs_train.add_graph_node_feature(graph_id, 0, k)
graphs_train.encode()

print("Printing graph:")
print(X[0])
print(graphs_train.number_of_graph_nodes[0])
graphs_train.print_graph(0)
print(X[1])
graphs_train.print_graph(1)
print(graphs_train.symbol_id.items())

print("Creating test data")
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

# Prepare nodes
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(
        graph_id=graph_id,
        number_of_graph_nodes=1,
    )
graphs_test.prepare_node_configuration()

# Prepare edges
for graph_id in range(X_test.shape[0]):
    graphs_test.add_graph_node(graph_id, 0, 0)
graphs_test.prepare_edge_configuration()

# Create the graph
for graph_id in range(X_test.shape[0]):
    for k in range(board_size**2):
        sym = X_test[graph_id][k]
        if sym == 'O':
            graphs_test.add_graph_node_feature(graph_id, 0, k)
graphs_test.encode()

# Train the Tsetlin Machine
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals
)

start_training = time()
for i in range(args.epochs):
    tm.fit(graphs_train, y, epochs=1, incremental=True)
    print(f"Epoch#{i+1} -- Accuracy train: {np.mean(y == tm.predict(graphs_train))}", end=' ')
    print(f"-- Accuracy test: {np.mean(y_test == tm.predict(graphs_test))} ", end=' ')
    print(np.sum(tm.predict(graphs_train))/X.shape[0], end=' ')
    print(np.sum(tm.predict(graphs_test))/X_test.shape[0])
stop_training = time()
print(f"Time: {stop_training - start_training}")


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

