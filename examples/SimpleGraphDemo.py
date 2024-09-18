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

number_of_nodes_train = np.empty(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
    number_of_nodes_train[i] = np.random.randint(1, args.max_sequence_length)

graphs_train = Graphs(number_of_nodes_train, symbols=['A', 'B'], hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)
Y_train = np.empty(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
    # Select class
    Y_train[i] = np.random.randint(args.number_of_classes)

    feature_pos = np.random.randint(number_of_nodes_train[i])
    for j in range(number_of_nodes_train[i]):
        if feature_pos == j:
            if Y_train[i] == 0:
                graphs_train.add_node_feature(i, j, 'A')
            else:
                graphs_train.add_node_feature(i, j, 'B')
   
Y_train = np.where(np.random.rand(args.number_of_examples) < args.noise, 1 - Y_train, Y_train)  # Adds noise

graphs_train.encode()

number_of_nodes_test = np.empty(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
    number_of_nodes_test[i] = np.random.randint(1, args.max_sequence_length)

graphs_test = Graphs(number_of_nodes_test, init_with=graphs_train)
Y_test = np.empty(args.number_of_examples, dtype=np.uint32)

for i in range(args.number_of_examples):
    # Select class
    Y_test[i] = np.random.randint(args.number_of_classes)

    feature_pos = np.random.randint(number_of_nodes_test[i])
    for j in range(number_of_nodes_test[i]):
        if feature_pos == j:
            if Y_test[i] == 0:
                graphs_test.add_node_feature(i, j, 'A')
            else:
                graphs_test.add_node_feature(i, j, 'B')
   
graphs_test.encode()

print(graphs_test.hypervectors)

tm = MultiClassGraphTsetlinMachine(args.number_of_clauses, args.T, args.s, (1, args.max_sequence_length, args.hypervector_size), (1, 1), max_included_literals=args.max_included_literals)

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