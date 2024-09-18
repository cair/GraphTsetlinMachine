from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from skimage.util import view_as_windows
from keras.datasets import mnist
from numba import jit

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=2500, type=int)
    parser.add_argument("--s", default=1.5, type=float)
    parser.add_argument("--hypervector_size", default=128, type=int)
    parser.add_argument("--hypervector_bits", default=2, type=int)
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

symbols = []
for i in range(100):
    symbols.append(i)

for i in range(19):
    symbols.append('c:' + str(i))
    symbols.append('r:' + str(i))

number_of_train_nodes = np.repeat(19*19, Y_train.shape[0]).astype(np.uint32)

graphs_train = Graphs(number_of_train_nodes, symbols=symbols, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)
for i in range(X_train.shape[0]):
    if i % 1000 == 0:
        print(i, X_train.shape[0])
     
    windows = view_as_windows(X_train[i,:,:], (10, 10))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q*19 + r
                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_train.add_node_feature(i, node_id, k)
                graphs_train.add_node_feature(i, node_id, 'c:'+str(q))
                graphs_train.add_node_feature(i, node_id, 'r:'+str(r))

graphs_train.encode()

print("Training data produced")

number_of_test_nodes = np.repeat(19*19, Y_test.shape[0]).astype(np.uint32)

graphs_test = Graphs(number_of_test_nodes, init_with=graphs_train)
for i in range(X_test.shape[0]):
    if i % 1000 == 0:
        print(i, X_test.shape[0])
     
    windows = view_as_windows(X_test[i,:,:], (10, 10))
    for q in range(windows.shape[0]):
            for r in range(windows.shape[1]):
                node_id = q*19 + r
                patch = windows[q,r].reshape(-1).astype(np.uint32)
                for k in patch.nonzero()[0]:
                    graphs_test.add_node_feature(i, node_id, k)
                graphs_test.add_node_feature(i, node_id, 'c:'+str(q))
                graphs_test.add_node_feature(i, node_id, 'r:'+str(r))

graphs_test.encode()

print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(args.number_of_clauses, args.T, args.s, (1, 19*19, args.hypervector_size), (1, 1), max_included_literals=args.max_included_literals)

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
            if tm.ta_action(i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))
        print(" AND ".join(l))


start_training = time()
tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
stop_training = time()

start_testing = time()
result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
stop_testing = time()

result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))

print(graphs_train.hypervectors)