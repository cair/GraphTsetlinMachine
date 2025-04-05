from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from sklearn.metrics import confusion_matrix

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--number-of-clauses", default=60, type=int)
    parser.add_argument("--T", default=600, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=256, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--number-of-examples", default=40000, type=int)
    parser.add_argument("--max-subsequence-length", default=3, type=int)
    parser.add_argument("--max-sequence-length", default=5, type=int)
    parser.add_argument("--max-included-literals", default=4, type=int)
    parser.add_argument("--distance-from-edge", default=5, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

print("Creating training data")

# Create train data

graphs_train = Graphs(
    args.number_of_examples,
    symbols=['A'],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding
)

for graph_id in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, args.max_sequence_length)

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
            edge_type = "Left"
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < graphs_train.number_of_graph_nodes[graph_id]-1:
            destination_node_id = node_id + 1
            edge_type = "Right"
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    subsequence_length = np.random.randint(args.max_subsequence_length)    
    Y_train[graph_id] = (subsequence_length >= args.max_subsequence_length - 1)
    node_id = np.random.randint(subsequence_length + args.distance_from_edge, args.max_sequence_length - 1 - args.distance_from_edge)
    for node_pos in range(subsequence_length + 1):
        graphs_train.add_graph_node_property(graph_id, node_id - node_pos, 'A')

    if np.random.rand() <= args.noise:
        Y_train[graph_id] = 1 - Y_train[graph_id]

graphs_train.encode()

# Create test data

print("Creating testing data")

graphs_test = Graphs(args.number_of_examples, init_with=graphs_train)

for graph_id in range(args.number_of_examples):
    graphs_test.set_number_of_graph_nodes(graph_id, args.max_sequence_length)

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
            edge_type = "Left"
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

        if node_id < graphs_test.number_of_graph_nodes[graph_id]-1:
            destination_node_id = node_id + 1
            edge_type = "Right"
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type)

    subsequence_length = np.random.randint(args.max_subsequence_length)    
    Y_test[graph_id] = (subsequence_length >= args.max_subsequence_length - 1)
    node_id = np.random.randint(subsequence_length + args.distance_from_edge, args.max_sequence_length - 1 - args.distance_from_edge)
    for node_pos in range(subsequence_length + 1):
        graphs_test.add_graph_node_property(graph_id, node_id - node_pos, 'A')

graphs_test.encode()

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth = args.depth,
    message_size = args.message_size,
    message_bits = args.message_bits,
    max_included_literals = args.max_included_literals,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding,
    q = args.q
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    y_test_pred = tm.predict(graphs_test)
    result_test = 100*(y_test_pred == Y_test).mean()
    stop_testing = time()

    print("Test\n", confusion_matrix(Y_test, y_test_pred))

    y_train_pred = tm.predict(graphs_train)
    result_train = 100*(y_train_pred == Y_train).mean()

    print("Train\n", confusion_matrix(Y_train, y_train_pred))

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

    number_of_classes = np.unique(Y_train).shape[0]

    weights = tm.get_state()[1].reshape(number_of_classes, tm.number_of_clauses)

    ta_states = {}
    ta_states[0] = tm.get_ta_states(0)
    for depth in range(1, args.depth):
        ta_states[depth] = tm.get_ta_states(depth)

    for i in range(tm.number_of_clauses):
            print("Clause #%d W:" % (i), weights[:,i], end=' ')
            l = []
            # for k in range(graphs_train.hypervector_size * 2):
            #     if tm.ta_action(0, i, k):
            #         if k < graphs_train.hypervector_size:
            #             l.append("x%d" % (k))
            #         else:
            #             l.append("NOT x%d" % (k - graphs_train.hypervector_size))

            if tm.ta_action(0, i, 0):
                l.append("A (%d)" % (ta_states[0][i, 0]))
            elif tm.ta_action(0, i, 1):
                l.append("NOT A (%d)" % (ta_states[0][i, 1]))

            for depth in range(1, args.depth):
                for k in range(tm.message_size * 2):
                    if k % 2 == 1:
                        if tm.ta_action(depth, i, k):
                            if k < tm.message_size:
                                l.append("l%d:%d (%d)" % (depth, k // 2, ta_states[depth][i, k]))
                            else:
                                l.append("NOT l%d:%d (%d)" % (depth, (k - tm.message_size) // 2, ta_states[depth][i, k]))
                    else:
                        if tm.ta_action(depth, i, k):
                            if k < tm.message_size:
                                l.append("r%d:%d (%d)" % (depth, k // 2, ta_states[depth][i, k]))
                            else:
                                l.append("NOT r%d:%d (%d)" % (depth, (k - tm.message_size) // 2, ta_states[depth][i, k]))

            print(" AND ".join(l))

print(graphs_test.hypervectors)
print(tm.hypervectors)
print(graphs_test.edge_type_id)