from GraphTsetlinMachine.graph import Graph
from GraphTsetlinMachine.graph import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time

epochs = 25

number_of_training_examples = 1000

max_sequence_length = 1

number_of_classes = 2 # Must be less than or equal to max sequence length

graphs_train = Graphs()
Y_train = np.empty(number_of_training_examples, dtype=np.uint32)

for i in range(number_of_training_examples):
    sequence_graph = Graph()
    
    # Select class
    Y_train[i] = np.random.randint(number_of_classes) 

    nodes = 2
    for j in range(nodes):
        sequence_graph.add_node(j)

    j = np.random.randint(nodes)

    if Y_train[i] == 0:
        sequence_graph.add_feature(j, 'A')
    else:
        sequence_graph.add_feature(j, 'B')

    graphs_train.add(sequence_graph)

graphs_train.encode(hypervector_size=16, hypervector_bits=1)

print(graphs_train.hypervectors)
print(graphs_train.edge_type_id)
print(graphs_train.node_count)

tm = MultiClassGraphTsetlinMachine(2, 100, 1.0, hypervector_size=16, hypervector_bits=1, depth=1)

for i in range(epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_train) == Y_train).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

#print(graphs_train.X)
#print()
#print(graphs_train.edges)
print(graphs_train.hypervectors)
print(graphs_train.edge_type_id)
print(graphs_train.node_count)