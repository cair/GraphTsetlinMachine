from GraphTsetlinMachine.graph import Graph
from GraphTsetlinMachine.graph import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

number_of_training_examples = 10

max_sequence_length = 5

number_of_classes = 2 # Must be less than or equal to max sequence length

graphs_train = Graphs()
Y_train = np.empty(number_of_training_examples, dtype=np.uint32)

for i in range(number_of_training_examples):
    # Create graph

    sequence_length = np.random.randint(number_of_classes, max_sequence_length+1)
    print("Length", sequence_length)

    sequence_graph = Graph()

    # Create nodes
    for j in range(sequence_length):
        sequence_graph.add_node(j)

    # Add edges in both directions
    for j in range(sequence_length):
        if j > 0:
            sequence_graph.add_edge(j, j-1, edge_type='left')

        if j < sequence_length-1:
            sequence_graph.add_edge(j, j+1, edge_type='right')

    # Select class
    Y_train[i] = np.random.randint(number_of_classes) 

    print("Target", Y_train[i])

    # Add features
    if sequence_length-Y[i]-1 == 0:
        position = 0
    else:
        position = np.random.randint(sequence_length-Y[i]-1)

    print("Position", position)

    for p in range(position, position + Y[i] + 1):
        sequence_graph.add_feature(p, 'A')

    graphs_train.add(sequence_graph)

print(graphs_train.X)
print()
print(graphs_train.edges)
print(graphs_train.hypervectors)
print(graphs_train.edge_type_id)
print(graphs_train.node_count)

tm = MultiClassGraphTsetlinMachine(100, 1000, 1.0, hypervector_size=16, depth=1)

for i in range(epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_train) == Y_train).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))