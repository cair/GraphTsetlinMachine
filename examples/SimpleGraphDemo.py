from GraphTsetlinMachine.graph import Graph
import numpy as np
from scipy.sparse import csr_matrix
import GraphTsetlinMachine.graph as graph

number_of_training_examples = 10

max_sequence_length = 5

number_of_classes = 2 # Must be less than or equal to max sequence length

training_examples = []


Y = np.empty(number_of_training_examples, dtype=np.uint32)
X = []

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
            sequence_graph.add_edge(j, j-1, edge_type=0)

        if j < sequence_length-1:
            sequence_graph.add_edge(j, j+1, edge_type=1)

    # Select class
    Y[i] = np.random.randint(number_of_classes) 

    print("Target", Y[i])

    # Add features
    if sequence_length-Y[i]-1 == 0:
        position = 0
    else:
        position = np.random.randint(sequence_length-Y[i]-1)

    print("Position", position)

    for p in range(position, position + Y[i] + 1):
        sequence_graph.add_feature(p, 'A')
        sequence_graph.add_feature(p, ('A','B'))
        sequence_graph.add_feature(p, ('A','B', 'C'))

    X.append(sequence_graph)

(data, edges) = graph.encode(X, hypervector_size=16, hypervector_bits=1)

print(data)
print()
print(edges)