from GraphTsetlinMachine.graph import Graph
import numpy as np

number_of_training_examples = 10000

max_sequence_length = 10

max_number_of_classes = 3 # Must be less than or equal to max sequence length

training_examples = []

for i in range(number_of_training_examples):
    # Create graph
    sequence_graph = Graph()

    # Create nodes
    for j in range(max_sequence_length):
        sequence_graph.add_node(j)
    
    # Add edges in both directions
    for j in range(max_sequence_length):
        if j > 0:
            sequence_graph.add_edge(j, j-1)

        if j < max_sequence_length-1:
            sequence_graph.add_edge(j, j+1)

    # Add features
    target = np.random.randint(max_number_of_classes) 
    position = np.random.randint(max_sequence_length-target-1)
    print(target, position, max_sequence_length)
    for p in range(position, position+target+1):
        sequence_graph.add_node_feature(p, 'A')

    print(sequence_graph.node_name)

    print(sequence_graph.node_edges)

    print(sequence_graph.node_features)