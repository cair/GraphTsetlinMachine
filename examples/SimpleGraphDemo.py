from GraphTsetlinMachine.graph import Graph

number_of_training_examples = 10000

max_sequence_length = 10

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
    for j in range(max_sequence_length):
        sequence_graph.add_node_feature(j, (0, 1))

    print(sequence_graph.node_name)

    print(sequence_graph.node_edges)