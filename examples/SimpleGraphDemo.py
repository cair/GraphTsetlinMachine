from GraphTsetlinMachine.graph import Graph

sequence_graph = Graph()

sequence_graph.add_node('A')
sequence_graph.add_node('B')
sequence_graph.add_node('C')
sequence_graph.add_edge('A', 'B')
sequence_graph.add_edge('B', 'C')

sequence_graph.add_node_feature('A', (0, 1))
sequence_graph.add_node_feature('B', (0, 1))
sequence_graph.add_node_feature('C', (0, 1))

print(sequence_graph.node_name)

print(sequence_graph.node_edges)