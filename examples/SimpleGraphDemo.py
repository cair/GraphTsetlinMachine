from GraphTsetlinMachine.graph import Graph

sequence_graph = Graph(256, 2)

sequence_graph.add_node('A')
sequence_graph.add_node('B')
sequence_graph.add_node('C')
sequence_graph.add_edge('A', 'B')
sequence_graph.add_edge('B', 'C')

print(sequence_graph.node_id)