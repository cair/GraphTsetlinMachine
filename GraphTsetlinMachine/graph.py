# Copyright (c) 2024 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.sparse import coo_matrix

class Graph():
	def __init__(self):
		self.node_name_id = {}
		self.node_id_name = {}
		self.node_edges = {}
		self.node_features = {}
		self.edge_counter = 0
		self.feature_counter = 0

	def add_node(self, node_name):
		self.node_name_id[node_name] = len(self.node_name_id)
		self.node_id_name[self.node_name_id[node_name]] = node_name
		self.node_features[node_name] = []
		self.node_edges[node_name] = []

	def add_edge(self, node_name_1, node_name_2, edge_type=0):
		self.node_edges[node_name_1].append((node_name_2, edge_type))
		self.edge_counter += 1

	def add_feature(self, node_name, symbols):
		if type(symbols) == tuple:
			symbols = " ".join(symbols)
		self.node_features[node_name].append(symbols)
		self.feature_counter += 1

def encode(graphs, hypervector_size=1024, hypervector_bits=3):
	global_feature_counter = 0
	global_edge_counter = 0
	for graph in graphs:
		global_feature_counter += graph.feature_counter*hypervector_bits
		global_edge_counter += len(graph.node_name_id) + graph.edge_counter*2

	hypervectors = {}
	indexes = np.arange(hypervector_size, dtype=np.uint32)

	edge_type_id = {}

	feature_row = np.empty(global_feature_counter, dtype=np.uint32)
	feature_col = np.empty(global_feature_counter, dtype=np.uint32)
	feature_data = np.empty(global_feature_counter, dtype=np.uint32)

	edge_row = np.empty(global_edge_counter, dtype=np.uint32)
	edge_col = np.empty(global_edge_counter, dtype=np.uint32)
	edge_data = np.empty(global_edge_counter, dtype=np.uint32)

	feature_position = 0
	edge_position = 0
	for i in range(len(graphs)):
		graph = graphs[i]
		local_edge_position = 0
		for j in range(len(graph.node_name_id)):
			node_name = graph.node_id_name[j]

			for symbols in graph.node_features[node_name]:
				if symbols not in hypervectors:
					hypervectors[symbols] = np.random.choice(indexes, size=(hypervector_bits), replace=False)
			
				for k in range(hypervector_bits):
					feature_row[feature_position] = i
					feature_col[feature_position] = hypervectors[symbols][k] + j*hypervector_size
					feature_data[feature_position] = 1
					feature_position += 1

			edge_row[edge_position] = i
			edge_col[edge_position] = local_edge_position
			edge_data[edge_position] = len(graph.node_edges[node_name])
			edge_position += 1
			local_edge_position += 1
			for edge in graph.node_edges[graph.node_id_name[j]]:
				edge_row[edge_position] = i
				edge_col[edge_position] = local_edge_position
				edge_data[edge_position] = graph.node_name_id[edge[0]]
				edge_position += 1
				local_edge_position += 1

				edge_row[edge_position] = i
				edge_col[edge_position] = local_edge_position

				if edge[1] not in edge_type_id:
					edge_type_id[edge[1]] = len(edge_type_id)
				edge_data[edge_position] = edge_type_id[edge[1]]
				edge_position += 1
				local_edge_position += 1

	return((coo_matrix((feature_data, (feature_row, feature_col))).tocsr(), coo_matrix((edge_data, (edge_row, edge_col))).tocsr(), hypervectors, edge_type_id))
