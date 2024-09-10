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

class Graph():
	def __init__(self):
		self.node_name_id = {}
		self.node_id_name = {}
		self.node_edges = {}
		self.node_features = {}
		self.edge_type_id = {}		
		self.node_counter = 0
		self.edge_counter = 0
		self.edge_type_counter = 0

	def add_node(self, node_name):
		self.node_name_id[node_name] = self.node_counter
		self.node_id_name[self.node_counter] = node_name
		self.node_features[node_name] = {}
		self.node_edges[node_name] = {}
		self.node_counter += 1

	def add_edge(self, node_name_1, node_name_2, edge_type='p'):
		if edge_type not in self.edge_type_id:
			self.edge_type_id[edge_type] = self.edge_type_counter
			self.edge_type_counter += 1

		self.node_edges[node_name_1][(node_name_2, edge_type)] = 1
		self.edge_counter += 1

	def add_node_feature(self, node_name, symbols):
		if type(symbols) != tuple:
			symbols = (symbols,)

		self.node_features[node_name][symbols] = 1

	def encode(self, hypervectors, hypervector_size=1024, hypervector_bits=3):
		# Vector for encoding node features
		Nf = np.zeros((len(self.node_name_id), hypervector_size), dtype=np.uint32)

		# Vector for encoding node edges
		Ne = np.zeros(len(self.node_name_id) + self.edge_counter*2, dtype=np.uint32)

		position = 0
		for i in range(self.node_counter):
			# Encodes node features
			for symbols in self.node_features[self.node_id_name[i]]:
				for symbol in symbols:
					if symbol not in hypervectors:
						indexes = np.arange(hypervector_size, dtype=np.uint32)
						hypervectors[symbol] = np.random.choice(indexes, size=(hypervector_bits), replace=False)
						print(symbol, hypervectors[symbol])

				base_indexes = hypervectors[symbols[0]]
				for j in range(1, len(symbols)):
					base_indexes = (base_indexes + (hypervectors[symbols[j]][0]+2)*j) % hypervector_size
				Nf[i][base_indexes] = 1

			# Encodes node edges
			Ne[position] = len(self.node_edges[self.node_id_name[i]])
			position += 1
			for edge in self.node_edges[self.node_id_name[i]]:
				Ne[position] = self.node_name_id[edge[0]]
				Ne[position+1] = self.edge_type_id[edge[1]]
				position += 2

		return (Nf, Ne)