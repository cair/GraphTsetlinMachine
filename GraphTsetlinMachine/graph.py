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
		self.node_name = {}
		self.node_edges = {}
		self.node_features = {}

	def add_node(self, node_name):
		self.node_name[node_name] = 1
		self.node_features[node_name] = {}
		self.node_edges[node_name] = {}

	def add_edge(self, node_name_1, node_name_2, edge_type='p'):
		self.node_edges[node_name_1][(node_name_2, edge_type)] = 1

	def add_node_feature(self, node_name, symbols):
		if type(symbols) != tuple:
			symbols = (symbols,)

		self.node_features[node_name][symbols] = 1

	def encode(self, hypervectors, hypervector_size=1024, hypervector_bits=3):
		# Vector for encoding graph
		Xi = np.zeros((len(self.node_name), hypervector_size), dtype=np.uint32)

		position = 0
		for n in self.node_name.keys():
			for symbols in self.node_features[n].keys():
				for symbol in symbols:
					if symbol not in hypervectors:
						indexes = np.arange(hypervector_size, dtype=np.uint32)
						hypervectors[symbol] = np.random.choice(indexes, size=(hypervector_bits), replace=False)
						print(symbol, hypervectors[symbol])

				base_indexes = hypervectors[symbols[0]]
				for symbol in symbols[1:]:
					base_indexes = (base_indexes + hypervectors[symbol][0]) % hypervector_size
				Xi[position][base_indexes] = 1
			position += 1