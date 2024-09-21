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
import hashlib
from numba import jit

class Graphs():
	def __init__(self, number_of_graphs, hypervector_size = 128, hypervector_bits = 2, init_with=None):
		self.number_of_graphs = number_of_graphs
		self.number_of_graph_nodes = np.zeros(self.number_of_graphs, dtype=np.uint32)

		self.hypervector_size = hypervector_size
		self.hypervector_bits = hypervector_bits
		self.number_of_hypervector_chunks = (self.hypervector_size*2 - 1) // 32 + 1
		
		self.graph_node_id = [None] * self.number_of_graphs
		for i in range(number_of_graphs):
			self.graph_node_id[i] = {}

		self.graph_node_feature = {}
		self.graph_node_edge = {}

		self.init_with = init_with
		if self.init_with == None:
			self.symbol_id = {}
			self.edge_id = {}
		else:
			self.symbol_id = self.init_with.symbol_id
			self.edge_id = self.init_with.edge_id

	def set_number_of_graph_nodes(self, graph_id, number_of_graph_nodes):
		self.number_of_graph_nodes[graph_id] = number_of_graph_nodes

	@staticmethod
	@jit(nopython=True)
	def _initialize_node_hypervectors(hypervector_size, X):
		for i in range(X.shape[0]):
			for k in range(hypervector_size, hypervector_size*2):
				chunk = k // 32
				pos = k % 32
				X[i,chunk] |= (1 << pos)

	def prepare_node_configuration(self):
		self.node_index = np.zeros(self.number_of_graph_nodes.shape[0], dtype=np.uint32)
		self.node_index[1:] = np.add.accumulate(self.number_of_graph_nodes[:-1])

		self.number_of_graph_node_edges = np.empty(self.number_of_graph_nodes.sum(), dtype=np.uint32)
		self.graph_node_edge_counter = np.zeros(self.number_of_graph_nodes.sum(), dtype=np.uint32)
		self.edge_index = np.zeros(self.number_of_graph_nodes.sum(), dtype=np.uint32)

		self.X = np.zeros((self.number_of_graph_nodes.sum(), self.number_of_hypervector_chunks), dtype=np.uint32)
		self._initialize_node_hypervectors(self.hypervector_size, self.X)

	def add_graph_node(self, graph_id, node_name, number_of_graph_node_edges):
		if node_name not in self.graph_node_id[graph_id]:
			self.graph_node_id[graph_id][node_name] = len(self.graph_node_id[graph_id])
		self.number_of_graph_node_edges[self.node_index[graph_id] + self.graph_node_id[graph_id][node_name]] = number_of_graph_node_edges

	def prepare_edge_configuration(self):		
		self.edge_index[1:] = np.add.accumulate(self.number_of_graph_node_edges[:-1])
		self.graph_node_edge = np.empty((self.number_of_graph_node_edges.sum(), 2), dtype=np.uint32)

	def add_graph_node_edge(self, graph_id, source_node_name, destination_node_name, edge_type):
		source_node_id = self.graph_node_id[graph_id][source_node_name]
		destination_node_id = self.graph_node_id[graph_id][destination_node_name]
		edge_index = self.edge_index[self.node_index[graph_id] + source_node_id] + self.graph_node_edge_counter[self.node_index[graph_id] + source_node_id]
		self.graph_node_edge[edge_index][0] = destination_node_id
		self.graph_node_edge[edge_index][1] = edge_type
		self.graph_node_edge_counter[self.node_index[graph_id] + source_node_id] += 1

	############

	def add_graph(self, graph_name):
		if graph_name not in self.graph_id:
			self.graph_id[graph_name] = len(self.graph_id)
			self.graph_node_id[self.graph_id[graph_name]] = {}
			self.graph_node_feature[self.graph_id[graph_name]] = {}
			self.graph_node_edge[self.graph_id[graph_name]] = {}

	def add_graph_node_old(self, graph_name, node_name):
		if node_name not in self.graph_node_id[self.graph_id[graph_name]]:
			self.graph_node_id[self.graph_id[graph_name]][node_name] = len(self.graph_node_id[self.graph_id[graph_name]])
			self.graph_node_feature[self.graph_id[graph_name]][self.graph_node_id[self.graph_id[graph_name]][node_name]] = {}
			self.graph_node_edge[self.graph_id[graph_name]][self.graph_node_id[self.graph_id[graph_name]][node_name]] = {}

	def add_graph_node_feature(self, graph_name, node_name, symbol_name):
		if symbol_name not in self.symbol_id:
			self.symbol_id[symbol_name] = len(self.symbol_id) 
		self.graph_node_feature[self.graph_id[graph_name]][self.graph_node_id[self.graph_id[graph_name]][node_name]][self.symbol_id[symbol_name]] = 1

	def add_graph_node_edge_old(self, graph_name, start_name, edge_name, end_name):
		if edge_name not in self.edge_id:
			self.edge_id[edge_name] = len(self.edge_id) 
		self.graph_node_edge[self.graph_id[graph_name]][self.graph_node_id[self.graph_id[graph_name]][start_name]][self.graph_node_id[self.graph_id[graph_name]][end_name]] = self.edge_id[edge_name]

	@staticmethod
	@jit(nopython=True)
	def _add_node_feature(hypervectors, hypervector_size, graph_index, node, symbol, X):
		for k in hypervectors[symbol,:]:
			chunk = k // 32
			pos = k % 32

			X[graph_index + node, chunk] |= (1 << pos)

			chunk = (k + hypervector_size) // 32
			pos = (k + hypervector_size)  % 32
			X[graph_index + node, chunk] &= ~(1 << pos)

	def add_node_feature(self, graph, node_name, symbol):
		self._add_node_feature(self.hypervectors, self.hypervector_size, self.node_index[graph], self.node_id[node_name], self.symbol_id[symbol], self.X)

	def encode(self, hypervector_size=128, hypervector_bits=2):
		# Create one hypervector per symbol

		self.number_of_symbols = len(self.symbol_id)

		if self.init_with == None:
			self.hypervector_size = hypervector_size
			self.hypervector_bits = hypervector_bits

			indexes = np.arange(self.hypervector_size, dtype=np.uint32)
			self.hypervectors = np.zeros((self.number_of_symbols, self.hypervector_bits), dtype=np.uint32)
			for i in range(self.number_of_symbols):
				self.hypervectors[i,:] = np.random.choice(indexes, size=(self.hypervector_bits), replace=False)
		else:
			self.hypervectors = self.init_with.hypervectors
			self.hypervector_size = self.init_with.hypervector_size
			self.hypervector_bits = self.init_with.hypervector_bits

		self.number_of_hypervector_chunks = (self.hypervector_size*2 - 1) // 32 + 1

		# Encode graphs and nodes

		self.number_of_graph_nodes = np.empty(len(self.graph_id), dtype=np.uint32)
		for graph_id in range(self.number_of_graph_nodes.shape[0]):
			self.number_of_graph_nodes[graph_id] = len(self.graph_node_id[graph_id])

		self.node_index = np.zeros(self.number_of_graph_nodes.shape[0], dtype=np.uint32)
		self.node_index[1:] = np.add.accumulate(self.number_of_graph_nodes[:-1])

		# Initialize node hypervectors

		self.X = np.zeros((self.number_of_graph_nodes.sum(), self.number_of_hypervector_chunks), dtype=np.uint32)
		for k in range(self.hypervector_size, self.hypervector_size*2):
			chunk = k // 32
			pos = k % 32
			self.X[:,chunk] |= (1 << pos)

		for graph_id in range(self.number_of_graph_nodes.shape[0]):
			for graph_node_id in range(self.number_of_graph_nodes[graph_id]):
				for symbol_id in self.graph_node_feature[graph_id][graph_node_id]:
					self._add_node_feature(self.hypervectors, self.hypervector_size, self.node_index[graph_id], graph_node_id, symbol_id, self.X)

		# Encode edges

		self.number_of_edges = np.empty(self.number_of_graph_nodes.sum(), dtype=np.uint32)
		for graph_id in range(self.number_of_graph_nodes.shape[0]):
			for graph_node_id in range(self.number_of_graph_nodes[graph_id]):
				self.number_of_edges[self.node_index[graph_id] + graph_node_id] = len(self.graph_node_edge[graph_id][graph_node_id])
		
		self.edge_index = np.zeros(self.number_of_graph_nodes.sum(), dtype=np.uint32)
		self.edge_index[1:] = np.add.accumulate(self.number_of_edges[:-1])

		self.edge_target = np.empty(self.number_of_edges.sum(), dtype=np.uint32)
		self.edge_type = np.empty(self.number_of_edges.sum(), dtype=np.uint32)
		for graph_id in range(self.number_of_graph_nodes.shape[0]):
			for graph_node_id in range(self.number_of_graph_nodes[graph_id]):
				edges = list(self.graph_node_edge[graph_id][graph_node_id].items())
				for edge_id in range(self.number_of_edges[self.node_index[graph_id] + graph_node_id]):
					self.edge_target[self.edge_index[self.node_index[graph_id] + graph_node_id] + edge_id] = edges[edge_id][0]
					self.edge_type[self.edge_index[self.node_index[graph_id] + graph_node_id] + edge_id] = edges[edge_id][1]

		m = hashlib.sha256()
		m.update(self.X.data)
		m.update(self.edge_target.data)
		m.update(self.edge_type.data)
		self.signature = m.digest()

		self.encoded = True

		return