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
import sys

class Graphs():
	def __init__(
		self,
		number_of_graphs,
		hypervector_size = 128,
		hypervector_bits = 2,
		double_hashing=False,
		one_hot_encoding=False,
		symbols=None,
		init_with=None
	):
		self.number_of_graphs = number_of_graphs
		self.number_of_graph_nodes = np.zeros(self.number_of_graphs, dtype=np.uint32)
	
		self.double_hashing = double_hashing
		self.one_hot_encoding = one_hot_encoding

		self.graph_node_id = [None] * self.number_of_graphs
		for i in range(number_of_graphs):
			self.graph_node_id[i] = {}

		self.init_with = init_with
		if self.init_with == None:
			self.edge_type_id = {}
			self.node_type_id = {}

			self.symbol_id = {}
			for symbol_name in symbols:
				self.symbol_id[symbol_name] = len(self.symbol_id)
			self.hypervector_size = hypervector_size
			self.hypervector_bits = hypervector_bits

			if self.one_hot_encoding:
				self.hypervector_size = len(self.symbol_id)
				self.hypervector_bits = 1
				self.hypervectors = np.zeros((len(self.symbol_id), self.hypervector_bits), dtype=np.uint32)
				for i in range(len(self.symbol_id)):
					self.hypervectors[i, 0] = i
			elif self.double_hashing:
				from sympy import prevprime
				self.hypervector_bits = 2
				self.hypervectors = np.zeros((len(self.symbol_id), self.hypervector_bits), dtype=np.uint32)
				prime = prevprime(self.hypervector_size)
				for i in range(len(self.symbol_id)):
					self.hypervectors[i, 0] = i % (self.hypervector_size)
					self.hypervectors[i, 1] = prime - (i % prime)
			else:
				self.hypervectors = np.zeros((len(self.symbol_id), self.hypervector_bits), dtype=np.uint32)
				indexes = np.arange(self.hypervector_size)
				for i in range(len(self.symbol_id)):
					self.hypervectors[i,:] = np.random.choice(indexes, size=(self.hypervector_bits), replace=False)

			self.number_of_hypervector_chunks = (self.hypervector_size*2 - 1) // 32 + 1
		else:
			self.edge_type_id = self.init_with.edge_type_id
			self.node_type_id = self.init_with.node_type_id
			self.symbol_id = self.init_with.symbol_id
			self.hypervector_size = self.init_with.hypervector_size
			self.hypervector_bits = self.init_with.hypervector_bits
			self.hypervectors = self.init_with.hypervectors
			self.number_of_hypervector_chunks = self.init_with.number_of_hypervector_chunks

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

		self.max_number_of_graph_nodes = self.number_of_graph_nodes.max()
		self.max_number_of_graph_node_chunks = (self.max_number_of_graph_nodes - 1) // 32 + 1
		self.number_of_nodes = self.number_of_graph_nodes.sum()
		self.node_type = np.empty(self.number_of_nodes, dtype=np.uint32)
		self.number_of_graph_node_edges = np.empty(self.number_of_nodes, dtype=np.uint32)
		self.graph_node_edge_counter = np.zeros(self.number_of_nodes, dtype=np.uint32)
		self.edge_index = np.zeros(self.number_of_nodes, dtype=np.uint32)

		self.X = np.zeros((self.number_of_nodes, self.number_of_hypervector_chunks), dtype=np.uint32)
		self._initialize_node_hypervectors(self.hypervector_size, self.X)

	def add_graph_node(self, graph_id, node_name, number_of_graph_node_edges, node_type_name='Plain'):
		if node_type_name not in self.node_type_id:
			self.node_type_id[node_type_name] = len(self.node_type_id)

		if node_name not in self.graph_node_id[graph_id]:
			self.graph_node_id[graph_id][node_name] = len(self.graph_node_id[graph_id])
		self.node_type[self.node_index[graph_id] + self.graph_node_id[graph_id][node_name]] = self.node_type_id[node_type_name]
		self.number_of_graph_node_edges[self.node_index[graph_id] + self.graph_node_id[graph_id][node_name]] = number_of_graph_node_edges

	def number_of_node_types(self):
		return len(self.node_type_id)

	def prepare_edge_configuration(self):		
		self.edge_index[1:] = np.add.accumulate(self.number_of_graph_node_edges[:-1])
		self.edge = np.empty((self.number_of_graph_node_edges.sum(), 2), dtype=np.uint32)

	def add_graph_node_edge(self, graph_id, source_node_name, destination_node_name, edge_type_name):
		source_node_id = self.graph_node_id[graph_id][source_node_name]

		if self.graph_node_edge_counter[self.node_index[graph_id] + source_node_id] >= self.number_of_graph_node_edges[self.node_index[graph_id] + source_node_id]:
			print("Too many edges added to node '%s' of graph %d." % (source_node_name, graph_id))
			sys.exit(-1)

		destination_node_id = self.graph_node_id[graph_id][destination_node_name]
		if edge_type_name not in self.edge_type_id:
			self.edge_type_id[edge_type_name] = len(self.edge_type_id)
		edge_type_id = self.edge_type_id[edge_type_name]

		edge_index = self.edge_index[self.node_index[graph_id] + source_node_id] + self.graph_node_edge_counter[self.node_index[graph_id] + source_node_id]
		self.edge[edge_index][0] = destination_node_id
		self.edge[edge_index][1] = edge_type_id
		self.graph_node_edge_counter[self.node_index[graph_id] + source_node_id] += 1

	@staticmethod
	@jit(nopython=True)
	def _add_graph_node_property(hypervectors, hypervector_size, graph_index, node, symbol, X):
		for k in hypervectors[symbol,:]:
			chunk = k // 32
			pos = k % 32

			X[graph_index + node, chunk] |= (1 << pos)

			chunk = (k + hypervector_size) // 32
			pos = (k + hypervector_size)  % 32
			X[graph_index + node, chunk] &= ~(1 << pos)

	def add_graph_node_property(self, graph_id, node_name, symbol):
		self._add_graph_node_property(self.hypervectors, self.hypervector_size, self.node_index[graph_id], self.graph_node_id[graph_id][node_name], self.symbol_id[symbol], self.X)

	def print_graph_nodes(self, graph_id):
		graphstr ='Printing nodes of Graph#'+str(graph_id)+':\n'
		for node_id in range(self.number_of_graph_nodes[graph_id]):
			nodestr='Node#'+str(node_id)+'( '
			for (symbol_name, symbol_id) in self.symbol_id.items():
				match = True
				for k in self.hypervectors[symbol_id,:]:
					chunk = k // 32
					pos = k % 32

					if (self.X[self.node_index[graph_id] + node_id][chunk] & (1 << pos)) == 0:
						match = False

				if match:
					nodestr+= symbol_name+' '
				else:
					nodestr+= '*'+' '
			nodestr+= ')'+' '
			graphstr+= nodestr
		print(graphstr)
		print()

	def print_graph_edges(self, graph_id):
		graphstr ='Printing edges of Graph#'+str(graph_id)+':\n'
		for node_id in range(0,self.number_of_graph_nodes[graph_id]):
			for node_edge_num in range(0, self.graph_node_edge_counter[self.node_index[graph_id] + node_id]):
				edge_index = self.edge_index[self.node_index[graph_id] + node_id] + node_edge_num
				edgestr='Edge#'+str(edge_index)
			
				edgestr+= ' SrcNode#'+str(node_id)

				edgestr+= ' DestNode#'+str(self.edge[edge_index][0])

				edgestr+= ' EdgeType#'+str(self.edge[edge_index][1])+'\n'
			
				graphstr+= edgestr
			graphstr+= '\n'
		print(graphstr)
		print()

	def print_graph(self, graph_id):
		self.print_graph_nodes(graph_id)
		self.print_graph_edges(graph_id)


	def encode(self):
		edges_missing = False
		for graph_id in range(self.number_of_graphs):
			for (node_name, node_id) in self.graph_node_id[graph_id].items():
				if self.graph_node_edge_counter[self.node_index[graph_id] + node_id] < self.number_of_graph_node_edges[self.node_index[graph_id] + node_id]:
					edges_missing = True
					print("Node '%s' of graph %d misses edges." % (node_name, graph_id))

		if edges_missing:
			sys.exit(-1)

		m = hashlib.sha256()
		m.update(self.X.data)
		m.update(self.edge.data)
		self.signature = m.digest()

		self.encoded = True
