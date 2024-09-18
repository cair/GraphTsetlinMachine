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
	def __init__(self, number_of_nodes, node_names=None, init_with=None, symbols=None, hypervector_size=128, hypervector_bits=2):
		self.node_names = node_names
		self.number_of_nodes = number_of_nodes
		self.node_index = np.zeros(self.number_of_nodes.shape[0], dtype=np.uint32)
		self.node_index[1:] = np.add.accumulate(self.number_of_nodes[:-1])
		self.max_number_of_nodes = self.number_of_nodes.max()

		self.node_id = {}
		if self.node_names == None:
			for i in range(self.max_number_of_nodes):
				self.node_id[i] = len(self.node_id)
		else:
			for node_name in self.node_names:
				self.node_id[node_name] = len(self.node_id)

		self.number_of_edges = np.zeros(self.number_of_nodes.sum(), dtype=np.uint32)

		if init_with == None:
			self.number_of_symbols = len(symbols)
			self.symbol_id = {}
			for symbol in symbols:
				self.symbol_id[symbol] = len(self.symbol_id)

			self.hypervector_size = hypervector_size
			self.hypervector_bits = hypervector_bits

			indexes = np.arange(self.hypervector_size, dtype=np.uint32)
			self.hypervectors = np.zeros((self.number_of_symbols, self.hypervector_bits), dtype=np.uint32)
			for i in range(self.number_of_symbols):
				self.hypervectors[i,:] = np.random.choice(indexes, size=(self.hypervector_bits), replace=False)
		else:
			self.number_of_symbols = init_with.number_of_symbols
			self.symbol_id = init_with.symbol_id
			self.hypervectors = init_with.hypervectors
			self.hypervector_size = init_with.hypervector_size
			self.hypervector_bits = init_with.hypervector_bits

		self.number_of_hypervector_chunks = (self.hypervector_size*2 - 1) // 32 + 1

		self.X = np.zeros((self.number_of_nodes.sum(), self.number_of_hypervector_chunks), dtype=np.uint32)
		for k in range(self.hypervector_size, self.hypervector_size*2):
			chunk = k // 32
			pos = k % 32
			self.X[:,chunk] |= (1 << pos)

	def set_number_edges(self, graph, node, number_of_edges):
		self.number_of_edges[self.node_index[graph] + node] = number_of_edges

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

	def encode(self):
		m = hashlib.sha256()
		m.update(self.X.data)
		self.signature = m.digest()

		#self.X = self.X.reshape(-1)

		self.encoded = True

		return