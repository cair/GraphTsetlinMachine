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

class Graph():
	def __init__(self, hypervector_size, hypervector_bits):
		self.hypervector_size = hypervector_size
		self.hypervector_bits = hypervector_bits

		self.symbol_id = {}
		self.node_id = {}
		self.node_edges = {}
		self.node_features = {}
		self.edge_type_id = {}

		self.node_counter = 0
		self.symbol_counter = 0
		self.edge_type_counter = 0

	def add_node(self, node_name):
		if node_name not in self.node_id:
			self.node_id[node_name] = self.node_counter
			self.node_edges[self.node_counter] = {}
			self.node_features[self.node_counter] = {}
			self.node_counter += 1

	def add_edge(self, node_name_1, node_name_2, edge_type='plain'):
		if edge_type not in self.edge_type_id:
			self.edge_type_id[edge_type] = self.edge_type_counter
			self.edge_type_counter += 1

		self.node_edges[self.node_id[node_name_1]][(self.node_id[node_name_2], self.edge_type_id[edge_type])] = True

	def add_feature(self, symbols):
		if feature_name not in self.all_features:
			self.feature_id[feature_name] = self.feature_counter
			self.feature_counter += 1

	# Tha input variable 'symbols' is a tuple of symbols or a single symbols
	def add_node_feature(self, node_name, symbols):
		if type(symbols) == tuple:
			for symbol in symbols:
				if not symbol in self.symbol_id:
					self.symbol_id[symbol] = self.symbol_counter
					self.symbol_counter += 1

		self.node_features[self.node_id[node_name]][self.feature_id[feature_name]] = True