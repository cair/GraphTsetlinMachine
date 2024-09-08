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
	def __init__(self):
		self.node_name = {}
		self.node_edges = {}
		self.node_features = {}

	def add_node(self, node_name):
		self.node_name[node_name] = 1
		self.node_features[node_name] = {}

	def add_edge(self, node_name_1, node_name_2, edge_type='plain'):
		if node_name_1 not in self.node_edges:
			self.node_edges[node_name_1] = {}

		self.node_edges[node_name_1][(node_name_2, edge_type)] = 1

	def add_node_feature(self, node_name, symbols):
		self.node_features[node_name][symbols] = 1