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

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np

import GraphTsetlinMachine.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix
import sys
from sympy import prevprime

from time import time

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			depth=1,
			message_size=256,
			message_bits=2,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		print("Initialization of sparse structure.")

		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)//32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.q = q
		self.max_included_literals = max_included_literals
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.depth = depth
		self.message_size = message_size
		self.message_bits = message_bits
		self.message_prime = prevprime(message_size//3)
		self.message_literals = message_size*2
		self.grid = grid
		self.block = block

		self.graphs_signature_train = np.array([])
		self.graphs_signature_test = np.array([])
		self.encoded_Y = np.array([])
		
		self.ta_state = np.array([])
		self.message_ta_state = np.array([])
		self.clause_weights = np.array([])

		indexes = np.arange(self.message_size, dtype=np.uint32)
		self.hypervectors = np.zeros((self.number_of_clauses, self.message_bits), dtype=np.uint32)
		for i in range(self.number_of_clauses):
			self.hypervectors[i,:] = np.random.choice(indexes, size=(self.message_bits), replace=False)

		self.initialized = False

	def allocate_gpu_memory(self):
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)

		self.message_ta_state_gpu = []
		for depth in range(self.depth - 1):
			self.message_ta_state_gpu.append(cuda.mem_alloc(self.number_of_clauses*self.number_of_message_chunks*self.number_of_state_bits*4))

		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		self.clause_weights_dummy_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)

		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)
		self.clause_node_gpu = cuda.mem_alloc(int(self.number_of_clauses) * 4)
		self.hypervectors_gpu = cuda.mem_alloc(self.hypervectors.nbytes)
		cuda.memcpy_htod(self.hypervectors_gpu, self.hypervectors)

	def ta_action(self, depth, clause, ta):
		if depth == 0:
			if np.array_equal(self.ta_state, np.array([])):
				self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
				cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
			ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))

			return (ta_state[clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0
		else:
			#if np.array_equal(self.message_ta_state, np.array([])):
			self.message_ta_state = np.empty(self.number_of_clauses*self.number_of_message_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.message_ta_state, self.message_ta_state_gpu[depth-1])
			message_ta_state = self.message_ta_state.reshape((self.number_of_clauses, self.number_of_message_chunks, self.number_of_state_bits))

			return (message_ta_state[clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0
			
	def get_state(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
			self.clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return((self.ta_state, self.clause_weights, self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.depth, self.number_of_state_bits, self.number_of_ta_chunks, self.min_y, self.max_y))

	def set_state(self, state):
		self.number_of_outputs = state[2]
		self.number_of_clauses = state[3]
		self.number_of_literals = state[4]
		self.depth = state[5]
		self.number_of_state_bits = state[6]
		self.number_of_ta_chunks = state[7]
		self.min_y = state[8]
		self.max_y = state[9]
		
		self.ta_state_gpu = cuda.mem_alloc(self.depth*self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		cuda.memcpy_htod(self.ta_state_gpu, state[0])
		cuda.memcpy_htod(self.clause_weights_gpu, state[1])

		self.graphs_signature_train = np.array([])
		self.graphs_signature_test = np.array([])

		self.encoded_Y = np.array([])

		self.ta_state = np.array([])
		self.clause_weights = np.array([])

	def _init(self, graphs):
		self.number_of_features = graphs.hypervector_size
		self.number_of_literals = self.number_of_features*2
		self.number_of_ta_chunks = int((self.number_of_literals-1)//32 + 1)

		self.number_of_message_features = self.message_size
		self.number_of_message_literals = self.number_of_message_features*2
		self.number_of_message_chunks = int((self.number_of_message_literals-1)//32 + 1)

		if self.max_included_literals == None:
			self.max_included_literals = self.number_of_literals

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define LITERALS %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d
#define Q %f
#define MAX_INCLUDED_LITERALS %d
#define NEGATIVE_CLAUSES %d
#define MAX_NODES %d
#define MESSAGE_SIZE %d
#define MESSAGE_BITS %d
#define MESSAGE_PRIME %d
#define DETERMINISTIC %d
#define NUMBER_OF_EXAMPLES %d
""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.q, self.max_included_literals, self.negative_clauses, graphs.max_number_of_graph_nodes, self.message_size, self.message_bits, self.message_prime, int(not (self.s > 1.0)), graphs.number_of_graphs)

		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare = mod_prepare.get_function("prepare")

		self.prepare_message_ta_state = mod_prepare.get_function("prepare_message_ta_state")

		self.allocate_gpu_memory()

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
		self.update = mod_update.get_function("update")
		self.update.prepare("PPiiPPP")

		self.update_message = mod_update.get_function("update_message")
		self.update_message.prepare("PPiPPP")

		mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
		self.evaluate = mod_evaluate.get_function("evaluate")
		self.evaluate.prepare("PPiP")

		self.select_clause_node = mod_evaluate.get_function("select_clause_node")
		self.select_clause_node.prepare("PPiP")

		self.select_clause_updates = mod_evaluate.get_function("select_clause_updates")
		self.select_clause_updates.prepare("PPPPiPP")

		self.calculate_messages = mod_evaluate.get_function("calculate_messages")
		self.calculate_messages.prepare("PiiPP")

		self.calculate_messages_conditional = mod_evaluate.get_function("calculate_messages_conditional")
		self.calculate_messages_conditional.prepare("PiPPP")

		self.prepare_messages = mod_evaluate.get_function("prepare_messages")
		self.prepare_messages.prepare("iP")

		self.exchange_messages = mod_evaluate.get_function("exchange_messages")
		self.exchange_messages.prepare("iPPiiPPP")

		self.encode_messages = mod_evaluate.get_function("encode_messages")
		self.encode_messages.prepare("iPP")

		self.initialized = True

	def _init_fit(self, graphs, encoded_Y, incremental):
		if not self.initialized:
			self._init(graphs)
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)

			for depth in range(self.depth-1):
				self.prepare_message_ta_state(self.message_ta_state_gpu[depth], grid=self.grid, block=self.block)

			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		if not np.array_equal(self.graphs_signature_train, graphs.signature):
			self.graphs_signature_train = graphs.signature

			self.encoded_X_train_gpu = cuda.mem_alloc(graphs.X.nbytes)
			cuda.memcpy_htod(self.encoded_X_train_gpu, graphs.X)

			self.current_clause_node_output_train_gpu = cuda.mem_alloc(int(self.number_of_clauses * graphs.max_number_of_graph_node_chunks) * 4)
			self.next_clause_node_output_train_gpu = cuda.mem_alloc(int(self.number_of_clauses * graphs.max_number_of_graph_node_chunks) * 4)
			
			self.clause_X_int_train_gpu = cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.message_literals) * 4)
			
			self.clause_X_train_gpu = []
			for depth in range(self.depth-1):
				self.clause_X_train_gpu.append(cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.number_of_message_chunks) * 4))

			self.number_of_graph_node_edges_train_gpu = cuda.mem_alloc(graphs.number_of_graph_node_edges.nbytes)
			cuda.memcpy_htod(self.number_of_graph_node_edges_train_gpu, graphs.number_of_graph_node_edges)

			if graphs.edge.nbytes > 0:
				self.edge_train_gpu = cuda.mem_alloc(graphs.edge.nbytes)
				cuda.memcpy_htod(self.edge_train_gpu, graphs.edge)
			else:
				self.edge_train_gpu = cuda.mem_alloc(1)

			self.class_clause_update_gpu = cuda.mem_alloc(int(self.number_of_outputs * self.number_of_clauses) * 4)

		if not np.array_equal(self.encoded_Y, encoded_Y):
			self.encoded_Y = encoded_Y

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

	def _evaluate(
			self,
			graphs,
			number_of_graph_nodes,
			node_index,
			edge_index,
			current_clause_node_output,
			next_clause_node_output,
			number_of_graph_node_edges,
			edge,
			clause_X_int,
			clause_X,
			encoded_X
	):
		class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
		cuda.memcpy_htod(self.class_sum_gpu, class_sum)

		# Calculate messages to be submitted from layer one
		self.calculate_messages.prepared_call(
			self.grid,
			self.block,
			self.ta_state_gpu,
			np.int32(number_of_graph_nodes),
			np.int32(node_index),
			current_clause_node_output,
			encoded_X
		)
		cuda.Context.synchronize()

		# Iterate over layers
		for depth in range(self.depth-1):
			# Prepare messages
			self.prepare_messages.prepared_call(
				self.grid,
				self.block,
				number_of_graph_nodes,
				clause_X_int
			)
			cuda.Context.synchronize()

			# Send messages to neighbors
			self.exchange_messages.prepared_call(
				self.grid,
				self.block,
				number_of_graph_nodes,
				self.hypervectors_gpu,
				current_clause_node_output,
				np.int32(node_index),
				np.int32(edge_index),
				number_of_graph_node_edges,
				edge,
				clause_X_int
			)
			cuda.Context.synchronize()

			# Encode messages bitwise
			self.encode_messages.prepared_call(
				self.grid,
				self.block,
				number_of_graph_nodes,
				clause_X_int,
				clause_X[depth]
			)
			cuda.Context.synchronize()

			# Calculate next round of messages
			self.calculate_messages_conditional.prepared_call(
				self.grid,
				self.block,
				self.message_ta_state_gpu[depth],
				number_of_graph_nodes,
				current_clause_node_output,
				next_clause_node_output,
				clause_X[depth]
			)
			cuda.Context.synchronize()

			tmp = current_clause_node_output
			current_clause_node_output = next_clause_node_output
			next_clause_node_output = tmp

		self.evaluate.prepared_call(
			self.grid,
			self.block,
			current_clause_node_output,
			self.clause_weights_gpu,
			number_of_graph_nodes,
			self.class_sum_gpu
		)
		cuda.Context.synchronize()

		return current_clause_node_output

	def _fit(self, graphs, encoded_Y, epochs=100, incremental=False):
		self._init_fit(graphs, encoded_Y, incremental)

		class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
		for epoch in range(epochs):
			for e in range(graphs.number_of_graphs):
				class_sum[:] = 0
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				### Inference 

				current_clause_node_output = self._evaluate(
					graphs,
					np.int32(graphs.number_of_graph_nodes[e]),
					np.int32(graphs.node_index[e]),
					np.int32(graphs.edge_index[graphs.node_index[e]]),
					self.current_clause_node_output_train_gpu,
					self.next_clause_node_output_train_gpu,
					self.number_of_graph_node_edges_train_gpu,
					self.edge_train_gpu,
					self.clause_X_int_train_gpu,
					self.clause_X_train_gpu,
					self.encoded_X_train_gpu
				)

				### Learning

				# Select one true node per clause
				self.select_clause_node.prepared_call(
					self.grid,
					self.block,
					g.state,
					current_clause_node_output,
					int(graphs.number_of_graph_nodes[e]),
					self.clause_node_gpu
				)
				cuda.Context.synchronize()

				# Select which clauses to update and update weights
				self.select_clause_updates.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_Y_gpu,
					np.int32(e),
					self.clause_node_gpu,
					self.class_clause_update_gpu
				)
				cuda.Context.synchronize()

				# Update clause Tsetlin automata blocks for layer one
				self.update.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.ta_state_gpu,
					np.int32(graphs.number_of_graph_nodes[e]),
					np.int32(graphs.node_index[e]),
					self.clause_node_gpu,
					self.encoded_X_train_gpu,
					self.class_clause_update_gpu
				)
				cuda.Context.synchronize()

				# Update clause Tsetlin automata blocks for deeper layers
				for depth in range(self.depth-1):
					self.update_message.prepared_call(
						self.grid,
						self.block,
						g.state,
						self.message_ta_state_gpu[depth],
						np.int32(graphs.number_of_graph_nodes[e]),
						self.clause_node_gpu,
						self.clause_X_train_gpu[depth],
						self.class_clause_update_gpu
					)
					cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, graphs):
		if not self.initialized:
			print("Error: Model not trained.")
			sys.exit(-1)

		if not np.array_equal(self.graphs_signature_test, graphs.signature):
			self.graphs_signature_test = graphs.signature

			self.encoded_X_test_gpu = cuda.mem_alloc(graphs.X.nbytes)
			cuda.memcpy_htod(self.encoded_X_test_gpu, graphs.X)

			self.current_clause_node_output_test_gpu = cuda.mem_alloc(int(self.number_of_clauses * graphs.max_number_of_graph_node_chunks) * 4)
			self.next_clause_node_output_test_gpu = cuda.mem_alloc(int(self.number_of_clauses * graphs.max_number_of_graph_node_chunks) * 4)
			
			self.clause_X_int_test_gpu = cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.number_of_message_literals) * 4)

			self.clause_X_test_gpu = []
			for depth in range(self.depth-1):
				self.clause_X_test_gpu.append(cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.number_of_message_chunks) * 4))

			self.number_of_graph_node_edges_test_gpu = cuda.mem_alloc(graphs.number_of_graph_node_edges.nbytes)
			cuda.memcpy_htod(self.number_of_graph_node_edges_test_gpu, graphs.number_of_graph_node_edges)

			if graphs.edge.nbytes > 0:
				self.edge_test_gpu = cuda.mem_alloc(graphs.edge.nbytes)
				cuda.memcpy_htod(self.edge_test_gpu, graphs.edge)
			else:
				self.edge_test_gpu = cuda.mem_alloc(1)

		class_sum = np.zeros((graphs.number_of_graphs, self.number_of_outputs), dtype=np.int32)
		for e in range(graphs.number_of_graphs):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e,:])

			### Inference 

			self._evaluate(
				graphs,
				np.int32(graphs.number_of_graph_nodes[e]),
				np.int32(graphs.node_index[e]),
				np.int32(graphs.edge_index[graphs.node_index[e]]),
				self.current_clause_node_output_test_gpu,
				self.next_clause_node_output_test_gpu,
				self.number_of_graph_node_edges_test_gpu,
				self.edge_test_gpu,
				self.clause_X_int_test_gpu,
				self.clause_X_test_gpu,
				self.encoded_X_test_gpu
			)

			cuda.memcpy_dtoh(class_sum[e,:], self.class_sum_gpu)

		return class_sum
	
class MultiClassGraphTsetlinMachine(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			depth=1,
			message_size=256,
			message_bits=2,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(
			number_of_clauses,
			T,
			s,
			q=q,
			max_included_literals=max_included_literals,
			boost_true_positive_feedback=boost_true_positive_feedback,
			number_of_state_bits=number_of_state_bits,
			depth=depth,
			message_size=message_size,
			message_bits=message_bits,
			grid=grid,
			block=block
		)
		self.negative_clauses = 1

	def fit(self, graphs, Y, epochs=100, incremental=False):
		self.number_of_outputs = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(graphs, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, graphs):
		return self._score(graphs)

	def predict(self, graphs):
		return np.argmax(self.score(graphs), axis=1)
