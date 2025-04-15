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

import pickle
import sys

import numpy as np

import GraphTsetlinMachine.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix
import sys
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
			double_hashing=False,
			one_hot_encoding=False,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		print("Initialization of sparse structure.")

		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)//32 + 1
		self.T = int(T)

		self.depth = depth
		if type(s) != tuple:
			self.s = (s,) * self.depth
		else:
			self.s = s

		self.q = q
		self.max_included_literals = max_included_literals
		self.boost_true_positive_feedback = boost_true_positive_feedback

		self.number_of_state_bits = number_of_state_bits
		self.message_size = message_size
		self.message_bits = message_bits

		self.double_hashing = double_hashing
		self.one_hot_encoding = one_hot_encoding

		self.grid = grid
		self.block = block

		self.graphs_signature_train = np.array([])
		self.graphs_signature_test = np.array([])
		self.encoded_Y = np.array([])
		
		self.ta_state = np.array([])
		self.message_ta_state = [np.array([])] * (self.depth - 1)
		self.clause_weights = np.array([])

		if self.one_hot_encoding:
			self.message_bits = 1
			self.hypervectors = np.zeros((self.number_of_clauses, self.message_bits), dtype=np.uint32)
			# Initialized when the number of edge types is known
		elif self.double_hashing:
			from sympy import prevprime
			self.message_bits = 2
			self.hypervectors = np.zeros((self.number_of_clauses, self.message_bits), dtype=np.uint32)
			prime = prevprime(self.message_size)
			for i in range(self.number_of_clauses):
				self.hypervectors[i, 0] = i % (self.message_size)
				self.hypervectors[i, 1] = prime - (i % prime)
		else:
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

		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs * self.number_of_clauses * 4)
		# self.clause_weights_dummy_gpu = cuda.mem_alloc(self.number_of_outputs * self.number_of_clauses * 4) # Never used

		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)
		self.clause_node_gpu = cuda.mem_alloc(int(self.number_of_clauses) * 4)
		self.number_of_include_actions = cuda.mem_alloc(int(self.number_of_clauses) * 4)
		self.hypervectors_gpu = cuda.mem_alloc(self.hypervectors.nbytes)
		cuda.memcpy_htod(self.hypervectors_gpu, self.hypervectors)

	def ta_action(self, depth, clause, ta):
		if depth == 0:
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
			ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))

			return (ta_state[clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0
		else:
			self.message_ta_state[depth - 1] = np.empty(
				self.number_of_clauses * self.number_of_message_chunks * self.number_of_state_bits, dtype=np.uint32
			)
			cuda.memcpy_dtoh(self.message_ta_state[depth - 1], self.message_ta_state_gpu[depth - 1])
			message_ta_state_depth = self.message_ta_state[depth - 1].reshape(
				(self.number_of_clauses, self.number_of_message_chunks, self.number_of_state_bits)
			)

			return (message_ta_state_depth[clause, ta // 32, self.number_of_state_bits - 1] & (1 << (ta % 32))) > 0

	def get_hyperliterals(self, depth):
		if depth == 0:
			literals_gpu = cuda.mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
			self.get_hyperliterals_gpu(
				self.ta_state_gpu,
				np.int32(self.number_of_ta_chunks),
				np.int32(self.number_of_literals),
				literals_gpu,
				grid=self.grid,
				block=self.block,
			)
			literals = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
			cuda.memcpy_dtoh(literals, literals_gpu)
			literals = literals.reshape((self.number_of_clauses, self.number_of_literals))
		else:
			literals_gpu = cuda.mem_alloc(self.number_of_clauses * self.number_of_message_literals * 4)
			self.get_hyperliterals_gpu(
				self.message_ta_state_gpu[depth - 1],
				np.int32(self.number_of_message_chunks),
				np.int32(self.number_of_message_literals),
				literals_gpu,
				grid=self.grid,
				block=self.block,
			)
			literals = np.empty(self.number_of_clauses * self.number_of_message_literals, dtype=np.uint32)
			cuda.memcpy_dtoh(literals, literals_gpu)
			literals = literals.reshape((self.number_of_clauses, self.number_of_message_literals))

		return literals

	def get_ta_states(self, depth):
		if depth == 0:
			ta_states_gpu = cuda.mem_alloc(self.number_of_clauses * self.number_of_literals * 4)
			self.get_ta_states_gpu(
				self.ta_state_gpu,
				np.int32(self.number_of_ta_chunks),
				np.int32(self.number_of_literals),
				ta_states_gpu,
				grid=self.grid,
				block=self.block,
			)
			cuda.Context.synchronize()

			ta_states = np.empty(self.number_of_clauses * self.number_of_literals, dtype=np.uint32)
			cuda.memcpy_dtoh(ta_states, ta_states_gpu)

			return ta_states.reshape((self.number_of_clauses, self.number_of_literals))
		else:
			message_ta_states_gpu = cuda.mem_alloc(self.number_of_clauses * self.number_of_message_literals * 4)
			self.get_ta_states_gpu(
				self.message_ta_state_gpu[depth - 1],
				np.int32(self.number_of_message_chunks),
				np.int32(self.number_of_message_literals),
				message_ta_states_gpu,
				grid=self.grid,
				block=self.block,
			)
			cuda.Context.synchronize()

			message_ta_state = np.empty(self.number_of_clauses * self.number_of_message_literals, dtype=np.uint32)
			cuda.memcpy_dtoh(message_ta_state, message_ta_states_gpu)
			return message_ta_state.reshape((self.number_of_clauses, self.number_of_message_literals))

	def get_weights(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return self.clause_weights.reshape((self.number_of_outputs, self.number_of_clauses))

	def get_clause_literals(self, symbol_hv):
		"""
		Convert HV clauses to literals and return them.

		:params:
			symbol_hv: Symbol hypervectors from graphs created for training (graphs.hypervectors)

		:returns:
			float NDArray of shape (number_of_clauses, 2 * number_of_symbols)
			Indicating the presence of symbols in each clause. Each value is the number of symbol bits present in the clause.
		"""
		# Get clause literals in HV format
		hv_clauses = self.get_hyperliterals(0)

		# Must be float. Store the symbols present in each clause.
		clause_literals = np.zeros((self.number_of_clauses, 2 * symbol_hv.shape[0]))

		# Expand symbol indices to actual hypervectors (in column format)
		expanded_sym = np.zeros((self.number_of_literals // 2, symbol_hv.shape[0]), dtype=np.uint8)
		for sym_id in range(symbol_hv.shape[0]):
			sym_hv = symbol_hv[sym_id].ravel()
			expanded_sym[sym_hv, sym_id] = 1

		# Check if the symbols are present in the clause.
		clause_literals[:, : symbol_hv.shape[0]] = hv_clauses[:, : (self.number_of_literals // 2)] @ expanded_sym
		clause_literals[:, symbol_hv.shape[0] :] = hv_clauses[:, (self.number_of_literals // 2) :] @ expanded_sym

		return clause_literals

	def get_messages(self, depth: int, edge_types: int):
		"""
		Convert HV Message to clause indexes (considered as literals) and return them

		:params:
			depth: Should be greater than 0. For depth = 0, use get_clause_literals()
			edge_types: Number of edge types in the input graphs. (len(graphs.edge_type_id))

		:returns:
			NDArray of shape (edge_types, number_of_clauses, 2 * number_of_clauses).
			Indicating messages recieved over each edge type, at depth 'depth'. Each value is the number of symbol bits present in the message.
		"""
		assert depth > 0, f"Expected depth > 0, got {depth}. Depth <= 0 means surface, use get_clause_literals()"

		# Get message literals in HV format
		hv_messages = self.get_hyperliterals(depth).astype(np.float32)

		# Store message literals as symbols(in this case, clause indices)
		message_literals = np.zeros((edge_types, self.number_of_clauses, 2 * self.number_of_clauses))

		for edge_type in range(edge_types):
			# Expand symbol indices to actual hypervectors and shift based on edge type
			expanded_sym = np.zeros((self.number_of_message_literals // 2, self.number_of_clauses))
			for sym_id in range(self.number_of_clauses):
				sym_hv = self.hypervectors[sym_id].ravel()
				sym_hv = (sym_hv + edge_type) % self.message_size
				expanded_sym[sym_hv, sym_id] = 1

			# Check if the symbols are present in the message.
			# Using numpy matrix multiplication for way faster computation compared to python loops.
			message_literals[edge_type, :, : self.number_of_clauses] = (
				hv_messages[:, : (self.number_of_message_literals // 2)] @ expanded_sym
			)
			message_literals[edge_type, :, self.number_of_clauses :] = (
				hv_messages[:, (self.number_of_message_literals // 2) :] @ expanded_sym
			)

		return message_literals

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

	def save(self, fname=""):
		# Copy data from GPU to CPU
		if np.array_equal(self.ta_state, np.array([])):
			self.ta_state = np.empty(
				self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, dtype=np.uint32
			)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)

		for depth in range(self.depth - 1):
			if np.array_equal(self.message_ta_state[depth], np.array([])):
				self.message_ta_state[depth] = np.empty(
					self.number_of_clauses * self.number_of_message_chunks * self.number_of_state_bits, dtype=np.uint32
				)
				cuda.memcpy_dtoh(self.message_ta_state[depth], self.message_ta_state_gpu[depth])

		if np.array_equal(self.clause_weights, np.array([])):
			self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)

		state_dict = {
			# State arrays
			"ta_state": self.ta_state,
			"message_ta_state": self.message_ta_state,
			"clause_weights": self.clause_weights,
			"hypervectors": self.hypervectors,
			"number_of_outputs": self.number_of_outputs,
			"number_of_literals": self.number_of_literals,
			"number_of_message_literals": self.number_of_message_literals,
			"min_y": self.min_y,
			"max_y": self.max_y,
			"negative_clauses": self.negative_clauses,  # Set in children classes, should be set in this class.
			"max_number_of_graph_nodes": self.max_number_of_graph_nodes,
			# Parameters
			"number_of_clauses": self.number_of_clauses,
			"T": self.T,
			"s": self.s,
			"q": self.q,
			"max_included_literals": self.max_included_literals,
			"boost_true_positive_feedback": self.boost_true_positive_feedback,
			"number_of_state_bits": self.number_of_state_bits,
			"depth": self.depth,
			"message_size": self.message_size,
			"message_bits": self.message_bits,
			"double_hashing": self.double_hashing,
			"one_hot_encoding": self.one_hot_encoding,
		}

		# Save to file
		if len(fname) > 0:
			print(f"Saving model to {fname}.")
			with open(fname, "wb") as f:
				pickle.dump(state_dict, f)

		return state_dict

	def load(self, state_dict={}, fname=""):
		if len(fname) == 0 and len(state_dict) == 0:
			print("Error: No file or state_dict provided. Pass either a file name or a state_dict.")
			return

		# Load from file
		if len(fname) > 0:
			print(f"Loading model from {fname}.")
			with open(fname, "rb") as f:
				state_dict = pickle.load(f)

		# Load arrays state_dict
		self.ta_state = state_dict["ta_state"]
		self.message_ta_state = state_dict["message_ta_state"]
		self.clause_weights = state_dict["clause_weights"]
		self.hypervectors = state_dict["hypervectors"]
		self.number_of_outputs = state_dict["number_of_outputs"]
		self.number_of_literals = state_dict["number_of_literals"]
		self.number_of_message_literals = state_dict["number_of_message_literals"]
		self.min_y = state_dict["min_y"]
		self.max_y = state_dict["max_y"]
		self.negative_clauses = state_dict["negative_clauses"]
		self.max_number_of_graph_nodes = state_dict["max_number_of_graph_nodes"]

		# Message size can change if one-hot encoding is used (when merged)
		self.message_size = state_dict["message_size"]

		# Initialize variables required in the _init() function
		self.number_of_features = self.number_of_literals // 2
		self.number_of_ta_chunks = int((self.number_of_literals - 1) // 32 + 1)

		self.number_of_message_features = self.number_of_message_literals // 2
		self.number_of_message_chunks = int((self.number_of_message_literals - 1) // 32 + 1)

		if self.max_included_literals is None:
			self.max_included_literals = self.number_of_literals

		# Initialize the gpu kernels and allocate gpu memory
		self._init_gpu_kernels()
		self.allocate_gpu_memory()

		# Copy states and weights to GPU
		cuda.memcpy_htod(self.ta_state_gpu, self.ta_state)
		for depth in range(self.depth - 1):
			cuda.memcpy_htod(self.message_ta_state_gpu[depth], self.message_ta_state[depth])
		cuda.memcpy_htod(self.clause_weights_gpu, self.clause_weights)

		# Now we are initialized
		self.initialized = True

	def _init(self, graphs):
		if self.one_hot_encoding:
			self.message_size = self.number_of_clauses * max(1, len(graphs.edge_type_id))
			for i in range(self.number_of_clauses):
				self.hypervectors[i, 0] = i * len(graphs.edge_type_id)

		self.number_of_features = graphs.hypervector_size
		self.number_of_literals = self.number_of_features*2
		self.number_of_ta_chunks = int((self.number_of_literals-1)//32 + 1)

		self.number_of_message_features = self.message_size
		self.number_of_message_literals = self.number_of_message_features*2
		self.number_of_message_chunks = int((self.number_of_message_literals-1)//32 + 1)

		if self.max_included_literals == None:
			self.max_included_literals = self.number_of_literals

		self.max_number_of_graph_nodes = graphs.max_number_of_graph_nodes

		self._init_gpu_kernels()
		self.allocate_gpu_memory()
		self.initialized = True

	def _init_gpu_kernels(self):
		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define LITERALS %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define THRESHOLD %d
#define Q %f
#define MAX_INCLUDED_LITERALS %d
#define NEGATIVE_CLAUSES %d
#define MAX_NODES %d
#define MESSAGE_SIZE %d
#define MESSAGE_BITS %d
""" % (
			self.number_of_outputs,
			self.number_of_clauses,
			self.number_of_literals,
			self.number_of_state_bits,
			self.boost_true_positive_feedback,
			self.T,
			self.q,
			self.max_included_literals,
			self.negative_clauses,
			self.max_number_of_graph_nodes,
			self.message_size,
			self.message_bits,
		)

		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare = mod_prepare.get_function("prepare")
		self.prepare_message_ta_state = mod_prepare.get_function("prepare_message_ta_state")

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
		self.update = mod_update.get_function("update")
		self.update.prepare("PfPiiPPPP")

		self.update_message = mod_update.get_function("update_message")
		self.update_message.prepare("PfPiPPPP")

		mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
		self.evaluate = mod_evaluate.get_function("evaluate")
		self.evaluate.prepare("PPiP")

		self.select_clause_node = mod_evaluate.get_function("select_clause_node")
		self.select_clause_node.prepare("PPiP")

		self.select_clause_updates = mod_evaluate.get_function("select_clause_updates")
		self.select_clause_updates.prepare("PPPPiPP")

		self.calculate_messages = mod_evaluate.get_function("calculate_messages")
		self.calculate_messages.prepare("PPiiiPPP")

		self.calculate_messages_conditional = mod_evaluate.get_function("calculate_messages_conditional")
		self.calculate_messages_conditional.prepare("PPiiiPPPP")

		self.prepare_messages = mod_evaluate.get_function("prepare_messages")
		self.prepare_messages.prepare("iP")

		self.exchange_messages = mod_evaluate.get_function("exchange_messages")
		self.exchange_messages.prepare("iPPiiPPP")

		self.encode_messages = mod_evaluate.get_function("encode_messages")
		self.encode_messages.prepare("iPP")

		mod_transform = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
		self.transform_gpu = mod_transform.get_function("transform")
		self.transform_gpu.prepare("PiP")

		self.transform_nodewise_gpu = mod_transform.get_function("transform_nodewise")
		self.transform_nodewise_gpu.prepare("PiP")

		mod_clauses = SourceModule(parameters + kernels.code_header + kernels.code_clauses, no_extern_c=True)
		self.get_ta_states_gpu = mod_clauses.get_function("get_ta_states")
		self.get_ta_states_gpu.prepare("PiiP")
		self.get_hyperliterals_gpu = mod_clauses.get_function("get_hyperliterals")
		self.get_hyperliterals_gpu.prepare("PiiP")

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
			
			self.clause_X_int_train_gpu = cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.number_of_message_literals) * 4)
			
			self.clause_X_train_gpu = []
			for depth in range(self.depth-1):
				self.clause_X_train_gpu.append(cuda.mem_alloc(int(graphs.max_number_of_graph_nodes * self.number_of_message_chunks) * 4))

			self.node_type_train_gpu = cuda.mem_alloc(graphs.node_type.nbytes)
			cuda.memcpy_htod(self.node_type_train_gpu, graphs.node_type)

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
			node_type,
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
			node_type,
			np.int32(graphs.number_of_node_types()),
			np.int32(number_of_graph_nodes),
			np.int32(node_index),
			current_clause_node_output,
			self.number_of_include_actions,
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
				node_type,
				np.int32(graphs.number_of_node_types()),
				np.int32(number_of_graph_nodes),
				np.int32(node_index),
				current_clause_node_output,
				next_clause_node_output,
				self.number_of_include_actions,
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
					self.node_type_train_gpu,
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
					self.s[0],
					self.ta_state_gpu,
					np.int32(graphs.number_of_graph_nodes[e]),
					np.int32(graphs.node_index[e]),
					self.clause_node_gpu,
					self.number_of_include_actions,
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
						self.s[depth+1],
						self.message_ta_state_gpu[depth],
						np.int32(graphs.number_of_graph_nodes[e]),
						self.clause_node_gpu,
						self.number_of_include_actions,
						self.clause_X_train_gpu[depth],
						self.class_clause_update_gpu
					)
					cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score_init(self, graphs):
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

			self.node_type_test_gpu = cuda.mem_alloc(graphs.node_type.nbytes)
			cuda.memcpy_htod(self.node_type_test_gpu, graphs.node_type)

			self.number_of_graph_node_edges_test_gpu = cuda.mem_alloc(graphs.number_of_graph_node_edges.nbytes)
			cuda.memcpy_htod(self.number_of_graph_node_edges_test_gpu, graphs.number_of_graph_node_edges)

			if graphs.edge.nbytes > 0:
				self.edge_test_gpu = cuda.mem_alloc(graphs.edge.nbytes)
				cuda.memcpy_htod(self.edge_test_gpu, graphs.edge)
			else:
				self.edge_test_gpu = cuda.mem_alloc(1)

	def _score(self, graphs):
		self._score_init(graphs)

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
				self.node_type_test_gpu,
				self.number_of_graph_node_edges_test_gpu,
				self.edge_test_gpu,
				self.clause_X_int_test_gpu,
				self.clause_X_test_gpu,
				self.encoded_X_test_gpu
			)

			cuda.memcpy_dtoh(class_sum[e,:], self.class_sum_gpu)

		return class_sum

	def transform(self, graphs):
		self._score_init(graphs)

		class_sum = np.zeros((graphs.number_of_graphs, self.number_of_outputs), dtype=np.int32)
		transformed_X = np.zeros((graphs.number_of_graphs, self.number_of_clauses), dtype=np.int32)
		for e in range(graphs.number_of_graphs):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e, :])

			### Inference

			current_clause_node_output = self._evaluate(
				graphs,
				np.int32(graphs.number_of_graph_nodes[e]),
				np.int32(graphs.node_index[e]),
				np.int32(graphs.edge_index[graphs.node_index[e]]),
				self.current_clause_node_output_test_gpu,
				self.next_clause_node_output_test_gpu,
				self.node_type_test_gpu,
				self.number_of_graph_node_edges_test_gpu,
				self.edge_test_gpu,
				self.clause_X_int_test_gpu,
				self.clause_X_test_gpu,
				self.encoded_X_test_gpu,
			)

			transformed_X_sample_gpu = cuda.mem_alloc(self.number_of_clauses * 4)
			self.transform_gpu.prepared_call(
				self.grid,
				self.block,
				current_clause_node_output,
				np.int32(graphs.number_of_graph_nodes[e]),
				transformed_X_sample_gpu,
			)

			cuda.memcpy_dtoh(class_sum[e, :], self.class_sum_gpu)
			cuda.memcpy_dtoh(transformed_X[e, :], transformed_X_sample_gpu)

		return transformed_X, class_sum

	def transform_nodewise(self, graphs):
		self._score_init(graphs)

		class_sum = np.zeros((graphs.number_of_graphs, self.number_of_outputs), dtype=np.int32)
		transformed_X = np.zeros(
			(graphs.number_of_graphs, self.number_of_clauses, np.max(graphs.number_of_graph_nodes)), dtype=np.int32
		)
		for e in range(graphs.number_of_graphs):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e, :])

			### Inference

			current_clause_node_output = self._evaluate(
				graphs,
				np.int32(graphs.number_of_graph_nodes[e]),
				np.int32(graphs.node_index[e]),
				np.int32(graphs.edge_index[graphs.node_index[e]]),
				self.current_clause_node_output_test_gpu,
				self.next_clause_node_output_test_gpu,
				self.node_type_test_gpu,
				self.number_of_graph_node_edges_test_gpu,
				self.edge_test_gpu,
				self.clause_X_int_test_gpu,
				self.clause_X_test_gpu,
				self.encoded_X_test_gpu,
			)

			transformed_X_sample_gpu = cuda.mem_alloc(
				int(self.number_of_clauses * np.max(graphs.number_of_graph_nodes) * 4)
			)
			self.transform_nodewise_gpu.prepared_call(
				self.grid,
				self.block,
				current_clause_node_output,
				np.int32(graphs.number_of_graph_nodes[e]),
				transformed_X_sample_gpu,
			)
			t = np.zeros((self.number_of_clauses * graphs.number_of_graph_nodes[e]), dtype=np.uint32)
			cuda.memcpy_dtoh(t, transformed_X_sample_gpu)
			transformed_X[e, :, : graphs.number_of_graph_nodes[e]] = t.reshape(
				(self.number_of_clauses, graphs.number_of_graph_nodes[e])
			)

			cuda.memcpy_dtoh(class_sum[e, :], self.class_sum_gpu)

		return transformed_X.reshape(
			(graphs.number_of_graphs, self.number_of_clauses, np.max(graphs.number_of_graph_nodes))
		), class_sum


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
			double_hashing=False,
			one_hot_encoding=False,
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
			double_hashing=double_hashing,
			one_hot_encoding=one_hot_encoding,
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

class MultiOutputGraphTsetlinMachine(CommonTsetlinMachine):
	"""
	This class implements the MultiOuput GTM...
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
		double_hashing=False,
		one_hot_encoding=False,
		grid=(16*13*4, 1, 1),
		block=(128, 1, 1),
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
			double_hashing=double_hashing,
			one_hot_encoding=one_hot_encoding,
			grid=grid,
			block=block
		)
		self.negative_clauses = 1

	def fit(self, graphs, Y, epochs=100, incremental=False):
		self.number_of_outputs = Y.shape[1]

		self.max_y = None
		self.min_y = None

		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(graphs, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, graphs):
		return self._score(graphs)

	def predict(self, graphs):
		class_sums = self.score(graphs)
		preds = (class_sums >= 0).astype(np.uint32)
		return preds

class GraphTsetlinMachine(CommonTsetlinMachine):
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
			double_hashing=False,
			one_hot_encoding=False,
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
			double_hashing=double_hashing,
			one_hot_encoding=one_hot_encoding,
			grid=grid,
			block=block
		)
		self.negative_clauses = 1

	def fit(self, graphs, Y):
		self.number_of_outputs = 1
		
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(graphs, encoded_Y)

		return

	def score(self, graphs):
		return self._score(graphs)[:,0]

	def predict(self, X):
		score = self.score(X)
		return score >= 0
