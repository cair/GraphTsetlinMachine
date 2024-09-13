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

from time import time

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			hypervector_size=1024,
			hypervector_bits=3,
			depth=1,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		print("Initialization of sparse structure.")

		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.hypervector_size = hypervector_size
		self.depth = depth
		self.q = q
		self.max_included_literals = max_included_literals
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

		self.train_signature = np.array([])
		self.test_signature = np.array([])
		self.encoded_Y = np.array([])
		
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.encode = mod_encode.get_function("encode")
		self.encode.prepare("PPPiiiiiiii")
		
		self.restore = mod_encode.get_function("restore")
		self.restore.prepare("PPPiiiiiiii")

		self.encode_packed = mod_encode.get_function("encode_packed")
		self.encode_packed.prepare("PPPiiiiiiii")
		
		self.restore_packed = mod_encode.get_function("restore_packed")
		self.restore_packed.prepare("PPPiiiiiiii")

		self.produce_autoencoder_examples= mod_encode.get_function("produce_autoencoder_example")
		self.produce_autoencoder_examples.prepare("PPiPPiPPiPPiiii")

		self.initialized = False

	def allocate_gpu_memory(self):
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)

		self.included_literals_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_literals*2*4) # Contains index and state of included literals per clause, none at start
		self.included_literals_length_gpu = cuda.mem_alloc(self.number_of_clauses*4) # Number of included literals per clause

	def ta_action(self, mc_tm_class, clause, ta):
		if np.array_equal(self.ta_state, np.array([])):
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
		ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))

		return (ta_state[mc_tm_class, clause, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0

	def get_state(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
			self.clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return((self.ta_state, self.clause_weights, self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.dim, self.patch_dim, self.number_of_patches, self.number_of_state_bits, self.number_of_ta_chunks, self.append_negated, self.min_y, self.max_y))

	def set_state(self, state):
		self.number_of_outputs = state[2]
		self.number_of_clauses = state[3]
		self.number_of_literals = state[4]
		self.dim = state[5]
		self.patch_dim = state[6]
		self.number_of_patches = state[7]
		self.number_of_state_bits = state[8]
		self.number_of_ta_chunks = state[9]
		self.append_negated = state[10]
		self.min_y = state[11]
		self.max_y = state[12]
		
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		cuda.memcpy_htod(self.ta_state_gpu, state[0])
		cuda.memcpy_htod(self.clause_weights_gpu, state[1])

		self.train_signature = np.array([])
		self.test_signature = np.array([])

		self.encoded_Y = np.array([])

		self.ta_state = np.array([])
		self.clause_weights = np.array([])

	# Transform input data for processing at next layer
	def transform(self, X):
		X = csr_matrix(X)

		number_of_examples = X.shape[0]
		
		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define S %f
#define THRESHOLD %d

#define NEGATIVE_CLAUSES %d

#define PATCHES %d

#define NUMBER_OF_EXAMPLES %d
		""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples)

		mod = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
		transform_gpu = mod.get_function("transform")

		X_transformed_gpu = cuda.mem_alloc(self.number_of_clauses*4)

		X_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
		cuda.memcpy_htod(X_indptr_gpu, X.indptr)

		X_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
		cuda.memcpy_htod(X_indices_gpu, X.indices)

		X_transformed = np.empty((number_of_examples, self.number_of_clauses), dtype=np.uint32)
		for e in range(number_of_examples):
			self.encode_packed.prepared_call(self.grid, self.block, X_indptr_gpu, X_indices_gpu, self.encoded_X_packed_gpu, np.int32(e), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(self.append_negated), np.int32(0))
			cuda.Context.synchronize()

			transform_gpu(
				self.included_literals_gpu,
				self.included_literals_length_gpu,
				self.encoded_X_packed_gpu,
				X_transformed_gpu,
				grid=self.grid,
				block=self.block
			)
			cuda.Context.synchronize()

			cuda.memcpy_dtoh(X_transformed[e,:], X_transformed_gpu)

			self.restore_packed.prepared_call(self.grid, self.block, X_indptr_gpu, X_indices_gpu, self.encoded_X_packed_gpu, np.int32(e), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(self.append_negated), np.int32(0))
			cuda.Context.synchronize()
		
		return csr_matrix(X_transformed)

	def _init(self, graphs):
		self.number_of_features = self.hypervector_size * self.depth
		if self.append_negated:
			self.number_of_literals = self.number_of_features * 2
		else:
			self.number_of_literals = self.number_of_features

		if self.max_included_literals == None:
			self.max_included_literals = self.number_of_literals

		self.number_of_ta_chunks = int((self.number_of_literals-1)/32 + 1)

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
#define NUMBER_OF_EXAMPLES %d
""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.q, self.max_included_literals, self.negative_clauses, X.shape[0])

		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare = mod_prepare.get_function("prepare")
		self.prepare_packed = mod_prepare.get_function("prepare_packed")

		self.allocate_gpu_memory()

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
		self.update = mod_update.get_function("update")
		self.update.prepare("PPPPPPi")

		self.evaluate_update = mod_update.get_function("evaluate")
		self.evaluate_update.prepare("PPPiP")

		mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
		self.evaluate = mod_evaluate.get_function("evaluate")
		self.evaluate.prepare("PPPP")

		self.evaluate_packed = mod_evaluate.get_function("evaluate_packed")
		self.evaluate_packed.prepare("PPPPPPP")

		encoded_X = np.zeros((graphs.max_node_count, self.number_of_ta_chunks), dtype=np.uint32)
		if self.append_negated:
			for n in range(self.max_node_count):
				for k in range(self.number_of_features, self.number_of_features*2):
					chunk = k // 32
					pos = k % 32
					encoded_X[n, chunk] |= (1 << pos)

		encoded_X = encoded_X.reshape(-1)
		self.encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes)
		cuda.memcpy_htod(self.encoded_X_gpu, encoded_X)

		# Encoded X packed

		encoded_X_packed = np.zeros(((graphs.max_node_count-1)//32 + 1, self.number_of_literals), dtype=np.uint32)
		if self.append_negated:
			for n_chunk in range((graphs.max_node_count-1)//32 + 1):
				for k in range(self.number_of_features, self.number_of_literals):
					encoded_X_packed[n_chunk, k] = (~0) 

		encoded_X_packed = encoded_X_packed.reshape(-1)
		self.encoded_X_packed_gpu = cuda.mem_alloc(encoded_X_packed.nbytes)
		cuda.memcpy_htod(self.encoded_X_packed_gpu, encoded_X_packed)

		self.initialized = True

	def _init_fit(self, graphs, encoded_Y, incremental):
		if not self.initialized:
			self._init(graphs)
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		if not np.array_equal(self.train_signature, graphs.signature):
			self.train_signature = graphs.signature
			
			self.X_train_indptr_gpu = cuda.mem_alloc(graphs.X.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_indptr_gpu, graphs.X.indptr)

			self.X_train_indices_gpu = cuda.mem_alloc(graphs.X.indices.nbytes)
			cuda.memcpy_htod(self.X_train_indices_gpu, graphs.X.indices)

			self.edges_train_indptr_gpu = cuda.mem_alloc(graphs.edges.indptr.nbytes)
			cuda.memcpy_htod(self.edges_train_indptr_gpu, graphs.edges.indptr)

			self.edges_train_indices_gpu = cuda.mem_alloc(graphs.edges.indices.nbytes)
			cuda.memcpy_htod(self.edges_train_indices_gpu, graphs.edges.indices)

			self.edges_train_data_gpu = cuda.mem_alloc(graphs.edges.data.nbytes)
			cuda.memcpy_htod(self.edges_train_data_gpu, graphs.edges.data)

		if not np.array_equal(self.encoded_Y, encoded_Y):
			self.encoded_Y = encoded_Y

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

	def _fit(self, graphs, encoded_Y, epochs=100, incremental=False):
		self._init_fit(graphs, encoded_Y, incremental)

		class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
		for epoch in range(epochs):
			for e in range(graphs.X.shape[0]):
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				self.encode.prepared_call(
					self.grid,
					self.block,
					self.X_train_indptr_gpu,
					self.X_train_indices_gpu,
					self.encoded_X_gpu,
					np.int32(e),
					np.int32(self.hypervector_size),
					np.int32(self.depth),
					np.int32(self.append_negated)
				)
				cuda.Context.synchronize()

				self.evaluate_update.prepared_call(self.grid, self.block, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, self.node_count[e], self.encoded_X_gpu)
				cuda.Context.synchronize()

				self.update.prepared_call(self.grid, self.block, g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, self.node_count[e], self.encoded_X_gpu, self.encoded_Y_gpu, np.int32(e))
				cuda.Context.synchronize()

				self.restore.prepared_call(self.grid, self.block, self.X_train_indptr_gpu, self.X_train_indices_gpu, self.encoded_X_gpu, np.int32(e), np.int32(self.hypervector_size), np.int32(self.depth), np.int32(self.append_negated))
				cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, graphs):
		if not self.initialized:
			print("Error: Model not trained.")
			sys.exit(-1)

		if not np.array_equal(self.test_signature, graphs.signature):
			self.test_signature = graphs.signature

			self.X_test_indptr_gpu = cuda.mem_alloc(graphs.X.indptr.nbytes)
			cuda.memcpy_htod(self.X_test_indptr_gpu, graphs.X.indptr)

			self.X_test_indices_gpu = cuda.mem_alloc(graphs.X.indices.nbytes)
			cuda.memcpy_htod(self.X_test_indices_gpu, graphs.X.indices)

			self.edges_test_indptr_gpu = cuda.mem_alloc(graphs.edges.indptr.nbytes)
			cuda.memcpy_htod(self.edges_test_indptr_gpu, graphs.edges.indptr)

			self.edges_test_indices_gpu = cuda.mem_alloc(graphs.edges.indices.nbytes)
			cuda.memcpy_htod(self.edges_test_indices_gpu, graphs.edges.indices)

			self.edges_test_data_gpu = cuda.mem_alloc(graphs.edges.data.nbytes)
			cuda.memcpy_htod(self.edges_test_data_gpu, graphs.edges.data)

		self.prepare_packed(
			g.state,
			self.ta_state_gpu,
			self.included_literals_gpu,
			self.included_literals_length_gpu,
			grid=self.grid,
			block=self.block
		)
		cuda.Context.synchronize()
        
		class_sum = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.int32)
		for e in range(X.shape[0]):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e,:])

			self.encode_packed.prepared_call(
				self.grid,
				self.block,
				self.X_test_indptr_gpu,
				self.X_test_indices_gpu,
				self.encoded_X_packed_gpu,
				np.int32(e),
				np.int32(self.hypervector_size),
				np.int32(self.depth),
				np.int32(self.append_negated)
			)
			cuda.Context.synchronize()

			self.evaluate_packed.prepared_call(
				self.grid,
				self.block,
				self.included_literals_gpu,
				self.included_literals_length_gpu,
				self.excluded_literals_gpu,
				self.excluded_literals_length_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				self.encoded_X_packed_gpu
			)
			cuda.Context.synchronize()

			self.restore_packed.prepared_call(self.grid, self.block, self.X_test_indptr_gpu, self.X_test_indices_gpu, self.encoded_X_packed_gpu, np.int32(e), np.int32(self.dim[0]), np.int32(self.dim[1]), np.int32(self.dim[2]), np.int32(self.patch_dim[0]), np.int32(self.patch_dim[1]), np.int32(self.append_negated), np.int32(0))
			cuda.Context.synchronize()

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
			hypervector_size=1024,
			hypervector_bits=3,
			depth=1,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(
			number_of_clauses,
			T,
			s,
			hypervector_size=hypervector_size,
			hypervector_bits= hypervector_bits,
			depth=depth,
			q=q,
			max_included_literals=max_included_literals,
			boost_true_positive_feedback=boost_true_positive_feedback,
			number_of_state_bits=number_of_state_bits,
			append_negated=append_negated,
			grid=grid,
			block=block
		)
		self.negative_clauses = 1

	def fit(self, graphs, Y, epochs=100, incremental=False):
		if not graphs.encoded:
			graphs.encode(hypervector_size = self.hypervector_size, hypervector_bits = self.hypervector_bits)

		self.number_of_outputs = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(graphs, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)

class MultiOutputConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			dim,
			patch_dim,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.dim = dim
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = Y.shape[1]

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, X):
		X = csr_matrix(X)

		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32)

class MultiOutputTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = Y.shape[1]

		self.dim = (X.shape[1], 1, 1)
		self.patch_dim = (X.shape[1], 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32)

class MultiClassTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = int(np.max(Y) + 1)

		self.dim = (X.shape[1], 1, 1)
		self.patch_dim = (X.shape[1], 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)
		
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return int(self.score(X) >= 0)

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 0

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y

class AutoEncoderTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			active_output,
			q=1.0,
			max_included_literals=None,
			accumulation = 1,
			boost_true_positive_feedback=1,
			number_of_state_bits=8,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

		self.active_output = np.array(active_output).astype(np.uint32)
		self.accumulation = accumulation

	def _init_fit(self, X_csr, encoded_Y, incremental):
		if not self.initialized:
			self._init(X_csr)
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		if not np.array_equal(self.X_train, np.concatenate((X_csr.indptr, X_csr.indices))):
			self.train_X = np.concatenate((X_csr.indptr, X_csr.indices))

			X_csc = X_csr.tocsc()
			
			self.X_train_csr_indptr_gpu = cuda.mem_alloc(X_csr.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_csr_indptr_gpu, X_csr.indptr)

			self.X_train_csr_indices_gpu = cuda.mem_alloc(X_csr.indices.nbytes)
			cuda.memcpy_htod(self.X_train_csr_indices_gpu, X_csr.indices)

			self.X_train_csc_indptr_gpu = cuda.mem_alloc(X_csc.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_csc_indptr_gpu, X_csc.indptr)

			self.X_train_csc_indices_gpu = cuda.mem_alloc(X_csc.indices.nbytes)
			cuda.memcpy_htod(self.X_train_csc_indices_gpu, X_csc.indices)

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

			self.active_output_gpu = cuda.mem_alloc(self.active_output.nbytes)
			cuda.memcpy_htod(self.active_output_gpu, self.active_output)

	def _fit(self, X_csr, encoded_Y, number_of_examples, epochs, incremental=False):
		self._init_fit(X_csr, encoded_Y, incremental=incremental)

		for epoch in range(epochs):
			for e in range(number_of_examples):
				class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				target = np.random.choice(self.number_of_outputs)
				self.produce_autoencoder_examples.prepared_call(
                                            self.grid,
                                            self.block,
											g.state,
                                            self.active_output_gpu,
                                            self.active_output.shape[0],
                                            self.X_train_csr_indptr_gpu,
                                            self.X_train_csr_indices_gpu,
                                            X_csr.shape[0],
                                            self.X_train_csc_indptr_gpu,
                                            self.X_train_csc_indices_gpu,
                                            X_csr.shape[1],
                                            self.encoded_X_gpu,
                                            self.encoded_Y_gpu,
                                            target,
                                            int(self.accumulation),
                                            int(self.T),
                                            int(self.append_negated))
				cuda.Context.synchronize()

				self.evaluate_update.prepared_call(self.grid, self.block, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, self.encoded_X_gpu)
				cuda.Context.synchronize()

				self.update.prepared_call(self.grid, self.block, g.state, self.ta_state_gpu, self.clause_weights_gpu, self.class_sum_gpu, self.encoded_X_gpu, self.encoded_Y_gpu, np.int32(0))
				cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def fit(self, X, number_of_examples=2000, epochs=100, incremental=False):
		X_csr = csr_matrix(X)

		self.number_of_outputs = self.active_output.shape[0]

		self.dim = (X_csr.shape[1], 1, 1)
		self.patch_dim = (X_csr.shape[1], 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.zeros(self.number_of_outputs, dtype = np.int32)

		self._fit(X_csr, encoded_Y, number_of_examples, epochs, incremental = incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)
