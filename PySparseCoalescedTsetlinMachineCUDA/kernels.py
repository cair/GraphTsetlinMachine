# Copyright (c) 2021 Ole-Christoffer Granmo

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

code_header = """
	#include <curand_kernel.h>
	
	#define INT_SIZE 32

	#define LA_CHUNKS (((FEATURES-1)/INT_SIZE + 1))
	#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

	#if (FEATURES % 32 != 0)
	#define FILTER (~(0xffffffff << (FEATURES % INT_SIZE)))
	#else
	#define FILTER 0xffffffff
	#endif

	#define PATCH_CHUNKS (((PATCHES-1)/INT_SIZE + 1))

	#if (PATCH_CHUNKS % 32 != 0)
	#define PATCH_FILTER (~(0xffffffff << (PATCHES % INT_SIZE)))
	#else
	#define PATCH_FILTER 0xffffffff
	#endif
"""

code_update = """
	extern "C"
	{
		// Counts number of include actions for a given clause
	    __device__ inline int number_of_include_actions(unsigned int *ta_state)
	    {
	        int number_of_include_actions = 0;
	        for (int k = 0; k < LA_CHUNKS-1; ++k) {
	            unsigned int ta_pos = k*STATE_BITS + STATE_BITS-1;
	            number_of_include_actions += __popc(ta_state[ta_pos]);
	        }
	        unsigned int ta_pos = (LA_CHUNKS-1)*STATE_BITS + STATE_BITS-1;
	        number_of_include_actions += __popc(ta_state[ta_pos] & FILTER);

	        return(number_of_include_actions);
	    }

    	// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;

				carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] |= carry;
				}
			}   
		}

		// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void dec(unsigned int *ta_state, int clause, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;
				carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] &= ~carry;
				}
			} 
		}

		__device__ inline void calculate_clause_output(curandState *localState, unsigned int *ta_state, unsigned int *clause_output, int *clause_patch, int *X)
		{
			int output_one_patches[PATCHES];
			int output_one_patches_count;

			// Evaluate each patch (convolution)
			output_one_patches_count = 0;
			for (int patch = 0; patch < PATCHES; ++patch) {
				int patch_clause_output = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
						patch_clause_output = 0;
						break;
					}
				}

				if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS - 1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
					patch_clause_output = 0;
				}

				if (patch_clause_output) {
					output_one_patches[output_one_patches_count] = patch;
					output_one_patches_count++;
				}
			}
		
			if (output_one_patches_count > 0) {
				*clause_output = 1;
				int patch_id = curand(localState) % output_one_patches_count;
				*clause_patch = output_one_patches[patch_id];
			} else {
				*clause_output = 0;
				*clause_patch = -1;
			}
		}

		__device__ inline void update_clause(curandState *localState, int *clause_weight, unsigned int *ta_state, int clause_output, int clause_patch, int *X, int y, int class_sum)
		{
			int target = 1 - 2*(class_sum > y);
			
			if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, CLASSES-1)) {
				return;
			}

			int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		
			int absolute_prediction_error = abs(y - class_sum);
			if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
				if (target*sign > 0) {
					int included_literals = number_of_include_actions(ta_state);

					if (clause_output && abs(*clause_weight) < INT_MAX) {
						(*clause_weight) += sign;
					}

					// Type I Feedback
					for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
						// Generate random bit values
						unsigned int la_feedback = 0;
						for (int b = 0; b < INT_SIZE; ++b) {
							if (curand_uniform(localState) <= 1.0/S) {
								la_feedback |= (1 << b);
							}
						}

						if (clause_output && included_literals <= MAX_INCLUDED_LITERALS) {
							#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
								inc(ta_state, 0, la_chunk, X[clause_patch*LA_CHUNKS + la_chunk]);
							#else
								inc(ta_state, 0, la_chunk, X[clause_patch*LA_CHUNKS + la_chunk] & (~la_feedback));
							#endif

							dec(ta_state, 0, la_chunk, (~X[clause_patch*LA_CHUNKS + la_chunk]) & la_feedback);
						} else {
							dec(ta_state, 0, la_chunk, la_feedback);
						}
					}
				} else if (target*sign < 0 && clause_output) {
					// Type II Feedback

					(*clause_weight) -= sign;
					#if NEGATIVE_CLAUSES == 0
						if (*clause_weight < 1) {
							*clause_weight = 1;
						}
					#endif

					for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
						inc(ta_state, 0, la_chunk, (~X[clause_patch*LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
					}
				}
			}
		}

		// Evaluate example
		__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int clause = index; clause < CLAUSES; clause += stride) {
				unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

				int clause_output;
				for (int patch = 0; patch < PATCHES; ++patch) {
					clause_output = 1;
					for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
						if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
							clause_output = 0;
							break;
						}
					}

					if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
						clause_output = 0;
					}

					if (clause_output) {
						break;
					}
				}

				if (clause_output) {
					for (int class_id = 0; class_id < CLASSES; ++class_id) {
						int clause_weight = clause_weights[class_id*CLAUSES + clause];
						atomicAdd(&class_sum[class_id], clause_weight);					
					}
				}
			}
		}

		// Update state of Tsetlin Automata team
		__global__ void update(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			// Calculate clause output first
			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

				unsigned int clause_output;
				int clause_patch;
				calculate_clause_output(&localState, ta_state, &clause_output, &clause_patch, X);

				for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
					int local_class_sum = class_sum[class_id];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}
					update_clause(&localState, &clause_weights[class_id*CLAUSES + clause], ta_state, clause_output, clause_patch, X, y[example*CLASSES + class_id], local_class_sum);
				}
			}
		
			state[index] = localState;
		}
    }
"""

code_evaluate = """
	extern "C"
    {
		// Evaluate examples
		__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int clause = index; clause < CLAUSES; clause += stride) {
				unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

				int all_exclude = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
						all_exclude = 0;
						break;
					}
				}

				if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
					all_exclude = 0;
				}

				if (all_exclude) {
					continue;
				}

				int clause_output;
				for (int patch = 0; patch < PATCHES; ++patch) {
					clause_output = 1;
					for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
						if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
							clause_output = 0;
							break;
						}
					}

					if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
						clause_output = 0;
					}

					if (clause_output) {
						break;
					}
				}

				if (clause_output) {
					for (int class_id = 0; class_id < CLASSES; ++class_id) {
						int clause_weight = clause_weights[class_id*CLAUSES + clause];
						atomicAdd(&class_sum[class_id], clause_weight);					
					}
				}
			}
		}

		// Evaluate examples
		__global__ void evaluate_packed(
			unsigned int *included_literals,
			unsigned int *included_literals_length,
			unsigned int *excluded_literals,
			unsigned int *excluded_literals_length,
			int *clause_weights,
			int *class_sum,
			int *X
		)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int clause = index; clause < CLAUSES; clause += stride) {
				if (included_literals_length[clause] == 0) {
					continue;
				}

				unsigned int clause_output = 0;
				for (int patch_chunk = 0; patch_chunk < PATCH_CHUNKS-1; ++patch_chunk) {
					clause_output = (~(0U));
					for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
						clause_output &= X[patch_chunk*FEATURES + included_literals[clause*FEATURES*2 + literal*2]];
					}

					if (clause_output) {
						break;
					}
				}

				if (!clause_output) {
					clause_output = PATCH_FILTER;
					for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
						clause_output &= X[(PATCH_CHUNKS-1)*FEATURES + included_literals[clause*FEATURES*2 + literal*2]];
					}
				}

				if (clause_output) {
					for (int class_id = 0; class_id < CLASSES; ++class_id) {
						int clause_weight = clause_weights[class_id*CLAUSES + clause];
						atomicAdd(&class_sum[class_id], clause_weight);					
					}
				}
			}
		}
	}
"""

code_prepare = """
	extern "C"
    {
		__global__ void prepare(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			curandState localState = state[index];

			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
					#if NEGATIVE_CLAUSES == 1
						clause_weights[class_id*CLAUSES + clause] = 1 - 2 * (curand(&localState) % 2);
					#else
						clause_weights[class_id*CLAUSES + clause] = 1;
					#endif
				}

				unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					for (int b = 0; b < STATE_BITS-1; ++b) {
						ta_state[la_chunk*STATE_BITS + b] = ~0;
					}
					ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] = 0;
				}
			}

			state[index] = localState;
		}

		__global__ void prepare_packed(
			curandState *state,
			unsigned int *global_ta_state,
			unsigned int *included_literals,
			unsigned int *included_literals_length,
			unsigned int *excluded_literals,
			unsigned int *excluded_literals_length,
			int *clause_weights,
			int *class_sum
		)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			curandState localState = state[index];

			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

				included_literals_length[clause] = 0;
				for (int literal = 0; literal < FEATURES; ++literal) {
					int chunk = literal / INT_SIZE;
					int pos = literal % INT_SIZE;

					if ((ta_state[chunk*STATE_BITS + STATE_BITS - 1] & (1U << pos)) > 0) {
						included_literals[clause*FEATURES*2 + included_literals_length[clause]*2] = literal;
						included_literals_length[clause]++;
					}
				}
			}

			state[index] = localState;
		}
	}
"""

code_encode = """
	#include <curand_kernel.h>

	extern "C"
    {
		__global__ void encode(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

			int number_of_ta_chunks;
			if (append_negated) {
				number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
			} else {
				number_of_ta_chunks= (((number_of_features-1)/32 + 1));
			}

			unsigned int *indices = &X_indices[X_indptr[e]];
			int number_of_indices = X_indptr[e + 1] - X_indptr[e]; 

			for (int k = 0; k < number_of_indices; ++k) {
				int y = indices[k] / (dim_x*dim_z);
				int x = (indices[k] % (dim_x*dim_z)) / dim_z;
				int z = (indices[k] % (dim_x*dim_z)) % dim_z;

				for (int patch = index; patch < number_of_patches; patch += stride) {
					int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
					int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

					if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) || (x >= patch_coordinate_x + patch_dim_x)) {
						continue;
					}

					int p_y = y - patch_coordinate_y;
					int p_x = x - patch_coordinate_x;

					int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

					int chunk_nr = patch_pos / 32;
					int chunk_pos = patch_pos % 32;
					encoded_X[patch * number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos);

					if (append_negated) {
						int chunk_nr = (patch_pos + number_of_features) / 32;
						int chunk_pos = (patch_pos + number_of_features) % 32;
						encoded_X[patch * number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos);
					}
				}
		    }		
		}

		__global__ void restore(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

			int number_of_ta_chunks;
			if (append_negated) {
				number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
			} else {
				number_of_ta_chunks= (((number_of_features-1)/32 + 1));
			}

			unsigned int *indices = &X_indices[X_indptr[e]];
			int number_of_indices = X_indptr[e + 1] - X_indptr[e]; 

			for (int k = 0; k < number_of_indices; ++k) {
				int y = indices[k] / (dim_x*dim_z);
				int x = (indices[k] % (dim_x*dim_z)) / dim_z;
				int z = (indices[k] % (dim_x*dim_z)) % dim_z;

				for (int patch = index; patch < number_of_patches; patch += stride) {
					int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
					int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

					if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) || (x >= patch_coordinate_x + patch_dim_x)) {
						continue;
					}

					int p_y = y - patch_coordinate_y;
					int p_x = x - patch_coordinate_x;

					int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

					int chunk_nr = patch_pos / 32;
					int chunk_pos = patch_pos % 32;
					encoded_X[patch * number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos);

					if (append_negated) {
						int chunk_nr = (patch_pos + number_of_features) / 32;
						int chunk_pos = (patch_pos + number_of_features) % 32;
						encoded_X[patch * number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos);
					}
				}
		    }		
		}

		__global__ void encode_packed(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);
			int number_of_patch_chunks = (number_of_patches-1) / 32 + 1;

			int number_of_literals;
			if (append_negated) {
				number_of_literals = number_of_features*2;
			} else {
				number_of_literals = number_of_features;
			}
		
			unsigned int *indices = &X_indices[X_indptr[e]];
			int number_of_indices = X_indptr[e + 1] - X_indptr[e]; 

			for (int k = 0; k < number_of_indices; ++k) {
				int y = indices[k] / (dim_x*dim_z);
				int x = (indices[k] % (dim_x*dim_z)) / dim_z;
				int z = (indices[k] % (dim_x*dim_z)) % dim_z;

				for (int patch = index; patch < number_of_patches; patch += stride) {
					int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
					int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

					if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) || (x >= patch_coordinate_x + patch_dim_x)) {
						continue;
					}

					int chunk = patch / 32;
					int pos = patch % 32;

					int p_y = y - patch_coordinate_y;
					int p_x = x - patch_coordinate_x;

					int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

					encoded_X[chunk * number_of_literals + patch_pos] |= (1U << pos);

					if (append_negated) {
						encoded_X[chunk * number_of_literals + patch_pos + number_of_features] &= ~(1U << pos);
					}
				}
		    }		
		}

		__global__ void restore_packed(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int number_of_features = class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
			int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

			int number_of_literals;
			if (append_negated) {
				number_of_literals = number_of_features*2;
			} else {
				number_of_literals = number_of_features;
			}

			unsigned int *indices = &X_indices[X_indptr[e]];
			int number_of_indices = X_indptr[e + 1] - X_indptr[e]; 

			for (int k = 0; k < number_of_indices; ++k) {
				int y = indices[k] / (dim_x*dim_z);
				int x = (indices[k] % (dim_x*dim_z)) / dim_z;
				int z = (indices[k] % (dim_x*dim_z)) % dim_z;

				for (int patch = index; patch < number_of_patches; patch += stride) {
					int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
					int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

					if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) || (x >= patch_coordinate_x + patch_dim_x)) {
						continue;
					}

					int chunk = patch / 32;
					int pos = patch % 32;

					int p_y = y - patch_coordinate_y;
					int p_x = x - patch_coordinate_x;

					int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;

					encoded_X[chunk * number_of_literals + patch_pos] &= ~(1U << pos);

					if (append_negated) {
						encoded_X[chunk * number_of_literals + patch_pos + number_of_features] |= (1U << pos);
					}
				}
		    }		
		}

		__global__ void produce_autoencoder_example(
			curandState *state,
			unsigned int *active_output,
			int number_of_active_outputs,
			unsigned int *indptr_row,
			unsigned int *indices_row,
			int number_of_rows,
			unsigned int *indptr_col,
			unsigned int *indices_col,
			int number_of_cols,
			unsigned int *X,
			unsigned int *encoded_Y,
			int target,
			int accumulation,
			int T,
			int append_negated
		)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			if (index != 0) {
				return;
			}

			/* Copy state to local memory for efficiency */
	    	curandState localState = state[index];

			int row;

			int number_of_features = number_of_cols;
			int number_of_literals = 2*number_of_features;

			unsigned int number_of_literal_chunks = (number_of_literals-1)/32 + 1;

			// Initialize example vector X
			
			for (int k = 0; k < number_of_features; ++k) {
				int chunk_nr = k / 32;
				int chunk_pos = k % 32;
				X[chunk_nr] &= ~(1U << chunk_pos);
			}

			if (append_negated) {
				for (int k = number_of_features; k < number_of_literals; ++k) {
					int chunk_nr = k / 32;
					int chunk_pos = k % 32;
					X[chunk_nr] |= (1U << chunk_pos);
				}
			}

			if ((indptr_col[active_output[target]+1] - indptr_col[active_output[target]] == 0) || (indptr_col[active_output[target]+1] - indptr_col[active_output[target]] == number_of_rows)) {
				// If no positive/negative examples, produce a random example
				for (int a = 0; a < accumulation; ++a) {
					row = curand(&localState) % number_of_rows;
					for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
						int chunk_nr = indices_row[k] / 32;
						int chunk_pos = indices_row[k] % 32;
						X[chunk_nr] |= (1U << chunk_pos);

						if (append_negated) {
							chunk_nr = (indices_row[k] + number_of_features) / 32;
							chunk_pos = (indices_row[k] + number_of_features) % 32;
							X[chunk_nr] &= ~(1U << chunk_pos);
						}
					}
				}

				for (int i = 0; i < number_of_active_outputs; ++i) {
					if (i == target) {
						//int chunk_nr = active_output[i] / 32;
						//int chunk_pos = active_output[i] % 32;
						//X[chunk_nr] &= ~(1U << chunk_pos);

						encoded_Y[i] = T;
					} else {
						encoded_Y[i] = -T;
					}
				}

				state[index] = localState;

				return;
			}
		
			for (int a = 0; a < accumulation; ++a) {
				// Pick example randomly among positive examples
				int random_index = indptr_col[active_output[target]] + (curand(&localState) % (indptr_col[active_output[target]+1] - indptr_col[active_output[target]]));
				row = indices_col[random_index];
				
				for (int k = indptr_row[row]; k < indptr_row[row+1]; ++k) {
					int chunk_nr = indices_row[k] / 32;
					int chunk_pos = indices_row[k] % 32;
					X[chunk_nr] |= (1U << chunk_pos);

					if (append_negated) {
						chunk_nr = (indices_row[k] + number_of_features) / 32;
						chunk_pos = (indices_row[k] + number_of_features) % 32;
						X[chunk_nr] &= ~(1U << chunk_pos);
					}
				}
			}

			for (int i = 0; i < number_of_active_outputs; ++i) {
				if (i == target) {
					//int chunk_nr = active_output[i] / 32;
					//int chunk_pos = active_output[i] % 32;
					//X[chunk_nr] &= ~(1U << chunk_pos);

					encoded_Y[i] = T;
				} else {
					encoded_Y[i] = -T;
				}
			}
			
			state[index] = localState;
		}
	}
"""

code_transform = """
	extern "C"
    {
		// Transform examples
		__global__ void transform(unsigned int *global_ta_state, int *X, int *transformed_X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (int j = index; j < CLAUSES; j += stride) {
				unsigned int *ta_state = &global_ta_state[j*LA_CHUNKS*STATE_BITS];

				int all_exclude = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
						all_exclude = 0;
						break;
					}
				}

				if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
					all_exclude = 0;
				}

				if (all_exclude) {
					for (unsigned long long e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
						transformed_X[e*CLAUSES + j] = 0;
					}
					
					continue;
				}

				for (int e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
					int clause_output;
					for (int patch = 0; patch < PATCHES; ++patch) {
						clause_output = 1;
						for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
							if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
								clause_output = 0;
								break;
							}
						}

						if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
							clause_output = 0;
						}

						if (clause_output) {
							break;
						}
					}

					transformed_X[e*CLAUSES + j] = clause_output;
				}
			}
		}
	}
"""
