import argparse
from time import time
from random import random
import numpy as np

"""
"""

def main_norm():
	import theano
	from theano import config
	import theano.tensor as tensor

	A = tensor.matrix( dtype=config.floatX)
	x = tensor.vector(dtype=config.floatX)

	rval, updates = theano.scan(fn=lambda x, A:tensor.dot(A,x), outputs_info = x,non_sequences=A, name='Markv_chn', n_steps=1000)

	final_y = rval[-1]

	mark_chn = theano.function(inputs=[A,x], outputs=final_y, updates=updates)
	
	for l in xrange(5):
		N = 10**l
		# Initialize the matrix
		A = np.array(np.random.rand(N,N),dtype=config.floatX) 

		# Initialize the state vector
		x = np.array(np.random.rand(N),dtype=config.floatX) 
		
		t1 = time()
		# Now lets do the repeated multiplication
		y = mark_chn(A,x)
		dt = time() - t1
		print('Time taken for N = %d is %f seconds' % (N,dt)) 

def main_sparse():
	import scipy.sparse as sp
	import theano
	from theano import config
	import theano.tensor as tensor
	from theano import sparse

	A = sparse.csr_matrix(dtype=config.floatX)
	x = tensor.matrix(dtype=config.floatX)

	rval, updates = theano.scan(fn=lambda x, A: sparse.basic.structured_dot(A,x), outputs_info = x,non_sequences=A, name='Markv_chn', n_steps=1000)

	final_y = rval[-1]

	mark_chn = theano.function(inputs=[A,x], outputs=final_y, updates=updates)
	
	for s in range(7):
		S = 2**s
		for l in xrange(7):
			N = 10**l
			# Initialize the matrix
			indices = np.empty(N*S)
			for j in xrange(S):
				indices[j*N:(j+1)*N] = np.random.permutation(N)

			indptr = np.array(range(N+1))*S
			A = sp.csr_matrix((np.random.rand(S*N), indices, indptr), shape=(N, N),dtype=config.floatX)

			# Initialize the state vector
			x = np.array(np.random.rand(N,1),dtype=config.floatX) 
			
			t1 = time()
			# Now lets do the repeated multiplication
			y = mark_chn(A,x)
			dt = time() - t1
			print('Time taken for S = %d and N = %d is %f seconds' % (S, N,dt)) 

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--sparse_or_dense', type=str, default='dense', help='Should we run sparse or dense version?')
	
	args = parser.parse_args()
	params = vars(args)

	if(params['sparse_or_dense']  == 'sparse'): 
		main_sparse()
	else:
		main_norm()
