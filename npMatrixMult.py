import argparse
from time import time
from random import random
import numpy as np
"""
"""

def main_norm():
	
	for l in xrange(7):
		N = 10**l
		# Initialize the matrix
		A = np.array(np.random.rand(N,N), dtype = 'float32')

		# Initialize the state vector
		x =  np.array(np.random.rand(N,1) , dtype = 'float32')
		
		t1 = time()
		# Now lets do the repeated multiplication
		for t in xrange(1000):
			y = A.dot(x) 
			x = y.copy() 

		dt = time() - t1
		print('Time taken for N = %d is %f seconds' % (N,dt)) 

def main_sparse():
	from random import sample 
	import scipy.sparse as sp

	for s in range(7):
		S = 2**s
		for l in xrange(7):
			N = 10**l
			if S > N:
				continue

			# Initialize the sparse matrix
			indices = np.empty(N*S)
			# This initialization takes too long! 
			#for j in xrange(N):
			#	indices[j*S:(j+1)*S] = np.random.choice(N,S, replace=False)

			# Instead do this although technically a bit restrictive on type of sparse matrices
			# should be ok for our study
			for j in xrange(S):
				indices[j*N:(j+1)*N] = np.random.permutation(N)
			

			indptr = np.array(range(N+1))*S
			A = sp.csr_matrix((np.array(np.random.rand(S*N),dtype='float32'), indices, indptr), shape=(N, N))
			# Initialize the state vector
			x =  np.array(np.random.rand(N,1) , dtype = 'float32')
			
			t1 = time()
			# Now lets do the repeated multiplication
			for t in xrange(1000):
				y = A.dot(x) 
				x = y.copy() 

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
