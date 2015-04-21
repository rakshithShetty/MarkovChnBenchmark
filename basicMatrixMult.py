import argparse
from time import time
from random import random
#import numpy as np
"""
"""

def main_norm():
	
	for l in xrange(5):
		N = 10**l
		# Initialize the matrix
		A = [[random() for i in xrange(N)] for j in xrange(N)]

		# Initialize the state vector
		x = [random() for i in xrange(N)]
		
		y = [x[i] for i in xrange(N)]

		t1 = time()
		# Now lets do the repeated multiplication
		for t in xrange(1000):
			for i in xrange(N):
				y[i] = sum([ A[i][j] * x[j] for j in xrange(N)])
			
			x = [y[i] for i in xrange(N)]

		dt = time() - t1
		print('Time taken for N = %d is %f seconds' % (N,dt)) 

def main_sparse():
	from random import sample 
	
	for s in  [3]:#range(7):
		S = 2**s
		for l in xrange(8):
			N = 10**l
			if S > N:
				continue

			# Initialize the sparse matrix
			Aidx = [sample(range(N),S) for j in xrange(N)]
			A = [[random() for i in xrange(S)] for j in xrange(N)]

			# Initialize the state vector
			x = [random() for i in xrange(N)]
			
			y = [x[i] for i in xrange(N)]

			t1 = time()
			# Now lets do the repeated multiplication
			for t in xrange(1000):
				for i in xrange(N):
					y[i] = sum([ A[i][j] * x[Aidx[i][j]] for j in xrange(S)])
				
				x = [y[i] for i in xrange(N)]

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
