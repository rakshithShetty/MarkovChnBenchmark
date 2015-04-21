This code benchmarks a simple markov chain implementation sparse and dense transistion matrices using 
1) Native Python
2) Numpy
3) Theano (both cpu and gpu)

To run the basic tests you can just call 
	python <filename> 
To run the sparse tests you can call
	python <filename> -m sparse 


For theano version make sure theano is installed. 
If you want to run the test on GPU make sure GPU/CUDA is availaible and device is set to 'gpu' in ~/.theaonrc file
