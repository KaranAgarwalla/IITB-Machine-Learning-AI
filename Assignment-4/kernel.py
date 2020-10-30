import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return X@Y.T
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	Z = np.sum(X**2, axis = 1, keepdims = True) + \
		np.sum(Y**2, axis = 1, keepdims = True).T	- \
		2*X@Y.T
	return np.exp(-Z/(2*sigma*sigma))
	# END TODO

def my_kernel(X,Y,sigma):
	'''Returns the gram matrix for your designed kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	# return (1+X@Y.T)**4 
	return np.power(1+X@Y.T, 4) + np.power(1+X@Y.T, 2)
	# END TODO