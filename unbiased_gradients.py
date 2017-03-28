"""
Biased methods for the softmax probability distribution

"""
import numpy as np
from numpy.random import random, choice, randint
import pdb


# Auxiliary methods

def sigma(x):
	# The binary logistic loss function
	return 1.0/(1+np.exp(-x))

# Biased methods

def NS(x,y,W,n_classes):
	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n_classes, replace = False) )%K 
	grad = - sigma(W[indices,:].dot(x))

	# Add y index and gradient
	indices =  np.append( indices , int(y) )
	grad = np.append( grad , sigma(W[y,:].dot(x)) )
	return [ indices , grad ]

def OVE(x,y,W,n_classes):
	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n_classes, replace = False) )%K 
	grad = - sigma( W[indices,:].dot(x) - W[y,:].dot(x) )

	# Add y index and gradient
	indices =  np.append( indices , int(y) )
	grad = np.append( grad , -sum(grad) )
	return [ indices , grad ]

# Expectation of biased methods

def expected_NS(x,y,W,n_classes):
	# The expectation of NS
	K = W.shape[0]
	grad = - sigma(W.dot(x)) / float(K) * n_classes
	grad[y] = sigma(W[y,:].dot(x))
	return grad

def expected_OVE(x,y,W,n_classes):
	K = W.shape[0]
	grad = - sigma( W.dot(x) - W[y,:].dot(x) ) / float(K) * n_classes
	grad[y] = 0
	grad[y] = -sum(grad)
	return grad

# Uniased methods

def exact(x,y,W):
	"""
	Calculate the exact softmax gradient
	Input:
		X:			Covariates. Matrix with shape (K,dim)
		y:			Class that was sampled. y is an element of {0,1,...,K-1}
		w:			Current weter value of the softmax

	Output:
		grad:		Indices with non-zero gradients and
	"""
	K = W.shape[0]
	grad = np.exp(W.dot(x)) 
	grad = -grad/sum(grad)
	grad[y] += 1
	return [ range(K) , grad ]

def DNS(x,y,W,work_frac,n_classes):
	K = W.shape[0]
	p_2 = work_frac - float(n_classes)/K 				# This is the p_2 for unbiasedness
	assert(p_2>0)
	if random() > p_2 :
		return NS(x,y,W,n_classes)
	else :
		indices_exact , grad_exact = exact(x,y,W)
		grad_biased = expected_NS(x,y,W,n_classes)
		grad = (grad_exact - grad_biased*(1-p_2) ) / p_2
		return [indices_exact , grad]
	
def DOVE(x,y,W,work_frac,n_classes):
	K = W.shape[0]
	p_2 = work_frac - float(n_classes)/K 				# This is the p_2 for unbiasedness
	assert(p_2>0)
	if random() > p_2 :
		return NS(x,y,W,n_classes)
	else :
		indices_exact , grad_exact = exact(x,y,W)
		grad_biased = expected_OVE(x,y,W,n_classes)
		grad = (grad_exact - grad_biased*(1-p_2) ) / p_2
		return [indices_exact , grad]





