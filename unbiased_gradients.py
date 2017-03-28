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

# def sample_Q(q_cum_sum,n_samples,y=-1):
# 	"""Sample from the distribution Q given by its cumulative sum.
# 	If y!=-1 then do not sample y.
# 	"""
# 	if y!=-1:
# 		q_cum_sum = 
# 	return 

# Biased methods

def importance_sampling(x,y,W,n_classes):
	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n_classes, replace = False) )%K 	# Generate numbers without repetition excluding y
	indices = np.append( indices , int(y) )

	# Calculate gradient
	grad = np.exp(W[indices,:].dot(x))
	grad = - grad/np.sum(grad)	
	grad[n_classes] += 1

	return [ indices , grad ]

def negative_sampling(x,y,W,n_classes):
	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n_classes, replace = False) )%K 
	grad = - sigma(W[indices,:].dot(x))

	# Add y index and gradient
	indices =  np.append( indices , int(y) )
	grad = np.append( grad , sigma(W[y,:].dot(x)) )
	return [ indices , grad ]

def one_vs_each(x,y,W,n_classes):
	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n_classes, replace = False) )%K 
	grad = - sigma( W[indices,:].dot(x) - W[y,:].dot(x) )

	# Add y index and gradient
	indices =  np.append( indices , int(y) )
	grad = np.append( grad , -sum(grad) )
	return [ indices , grad ]

# Expectation of biased methods

def expected_negative_sampling(x,y,W,n_classes):
	# The expectation of negative_sampling
	K = W.shape[0]
	grad = - sigma(W.dot(x)) / float(K) * n_classes
	grad[y] = sigma(W[y,:].dot(x))
	return grad

def expected_one_vs_each(x,y,W,n_classes):
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

def unbiased_importance_sampling(x,y,W,work_frac,n_classes):
	K = W.shape[0]
	p_2 = work_frac - float(n_classes)/K 				# This is the p_2 for unbiasedness
	assert(p_2>0)
	indices_biased , grad_biased = importance_sampling(x,y,W,n_classes)
	if random() > p_2 :
		return [indices_biased , grad_biased]#grad_biased[np.argsort(indices_biased)]#
	else :
		indices_exact , grad_exact = exact(x,y,W)
		grad_exact /= p_2
		grad_exact[indices_biased] -= grad_biased*(1-p_2)/p_2
		return [indices_exact , grad_exact]

def unbiased_negative_sampling(x,y,W,work_frac,n_classes):
	K = W.shape[0]
	p_2 = work_frac - float(n_classes)/K 				# This is the p_2 for unbiasedness
	assert(p_2>0)
	if random() > p_2 :
		return negative_sampling(x,y,W,n_classes)
	else :
		indices_exact , grad_exact = exact(x,y,W)
		grad_biased = expected_negative_sampling(x,y,W,n_classes)
		grad = (grad_exact - grad_biased*(1-p_2) ) / p_2
		return [indices_exact , grad]
	
def unbiased_one_vs_each(x,y,W,work_frac,n_classes):
	K = W.shape[0]
	p_2 = work_frac - float(n_classes)/K 				# This is the p_2 for unbiasedness
	assert(p_2>0)
	if random() > p_2 :
		return negative_sampling(x,y,W,n_classes)
	else :
		indices_exact , grad_exact = exact(x,y,W)
		grad_biased = expected_one_vs_each(x,y,W,n_classes)
		grad = (grad_exact - grad_biased*(1-p_2) ) / p_2
		return [indices_exact , grad]





