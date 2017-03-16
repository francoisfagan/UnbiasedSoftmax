"""
Biased methods for the softmax probability distribution

"""
import numpy as np
from numpy.random import random, choice
import pdb

def truncated_geometric_pmf(p,trunc,j):
	# PMF of geometric variable with values in 1,...,trunc
	return p*(1- p)**(j-1) / (1-(1- p)**trunc)

	"""
	As a check of correctness, note that the probability of j is proportional to p*(1- p)**(j-1), 
	as it should be, and that j takes its minimum value j=1 with probability prortional to p.
	Finally, the normalization constant can be checked by confirming 
	1 = sum([truncated_geometric_pmf(p,trunc,j) for j in xrange(1, trunc+1)])
	"""

def truncated_geometric_sample(p,trunc):
	# Sample geometric variable with values in 1,...,trunc
	return np.ceil( np.log(1 - random()*(1-(1-p)**trunc )) / np.log(1-p) )


	"""
	An easy alternative is to sample from
	choice(range(1,int(trunc)+1), 1, p=[truncated_geometric_pmf(p,trunc,j) for j in xrange(1,int(trunc)+1)])[0]
	As a check of correctness, we can compare the empirical distribution of truncated_geometric_sample 
	to the pmf of truncated_geometric_pmf as follows:
	rep = 10**4
	samples = [truncated_geometric_sample(p,trunc) for _ in xrange(rep)]
	difference = sum([ abs(truncated_geometric_pmf(p,trunc,j)-float(samples.count(j))/rep) for j in xrange(1, trunc+1)])
	"""

def exact_gradient(X,y,w):
	"""
	Calculate the exact softmax gradient
	Input:
		X:			Covariates. Matrix with shape (K,dim)
		y:			Class that was sampled. y is an element of {0,1,...,K-1}
		w:			Current weter value of the softmax

	Output:
		Exact gradient of the softmax
	"""
	vec = np.exp(X.dot(w)) 
	return X[y] - vec.dot(X)/np.sum(vec)


def importance_gradient(X,y,w,q,n_samples):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w:		As for the exact_gradient function
		q:			Sampling probabilities of the indices. q[i] = Prob(y=i)
		n_samples:	Number of indices to sample

	Output:
		Importance sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**2
	print "Importance sample estimate: %s" %str(np.mean([importance_gradient(X,y,w,q,n_samples) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	K = X.shape[0]
	indices = choice(range(K), n_samples, p = q)
	vec = np.exp(X[indices].dot(w)) / q[indices]
	return X[y] - vec.dot(X[indices])/np.sum(vec)


def one_vs_each_gradient(X,y,w,q_without_y,n_samples):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w:			As for the exact_gradient function
		q_without_y:	Sampling probabilities of the indices excluding y.
							q_without_y[i] ~= Prob(y=i), q_without_y[y]=0
		n_samples:		Number of indices to sample

	Output:
		One-vs-each sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	q_without_y = np.array(q, copy=True) 
	q_without_y[y] = 0.0
	q_without_y = q_without_y/sum(q_without_y)
	rep = 10**2
	print "One-vs-each sample estimate: %s" %str(np.mean([one_vs_each_gradient(X,y,w,q_without_y,n_samples) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	K = X.shape[0]
	assert(q_without_y[y]==0.0)
	indices = choice(range(K), n_samples, p = q_without_y)
	return (1/(1+np.exp(X[y].dot(w))-X[indices].dot(w))).dot(X[y]-X[indices]) 


def gradual_gradient(X,y,w,q,p,base):
	# base = np.ceil(W_p* (1-1/(2*p)) ) 
	# The base number of samples in the geometric sequence so the average number of samples is W_p
	# Note this calculation is approximate and is only accurate for large K
	
	K = X.shape[0]

	# Maximum value of the geometric random variable
	J_max = np.ceil(np.log2(K/base))
	
	# Sample from truncated geometric distribution with weter p
	J = truncated_geometric_sample(p,J_max)

	# Calculate dot products to be used in the estimate R and also for gradient steps
	partial_dot_1 = np.exp(X[:base].dot(w))
	partial_dot_2 = np.exp(X[base: min(base*2**(J-1) , K) ].dot(w))
	partial_dot_3 = np.exp(X[min(base*2**(J-1) , K) : min(base*2**J , K) ].dot(w))

	# Calculate approx denomenator
	partial_sum_1 = np.sum(partial_dot_1)
	partial_sum_2 = partial_sum_1 + np.sum(partial_dot_2)
	partial_sum_3 = partial_sum_2 + np.sum(partial_dot_3)

	R = (base/partial_sum_1 
		+ (min(base*2**J , K)/partial_sum_3	- min(base*2**(J-1) , K)/partial_sum_2 ) 
		/ truncated_geometric_pmf(p,J_max,J) ) / K

	# Return softmax approximation
	#vec = 
	return X[y] - R*np.exp(X.dot(w)).dot(X)



def unbiased_importance_gradient(X,y,w,q,n_samples,p_2):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w,q, n_samples:		As for the importance_gradient function
		p_2:					Probability of sampling exact gradient

	Output:
		Unbiased importance sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**5
	print "Unbiased importance sample estimate: %s" %str(np.mean([unbiased_importance_gradient(X,y,w,q,n_samples,p_2) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	R = importance_gradient(X,y,w,q,n_samples)
	if random() > p_2 :  return R
	else : return (exact_gradient(X,y,w) - (1-p_2)*R)/p_2




def unbiased_one_vs_each_gradient(X,y,w,q_without_y,n_samples,p_2):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w, q_without_y, n_samples:		As for the one_vs_each_gradient function
		p_2:					Probability of sampling exact gradient

	Output:
		Unbiased one-vs-each sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	q_without_y = np.array(q, copy=True) 
	q_without_y[y] = 0.0
	q_without_y = q_without_y/sum(q_without_y)
	rep = 10**2
	print "Unbiased one-vs-each sample estimate: %s" %str(np.mean([unbiased_one_vs_each_gradient(X,y,w,q_without_y,n_samples,p_2) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	R = one_vs_each_gradient(X,y,w,q_without_y,n_samples)
	if random() > p_2 :  return R
	else : return (exact_gradient(X,y,w) - (1-p_2)*R)/p_2





