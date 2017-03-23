"""
Biased methods for the softmax probability distribution

"""
import numpy as np
from numpy.random import random, choice
import pdb


# Auxiliary methods

def sigma(x):
	# The binary logistic loss function
	return 1.0/(1+np.exp(-x))

def truncated_geometric_pmf(p,trunc,j):
	# PMF of geometric variable with values in 1,...,trunc
	return p*(1- p)**(j-1) / (1-(1- p)**trunc)

	"""
	As a check of correctness, note that the probability of j is proportional to p*(1- p)**(j-1), 
	as it should be, and that j takes its minimum value j=1 with probability prortional to p.
	Finally, the normalization constant can be checked by confirming 
	1 = sum([truncated_geometric_pmf(p,trunc,j) for j in xrange(1, trunc+1)])
	"""

def truncated_geometric_cdf(p,trunc,j):
	# CDF of geometric variable with values in 1,...,trunc
	return (1-(1- p)**j)/ (1-(1- p)**trunc)

def truncated_geometric_sample(p,trunc):
	# Sample geometric variable with values in 1,...,trunc
	return int(np.ceil( np.log(1 - random()*(1-(1-p)**trunc )) / np.log(1-p) ))


	"""
	An easy alternative is to sample from
	choice(range(1,int(trunc)+1), 1, p=[truncated_geometric_pmf(p,trunc,j) for j in xrange(1,int(trunc)+1)])[0]
	As a check of correctness, we can compare the empirical distribution of truncated_geometric_sample 
	to the pmf of truncated_geometric_pmf as follows:
	rep = 10**4
	samples = [truncated_geometric_sample(p,trunc) for _ in xrange(rep)]
	difference = sum([ abs(truncated_geometric_pmf(p,trunc,j)-float(samples.count(j))/rep) for j in xrange(1, trunc+1)])
	"""

# Biased methods

def exact(X,y,w):
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

def importance(X,y,w,sample_range=xrange(100)):
	"""
	Calculate the softmax gradient using the importance sampling method deterministically.
	This differs from the importance method in that the indices are not random, but the top n_samples
	The benefit is that this avoids the need for random sampling (slowing down the algorithm)
	and also should decrease the variance as the most important classes are always evaluated
	Input:
		X,y,w:		As for the exact function
		q:			Sampling probabilities of the indices. q[i] = Prob(y=i)
		n_samples:	Number of indices to sample

	Output:
		Importance sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**2
	print "Importance sample estimate: %s" %str(np.mean([importance(X,y,w,q,n_samples) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact(X,y,w))
	"""
	vec = np.exp(X[sample_range].dot(w))
	return vec.dot(X[sample_range])/np.sum(vec)

def negative_sampling(X,y,w,n_samples=5):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w:		As for the exact function
		q:			Sampling probabilities of the indices. q[i] = Prob(y=i)
		n_samples:	Number of indices to sample

	Output:
		Negative sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**2
	print "Negative sample estimate: %s" %str(np.mean([negative_sampling(X,y,w,q) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact(X,y,w))
	"""
	K = X.shape[0]
	indices = range(n_samples)#choice(range(K), n_samples, p = q)
	vec = sigma(X[indices].dot(w)) #/ n_samples * K #/ q[indices]
	return sigma(-X[y].dot(w))*X[y] - vec.dot(X[indices])

def one_vs_each(X,y,w,n_samples=5):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w:			As for the exact function
		n_samples:		Number of indices to sample

	Output:
		One-vs-each sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	q_without_y = np.array(q, copy=True) 
	q_without_y[y] = 0.0
	q_without_y = q_without_y/sum(q_without_y)
	rep = 10**2
	print "One-vs-each sample estimate: %s" %str(np.mean([one_vs_each(X,y,w,q_without_y,n_samples) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact(X,y,w))
	"""
	#K = X.shape[0]
	#assert(q_without_y[y]==0.0)
	# Correct way: indices = choice(range(K), n_samples, p = q_without_y)
	# Correct way: ret = np.multiply( sigma( X[y].dot(w)-X[indices].dot(w) ) , 1.0/q_without_y[indices] ).dot(X[y]-X[indices]) / n_samples
	ret = sigma( X[y].dot(w)-X[:n_samples].dot(w) ).dot(X[y]-X[:n_samples])
	return ret

# Unbiased methods

def multilevel(X,y,w,W_p,p = 1-2**(-3.0/2)):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w,q:		As for the importance function.
						Note that q should be ordered as a decreasing sequence for low variance (but will work either way)
		p:				Geometric parameter
		base:			Number of samples for the base: R_1.
						base = np.ceil(W_p* (1-1/(2*p)) ) ensures that the expected number of samples is W_p 


	Output:
		Gradual gradient sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**2
	print "Gradual sample estimate: %s" %str(np.mean([multilevel2(X,y,w,W_p) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact(X,y,w))
	"""
	# Note this calculation is approximate and is only accurate for large K
	base = int(W_p * (2*p-1) / (2*p))
	K = X.shape[0]
	J_max = np.ceil(np.log2(float(K)/base))		# Maximum value of the geometric random variable
	J = truncated_geometric_sample(p,J_max)		# Sample from truncated geometric distribution with weter p
	R_values = [importance(X,y,w,sample_range=xrange(int(min(base*2**j , K)) )) for j in xrange(J+1)]
	R = R_values[0] + sum([(R_values[j] - R_values[j-1] ) / (1-truncated_geometric_cdf(p,J_max,j-1) ) for j in xrange(1,J+1)])	
	return X[y] - R

def unbiased_importance(X,y,w,W_p,n_samples=100):
	"""
	Calculate the softmax gradient using the deterministic importance sampling method
	Input:
		X,y,w:					As for the exact function
		W_p:					Expected work per iteration
		n_samples:				Number of importance samples

	Output:
		Unbiased deterministic importance sampling estimate for the gradient of the softmax
	"""
	K = X.shape[0]
	p_2 = float(W_p - n_samples)/K 				# This is the p_2 for unbiasedness
	R = X[y] - importance(X,y,w,xrange(n_samples))	# Note the "X[y] - "
	if random() > p_2 :  return R
	else : return (exact(X,y,w) - (1-p_2)*R)/p_2

def unbiased_negative_sampling(X,y,w,W_p,n_samples=5):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		As for unbiased_importance
	Output:
		Unbiased negative sampling estimate for the gradient of the softmax
	"""
	K = X.shape[0]
	p_2 = float(W_p - n_samples)/K 				# This is the p_2 for unbiasedness
	R = negative_sampling(X,y,w,n_samples)
	if random() > p_2 :  return R
	else : return (exact(X,y,w) - (1-p_2)*R)/p_2

def unbiased_one_vs_each(X,y,w,W_p,n_samples=5):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		As for unbiased_importance
	Output:
		Unbiased one-vs-each sampling estimate for the gradient of the softmax
	"""
	K = X.shape[0]
	p_2 = float(W_p - n_samples)/K 				# This is the p_2 for unbiasedness
	R = one_vs_each(X,y,w,n_samples)
	if random() > p_2 :  return R
	else : return (exact(X,y,w) - (1-p_2)*R)/p_2

