"""
Biased methods for the softmax probability distribution

"""
import numpy as np
from numpy.random import random, choice
import pdb

def truncated_geometric_pmf(p,trunc,j):
	# PMF of geometric variable with values in 1,...,trunc
	return p*(1- p)**(j-1) / (1-(1- p)**trunc)


def truncated_geometric_sample(p,trunc):
	# Sample geometric variable with values in 1,...,trunc
	#return choice(range(1,int(trunc)+1), 1, p=[truncated_geometric_pmf(p,trunc,j) for j in xrange(1,int(trunc)+1)])[0]
	return np.ceil( np.log(1 - random()*(1-(1-p)**trunc )) / np.log(1-p) )


def unbiased_softmax(name, param, covar, indices, K, W_p, Q, p, p_2):

	"""
	Input:
		parameters
		covariates
		indices:	return approximate softmax probabilities for these indices

	Output:
		(Approximate) softmax probabilies at the specified indices

	"""

	if name == "exact":
		vec = covar.dot(param)
		return (vec/np.sum(vec))[indices]

	elif name == "importance_sample":
		vec = covar[indices].dot(param) / Q[indices]
		return vec/np.sum(vec)

	elif name == "unbiased_importance_sample":
		vec = covar[indices].dot(param) / Q[indices]
		if random() > p_2:
			return vec/np.sum(vec)
		else :
			return (unbiased_softmax("exact", param, covar, indices, K, W_p, Q, p, p_2) - (1-p_2)*vec/np.sum(vec))/p_2


	elif name == "gradual":

		# The base number of samples in the geometric sequence so the average number of samples is W_p
		# Note this calculation is approximate and is only accurate for large K
		base = np.ceil(W_p* (1-1/(2*p)) ) 

		# Maximum value of the geometric random variable
		J_max = np.ceil(np.log2(K/base))
		
		# Sample from truncated geometric distribution with parameter p
		J = truncated_geometric_sample(p,J_max)

		# Calculate approx denomenator
		partial_sum_1 = np.sum(covar[:base].dot(param))
		partial_sum_2 = partial_sum_1 + np.sum(covar[base: min(base*2**(J-1) , K) ].dot(param))
		partial_sum_3 = partial_sum_2 + np.sum(covar[min(base*2**(J-1) , K) : min(base*2**J , K) ].dot(param))

		R = (base/partial_sum_1 
			+ (min(base*2**J , K)/partial_sum_3	- min(base*2**(J-1) , K)/partial_sum_2 ) 
			/ truncated_geometric_pmf(p,J_max,J) ) / K
			

		# Return softmax approximation
		vec = covar[indices].dot(param)

		return vec*R



		#print "Difference"
		#print ((1/np.sum(covar.dot(param)) - R_cum) / (1/np.sum(covar.dot(param))))

		# R_cum = 0
		# rep = 1000
		# J_samples = [truncated_geometric_sample(p,J_max) for _ in xrange(rep)]
		# emp_freq = [J_samples.count(J) /float(rep) for J in xrange(1,int(J_max)+1)]
		# print emp_freq
		# print [truncated_geometric_pmf(p,J_max,J) for J in xrange(1,int(J_max)+1)]

		# for J in xrange(1,int(J_max)+1):

		# 	# Calculate approx denomenator
		# 	partial_sum_1 = np.sum(covar[:base].dot(param))
		# 	partial_sum_2 = partial_sum_1 + np.sum(covar[base: min(base*2**(J-1) , K) ].dot(param))
		# 	partial_sum_3 = partial_sum_2 + np.sum(covar[min(base*2**(J-1) , K) : min(base*2**J , K) ].dot(param))
		# 	R = (base/partial_sum_1 
		# 		+ (min(base*2**J , K)/partial_sum_3	- min(base*2**(J-1) , K)/partial_sum_2 ) 
		# 		/ truncated_geometric_pmf(p,J_max,J) ) / K

		# 	R_cum +=  R*emp_freq[J-1]#truncated_geometric_pmf(p,J_max,J)#


		# R_cum = base/np.sum(covar[:base].dot(param))/K
		# for J in xrange(1,int(J_max)+1):

		# 	# Calculate approx denomenator
		# 	partial_sum_1 = np.sum(covar[:base].dot(param))
		# 	partial_sum_2 = partial_sum_1 + np.sum(covar[base: min(base*2**(J-1) , K) ].dot(param))
		# 	partial_sum_3 = partial_sum_2 + np.sum(covar[min(base*2**(J-1) , K) : min(base*2**J , K) ].dot(param))
		# 	R = (base/partial_sum_1 
		# 		+ (min(base*2**J , K)/partial_sum_3	- min(base*2**(J-1) , K)/partial_sum_2 ) 
		# 		/ truncated_geometric_pmf(p,J_max,J) ) / K
		# 	#R_cum += R*truncated_geometric_pmf(p,J_max,J)
		# 	#pdb.set_trace()

		# 	R_cum +=  (min(base*2**J , K)/partial_sum_3	- min(base*2**(J-1) , K)/partial_sum_2 )/K


