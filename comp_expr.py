"""
Compare different baised estimates for the softmax probability distribution.


"""

import numpy as np
from numpy.random import normal, choice
from biased_methods import *
import pdb

if __name__ == "__main__":

	# Parameters
	K = 10000
	W_p_vec = [3,10,30,100,300,1000]
	Q_uniform = np.array([1.0/K]*K)

	# Generate f values
	f = normal(0,1,K)
	f.sort()
	covar = np.exp(f)
	param = 1
	p_2 = 0.01
	W_p = K*p_2*2 # So that W_p>p_2*K in random_indices
	p_gradual = 6.0/8 # p = probability success for the geometric random variable. Note 1/2<=p<=3/4
	Q = unbiased_softmax("exact", param, covar, range(K), K, W_p, Q_uniform, p_gradual, p_2)
	#print Q

	random_indices = choice(range(K), np.ceil(W_p-p_2*K), p = Q) # np.ceil(W_p-p_2*K) gives equivalent workload to importance sampler
	#print random_indices

	rep = 10**4
	u = unbiased_softmax("exact", param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0]
	unbiased_importance_sample_vec = [unbiased_softmax("unbiased_importance_sample", param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0] for _ in xrange(rep)]
	print np.mean(unbiased_importance_sample_vec)/u
	print np.std(unbiased_importance_sample_vec)/u
	#np.mean([unbiased_softmax("importance_sample", param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0] for _ in range(1)])
	gradual_vec = [unbiased_softmax("gradual", param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0] for _ in xrange(rep)]
	print np.mean(gradual_vec)/u
	print np.std(gradual_vec)/u


