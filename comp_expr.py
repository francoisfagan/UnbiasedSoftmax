"""
Compare different baised estimates for the softmax probability distribution.


"""

import numpy as np
from numpy.random import normal, choice
from biased_methods import *
import pdb

if __name__ == "__main__":

	# Parameters
	K = 100
	W_p_vec = [3,10,30,100,300,1000]
	Q_uniform = np.array([1.0/K]*K)

	# Generate f values
	f = normal(0,1,K)
	f.sort()
	covar = np.exp(f)
	param = 1
	p_2 = 0.1
	W_p = K*p_2*2 # So that W_p>p_2*K in random_indices
	p_gradual = 5.0/8 # p = probability success for the geometric random variable. Note 1/2<=p<=3/4
	Q = unbiased_softmax("exact", param, covar, range(K), K, W_p, Q_uniform, p_gradual, p_2)
	#print Q

	
	#print random_indices

	rep = 10**4
	u = unbiased_softmax("exact", param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0]
	for method in ["unbiased_importance_sample" , "gradual" , "unbiased_one_vs_each"]:
		method_vec = [unbiased_softmax(method, param, covar, random_indices, K, W_p, Q_uniform, p_gradual, p_2)[0] for _ in xrange(rep)]
		print np.mean(method_vec)/u
		print np.std(method_vec)/u


