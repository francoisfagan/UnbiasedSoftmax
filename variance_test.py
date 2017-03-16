# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import numpy as np
from numpy.random import randn, choice
from biased_methods import *


if __name__ == "__main__":

    # Hyperparameters
    K = 3#10**1
    scale = 10
    dim = 2


    #covar = np.exp(f)
    #param = 1
    #p_2 = 0.1
    #W_p = K*p_2*2 # So that W_p>p_2*K in random_indices
    #p_gradual = 5.0/8 # p = probability success for the geometric random variable. Note 1/2<=p<=3/4
    #Q = unbiased_softmax("exact", param, covar, range(K), K, W_p, Q_uniform, p_gradual, p_2)

    # True w parameter
    w_star = np.ones(dim)

    # Generate covariates
    X = randn(K,dim)

    # Class probability
    q = np.exp(X.dot(w_star))
    q = q/sum(q)

    # Sample from p
    y = choice(range(K), 1, p=q)[0]

    # Sample w
    w = (w_star + randn(1,dim))[0]

     # n_samples = np.ceil(W_p-p_2*K) gives equivalent workload to importance sampler
    
    # Exact gradient
    vec = np.exp(X.dot(w)) 
    print X[y] - vec.dot(X)/np.sum(vec)
    #g = exact_gradient(X,y,w)



