# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import numpy as np
from numpy.random import randn, choice
from biased_methods import *


if __name__ == "__main__":

    # General Hyperparameters
    K = 10**3
    scale = 10
    dim = 2

    # Hyperparameters for gradual method
    p = 5.0/8
    base = int(K/10.0)
    assert(base>0)
    # Expected work per iteration. We will use this as a baseline for all other methods.
    W_p = base * (2*p)/(2*p-1) 

    # Hyperparameters for importance sampling method
    p_2 = float(base)/K # Note this satisfies assert(p_2 < W_p/K) that will be needed for the assignment of n_samples in the next line
    n_samples = int(W_p - p_2*K)

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

    # Define q_without_y for the one_vs_each method
    q_without_y = np.array(q, copy=True) 
    q_without_y[y] = 0.0
    q_without_y = q_without_y/sum(q_without_y)
    
    # Exact gradient
    rep = 10**2
    exact = exact_gradient(X,y,w)
    #print "%s Exact" %str(exact_gradient(X,y,w))
    #print "%s Importance sample estimate" %str(np.mean([importance_gradient(X,y,w,q,n_samples) for _ in xrange(rep)],axis=0))
    #print "%s One_vs_each estimate" %str(np.mean([one_vs_each_gradient(X,y,w,q_without_y,n_samples) for _ in xrange(rep)],axis=0))
    for method in ["gradual_gradient(X,y,w,q,p,base)" , 
                    "unbiased_importance_gradient(X,y,w,q,n_samples,p_2)" ,
                    "unbiased_importance_gradient_deterministic(X,y,w,q,n_samples,p_2)" , 
                    "unbiased_one_vs_each_gradient(X,y,w,q_without_y,n_samples,p_2)"]:
        print method[:16] + " %s" %str(np.mean([ sum((eval(method)-exact)**2)/sum(exact**2) for _ in xrange(rep)],axis=0))
    #g = exact_gradient(X,y,w)



