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
    K = 10**5
    scale = 10
    dim = 2

    # Hyperparameters for gradual method
    p = 5.0/8
    W_p = int(K/1000.0) 

    base = int(W_p * (2*p-1) / (2*p))
    # Expected work per iteration. We will use this as a baseline for all other methods.
    

    # Hyperparameters for importance sampling method
    p_2 = float(base)/K # Note this satisfies assert(p_2 < W_p/K) that will be needed for the assignment of n_samples in the next line
    n_samples = int(W_p - p_2*K)

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
    
    # Exact gradient
    rep = 10**2
    exact = exact_gradient(X,y,w)
    #print "%s Exact" %str(exact_gradient(X,y,w))
    #print "%s Importance sample estimate" %str(np.mean([importance_gradient(X,y,w,q,n_samples) for _ in xrange(rep)],axis=0))
    #print "%s One_vs_each estimate" %str(np.mean([one_vs_each_gradient(X,y,w,q_without_y,n_samples) for _ in xrange(rep)],axis=0))
    for method in [ "unbiased_importance_gradient_deterministic(X,y,w,q,p_2)" ,
                    "gradual_gradient(X,y,w,q,p,base)" ,  
                    "unbiased_negative_sampling_gradient(X,y,w,q,p_2)" , 
                    "unbiased_one_vs_each_gradient(X,y,w,q_without_y,p_2)"]:
        print method[:16] + " %s" %str(np.mean([ [np.sqrt(sum((eval(method)-exact)**2))/np.sqrt(sum(exact**2))
                                                    , np.sqrt(sum((eval(method)-exact)**2))] for _ in xrange(rep)],axis=0))
    #g = exact_gradient(X,y,w)



