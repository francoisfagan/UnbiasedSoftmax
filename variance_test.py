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
    scale = 1
    dim = 20
    examples = 100
    repetitions = 10**2
    work_frac = 0.01

    W_p = int(K*work_frac)                      # Work per iteration
    assert(W_p>100)                             # So the importance sampling p_2 is valid

    w_star = np.ones(dim)                       # True w parameter
    methods = [ "unbiased_importance" , 
                "multilevel" , 
                "unbiased_negative_sampling" , 
                "unbiased_one_vs_each" ]
    method_variances_dict = { method: 0.0 for method in methods}

    for example in range(examples):
        if (example % int(examples/10)) == 0:
            print "Example: %d" % example
        X = randn(K,dim)                        # Generate covariates
        q = np.exp(X.dot(w_star))               # Class probability
        q = q/sum(q)

        """ Change indexing of X and q such that q is sorted from largest to smallest
         This is so the gradient methods can assume that the first indices are from the most common classes
         """
        sorted_indices = np.argsort(q)[::-1]
        q = q[sorted_indices]
        X = X[sorted_indices]
        
        y = choice(range(K), 1, p=q)[0]         # Sample from p
        w = (w_star + randn(1,dim))[0]          # Sample w
        
        # Run all of the methods
        exact_gradient = exact(X,y,w)           # Exact gradient
        for method in methods:
            variance_result = np.mean([ np.sqrt(sum((eval(method+"(X,y,w,W_p)")-exact_gradient)**2)) for _ in xrange(repetitions)], axis=0)
            method_variances_dict[method] += variance_result

    with open('variance_test_scale_%.2f_dim_%d_examples_%d_repetitions_%d_work_frac_%.3f'%(scale, dim, examples, repetitions, work_frac)+'.txt', 'w') as the_file:
        print ("\n".join(["%3.6f  " %(method_variances_dict[method]/examples) + method for method in methods]))
        #the_file.write("\n".join(["%3.6f  " %(method_variances_dict[method]/examples) + method for method in methods]))
    
    



