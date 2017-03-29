"""
Biased methods for the softmax probability distribution

"""
from numpy.random import random, choice, randint
from numpy.linalg import norm
import numpy as np
from math import sqrt
import pdb


# Auxiliary methods

def sigma(x):
	# The binary logistic loss function
	return 1.0/(1+np.exp(-x))

# Methods to calculate gradients

def EXACT_gradient(x,y,W):
	"""
	Calculate the EXACT softmax gradient
	Input:
		X:			Covariates. Matrix with shape (K,dim)
		y:			Class that was sampled. y is an element of {0,1,...,K-1}
		w:			Current weter value of the softmax

	Output:
		grad:		Indices with non-zero gradients and
	"""
	K 		= W.shape[0]
	grad 	= np.exp(W.dot(x)) 
	grad 	= -grad/sum(grad)
	grad[y] += 1
	return [ range(K) , grad , K ]

def NS_gradient(x,y,W,n):#_star
	# Negative sampling gradient

	K 		= W.shape[0]

	# Sample indices and calculate gradient
	# if n_star == np.floor(n_star):
	# 	n = n_star
	# else:
	# 	n = np.ceil(n_star) if random() < (n_star - np.floor(n_star)) else np.floor(n_star)
	
	indices = (int(y)+1+choice(K-1, n, replace = False) )%K 
	grad 	= - sigma(W[indices,:].dot(x))

	# Add y index and gradient
	indices = np.append( indices , int(y) )
	grad 	= np.append( grad , sigma(W[y,:].dot(x)) )
	return [ indices , grad , n ]

def OVE_gradient(x,y,W,n):
	# One-vs-each gradient

	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n, replace = False) )%K 
	grad 	= - sigma( W[indices,:].dot(x) - W[y,:].dot(x) ) / n #* (K-1)

	# Add y index and gradient
	indices =  np.append( indices , int(y) )
	grad 	= np.append( grad , -sum(grad) )
	return [ indices , grad , n ]

# Methods to calculate Rao-Blackwellized gradients

def RB_NS_gradient(x,y,W,n):
	# Rao-Blackwellized negative sampling gradient
	K 		= W.shape[0]
	grad 	= - n / float(K-1) * sigma(W.dot(x))
	grad[y] = sigma(W[y,:].dot(x))
	return grad

def RB_OVE_gradient(x,y,W,n):
	# Rao-Blackwellized one-vs-each gradient
	K 		= W.shape[0]
	grad 	= - sigma( W.dot(x) - W[y,:].dot(x) )  / (K-1)
	grad[y] = 0
	grad[y] = - sum(grad)
	#pdb.set_trace()
	return grad

""" Classes for gradients. 
These store variables like the number of samples n and debias sampling probability p2 
as well as run the gradient methods. """

class gradient:
	# Base class that necesitates that children classes support the calculate_gradient function
	def calculate_gradient(self):
		pass

# Basic gradient classes

class EXACT(gradient):
	def calculate_gradient(self, x,y,W):
		return EXACT_gradient(x,y,W)

class NS(gradient):
	# Negative sampling class
	def __init__(self, n):
		self.n = n

	def calculate_gradient(self, x,y,W):
		return NS_gradient(x,y,W,self.n)

class OVE(gradient):
	# One-vs-each class
	def __init__(self, n):
		self.n = n

	def calculate_gradient(self, x,y,W):
		return OVE_gradient(x,y,W,self.n)

# Debiased gradient classes

class DNS(gradient):
	def __init__(self, n, K, p2_scale):
		self.n = n
		self.K = K
		self.p2 = p2_scale*sqrt(float(n)/K)

		# self.n_numerator = 0
		# self.n_denominator = 0

		# # Variables for updating n and p2 values
		# self.mean_EXACT_RB_difference = 0
		# self.mean_EXACT_RB_inner_product = 0
		# self.mean_RB_norm = 0

	def calculate_gradient(self, x,y,W):
		if random() > self.p2 :
			return NS_gradient(x,y,W,self.n)
		else :
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			grad_biased = RB_NS_gradient(x,y,W,self.n)

			# # Update p2 variables
			# K = W.shape[0]

			# self.mean_EXACT_RB_difference += norm(grad_EXACT - grad_biased)**2 * norm(x)**2
			# self.mean_EXACT_RB_inner_product += grad_biased.dot(grad_EXACT-grad_biased) * norm(x)**2
			# self.mean_RB_norm += (grad_EXACT[y]**2 + (float(K)-1)/self.n * (norm(grad_biased[:y])**2 + norm(grad_biased[y+1:])**2)) * norm(x)**2
			# p2_new = self.mean_EXACT_RB_difference / (2*self.mean_EXACT_RB_inner_product + self.mean_RB_norm)
			# #print self.n/float(K)*p2_new

			# Update n
			# self.n_numerator += (grad_biased.dot(grad_EXACT) - grad_biased[y]*grad_EXACT[y])*self.n
			# self.n_denominator += (grad_biased.dot(grad_biased) - grad_biased[y]*grad_biased[y])
			# self.n = self.n_numerator / self.n_denominator
			#print self.n

			grad = (grad_EXACT - grad_biased*(1-self.p2) ) / self.p2
			return [indices_EXACT , grad, self.K]

class DOVE(gradient):
	def __init__(self, n, K, p2_scale):
		self.n = n
		self.K = K
		self.p2 = p2_scale*sqrt(float(n)/K)

	def calculate_gradient(self, x,y,W):
		if random() > self.p2 :
			return OVE_gradient(x,y,W,self.n)
		else :
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			grad_biased = RB_OVE_gradient(x,y,W,self.n)
			grad = (grad_EXACT - grad_biased*(1-self.p2) ) / self.p2
			return [indices_EXACT , grad, self.K]



