"""
Biased methods for the softmax probability distribution

"""
from numpy.random import random, choice, randint, permutation
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
	K 			= W.shape[0]
	inner_prod 	= W.dot(x)
	inner_prod  -= np.max(inner_prod)
	grad 		= np.exp(inner_prod) 
	grad 		= -grad/sum(grad)
	grad[y] 	= grad[y] + 1
	return [ range(K) , grad , K ]

def NS_gradient(x,y,W,n):
	# Negative sampling gradient

	K 		= W.shape[0]

	# Sample indices and calculate gradient	
	indices = (int(y)+1+choice(K-1, n, replace = False) )%K
	grad 	= - sigma(W[indices,:].dot(x))

	# Add y index and gradient
	indices = np.append( int(y), indices)
	grad 	= np.append( sigma(-W[y,:].dot(x)) , grad )
	return [ indices , grad , n ]

def OVE_gradient(x,y,W,n):
	# One-vs-each gradient

	K = W.shape[0]

	# Sample indices and calculate gradient
	indices = (int(y)+1+choice(K-1, n, replace = False) )%K
	grad 	= - sigma( W[indices,:].dot(x) - W[y,:].dot(x) ) / n

	# Add y index and gradient
	indices =  np.append( int(y), indices )
	grad 	= np.append( -sum(grad) , grad )
	return [ indices , grad , n ]

def IS_gradient(x,y,W,n,indices):
	# Importance sampling gradient

	K = W.shape[0]

	# Calculate inner products
	inner_y			= W[y,:].dot(x)
	inner			= W[indices,:].dot(x)
	inner_max		= max(np.max(inner),inner_y)

	# Calculate scores and denominator
	exp_y 			= np.exp(inner_y - inner_max)
	exp_indices 	= np.exp(inner - inner_max)
	sum_exp_indices = np.sum(exp_indices)

	# Add y index and gradient
	indices = np.append( int(y), indices )

	# # Importance sampling style
	# grad 	= np.append( 1-exp_y / (exp_y + sum_exp_indices/float(n) *(K-1) ) , 
	#                      -exp_indices / (exp_y + sum_exp_indices/float(n) *(K-1) )  / float(n) * (K-1)   )
	
	# # Blackout style
	# grad 	= np.append( 1-exp_y / (exp_y + sum_exp_indices ) , 
	#                      -exp_indices / (exp_y + sum_exp_indices)  )
	
	# My style
	gamma 	= 0.0 # 0=<gamma<=1. If gamma = 0 then same as IS, if gamma = 1 then my new method
	grad 	= np.append( 1-exp_y / (exp_y + sum_exp_indices/float(n) *(K-1) ) , 
	                     -exp_indices / (exp_y + gamma*exp_indices + (sum_exp_indices - gamma*exp_indices)/float(n-gamma) *(K-1-gamma) ) / float(n) * (K-1)   )


	# if np.linalg.norm(grad-grad_IS) / np.linalg.norm(grad_IS) > 0.5:
	# 	print(grad)
	# 	print(grad_IS)
	# 	pdb.set_trace()

	return [ indices , grad , n ]


# Methods to calculate Rao-Blackwellized gradients

def RB_NS_gradient(x,y,W,n):
	# Rao-Blackwellized negative sampling gradient
	K 		= W.shape[0]
	grad 	= - n / float(K-1) * sigma(W.dot(x))
	grad[y] = sigma(-W[y,:].dot(x))
	return grad

def RB_OVE_gradient(x,y,W,n):
	# Rao-Blackwellized one-vs-each gradient
	K 		= W.shape[0]
	grad 	= - sigma( W.dot(x) - W[y,:].dot(x) )  / (K-1)
	grad[y] = 0
	grad[y] = - sum(grad)
	return grad

def RB_IS_gradient(x,y,W,n,perm):
	# Rao-Blackwellized importance sampling gradient
	K 				= W.shape[0]
	grad 			= np.zeros(K)

	num_partitions 	= int(np.ceil(float(K-1)/n))
	for i_perm in xrange(num_partitions):
		indices = perm[i_perm*n : min((i_perm+1)*n,K-1)]
		indices , grad_indices , _ = IS_gradient(x,y,W,n, indices) # Note that now indices will contain the y index whereas previously it didn't
		grad[indices] 	+= grad_indices / num_partitions
#	print(n)
#	pdb.set_trace()
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

class IS(gradient):
	# Importance sampling
	def __init__(self, n):
		self.n = n

	def calculate_gradient(self, x,y,W):
		# Sample indices
		K 		= W.shape[0]
		self.n = int(sqrt(K))
		indices = (int(y)+1+choice(K-1, self.n, replace = False) )%K
		return IS_gradient(x,y,W,self.n, indices)

class IS_RB(gradient):
	# Importance sampling
	def __init__(self, n):
		self.n = n

	def calculate_gradient(self, x,y,W):
		K 			= W.shape[0]
		indices_D 	= (int(y)+1+choice(K-1, self.n, replace = False) )%K
		return IS_RB_gradient(x,y,W,self.n, indices_D)

# Debiased gradient classes

class DNS(gradient):
	def __init__(self, n, K, p2_scale, alpha, time_total):
		self.n = n
		self.K = K
		self.p2 = p2_scale*sqrt(float(n)/K)
		self.p2_counter = 0
		self.alpha_initial = alpha
		self.time_total = float(time_total)
		self.time = 0.0

		# Variables for updating n and p2 values
		self.mean_EXACT_RB_difference = 0
		self.mean_EXACT_RB_inner_product = 0
		self.mean_RB_norm = 0

	def calculate_gradient(self, x,y,W):
		
		alpha = 1.0#min(self.alpha_initial * self.time / self.time_total,1.0)#self.alpha_initial #
		self.p2_counter += alpha*self.p2
		if self.p2_counter <= 1-self.p2 :
			self.time += self.n
			return NS_gradient(x,y,W,self.n)
		else :
			self.time += self.K
			self.p2_counter = 0
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			grad_biased = RB_NS_gradient(x,y,W,self.n)

			# # Update p2 variables
			# K = W.shape[0]

			# # I HAVE REMOVED + FROM += BELOW
			# self.mean_EXACT_RB_difference += norm(grad_EXACT - grad_biased)**2 * norm(x)**2
			# self.mean_EXACT_RB_inner_product += grad_biased.dot(grad_EXACT-grad_biased) * norm(x)**2
			# self.mean_RB_norm += (grad_EXACT[y]**2 + (float(K)-1)/self.n * (norm(grad_biased[:y])**2 + norm(grad_biased[y+1:])**2)) * norm(x)**2
			# p2_new = self.mean_EXACT_RB_difference / (2*self.mean_EXACT_RB_inner_product + self.mean_RB_norm)
			# if random()<0.001:
			# 	print p2_new#*sqrt(float(self.n)/K)

			
			grad = (grad_EXACT - grad_biased ) / self.p2 + grad_biased #alpha*
			return [indices_EXACT , grad, self.K]

			# Debiased gradient classes

class DOVE(gradient):
	def __init__(self, n, K, p2_scale, alpha, time_total):
		self.n = n
		self.K = K
		self.p2 = sqrt(float(1.0)/K)#p2_scale*sqrt(float(n)/K)#p2_scale*(float(n)/K)**(1.0/4)#0.5#
		self.p2_counter = 0
		self.alpha_initial = alpha
		self.time_total = float(time_total)
		self.time = 0.0


	def calculate_gradient(self, x,y,W):
		alpha = min(self.alpha_initial * self.time / self.time_total,1.0)#self.alpha_initial #
		self.p2_counter += alpha*self.p2
		if self.p2_counter <= 1-self.p2 :
			self.time += self.n
			return OVE_gradient(x,y,W,self.n)
		else :
			self.time += self.K
			self.p2_counter = 0
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			grad_biased = RB_OVE_gradient(x,y,W,self.n)
			grad = (grad_EXACT - grad_biased ) / self.p2 + grad_biased #alpha*
			return [indices_EXACT , grad, self.K]

class DIS(gradient):
	def __init__(self, n, K, p2_scale, alpha, time_total):
		self.n = int(sqrt(K))
		self.K = K
		self.p2 = p2_scale*sqrt(1.0/K)
		self.p2_counter = 0
		self.p2_mean = 0.5
		self.iteration_counter = 1
		self.alpha_initial = alpha
		self.time_total = float(time_total)
		self.time = 0.0

	def calculate_gradient(self, x,y,W):
		K 				= W.shape[0]
		perm 			= permutation(range(int(y)) + range(int(y)+1,K))
		alpha 			= min(self.alpha_initial * self.time / self.time_total,1.0)#self.alpha_initial #
		self.p2_counter += alpha*self.p2
		if self.p2_counter <= 1-self.p2 :
			self.time += self.n
			return IS_gradient(x,y,W,self.n, perm[:self.n]) #RB_
		else :
			self.time += self.K
			self.p2_counter = 0
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			grad_biased = RB_IS_gradient(x,y,W,self.n,perm)
			grad = (grad_EXACT - grad_biased*(1-self.p2) ) / self.p2
			self.p2_mean = self.p2_mean * (1-1.0 / self.iteration_counter) + sqrt(np.linalg.norm(grad_biased-grad_EXACT) / np.linalg.norm(grad_EXACT)) / self.iteration_counter
			self.iteration_counter += 1
			# self.p2 = min(1,self.p2_mean * sqrt(self.n/float(K)))
			# if random()<0.01:
			# 	print(self.p2)
			# 	#print(np.linalg.norm(grad_biased-grad_EXACT) / np.linalg.norm(grad_EXACT))
			# 	#pdb.set_trace()
			return [indices_EXACT , grad, self.K]



# Debiased gradient classes non-RB

class DNS_nonRB(gradient):
	def __init__(self, n, K, p2_scale):
		self.n = n
		self.K = K
		self.p2 = p2_scale*sqrt(float(n)/K)

	def calculate_gradient(self, x,y,W):
		if random() > self.p2 :
			return NS_gradient(x,y,W,self.n)
		else :
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			indices_biased , grad_biased , _  = NS_gradient(x,y,W,self.n)
			grad = grad_EXACT / self.p2
			grad[indices_biased] =  - grad_biased*(1-self.p2)  / self.p2
			return [indices_EXACT , grad, self.K]

class DOVE_nonRB(gradient):
	def __init__(self, n, K, p2_scale):
		self.n = n
		self.K = K
		self.p2 = p2_scale*sqrt(float(n)/K)

	def calculate_gradient(self, x,y,W):
		if random() > self.p2 :
			return OVE_gradient(x,y,W,self.n)
		else :
			indices_EXACT , grad_EXACT , _ = EXACT_gradient(x,y,W)
			indices_biased , grad_biased , _ = OVE_gradient(x,y,W,self.n)
			grad = grad_EXACT / self.p2
			grad[indices_biased] =  - grad_biased*(1-self.p2)  / self.p2
			return [indices_EXACT , grad, self.K]



