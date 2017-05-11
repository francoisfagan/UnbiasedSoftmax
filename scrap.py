
	# # Rao-Blackwellized importance sampling gradient
	# K 				= W.shape[0]

	# # Calculate inner products and denominator
	# inner 			= W.dot(x)
	# inner_max_yD	= max([ inner[y] , np.max(inner[indices_D]) ])
	# inner			-= inner_max_yD

	# exp_inner 		= np.exp(inner)
	# Z 				= exp_inner[y] + np.sum(exp_inner[indices_D]) * (K-1)/n

	# grad 			= - exp_inner / (exp_inner + Z) #* n / (K-1)    
	# grad[y] 		= 1- exp_inner[y] / Z
	# return grad

def IS_RB_gradient(x,y,W,n,indices_D):
	# Importance sampling gradient that is Rao-Blackwellizable

	K = W.shape[0]

	# Sample indices
	indices_U = (int(y)+1+choice(K-1, n, replace = True) )%K
	#indices_U = list(set(indices_U) - set(indices_D))

	# Calculate inner products and denominator
	inner_y			= W[y,:].dot(x)
	inner_D 		= W[indices_D,:].dot(x)
	inner_U 		= W[indices_U,:].dot(x)
	inner_max_yD	= max([ inner_y , np.max(inner_D) ])

	# Log-sum-exp trick. Note that we only take the max over terms in the denominator, else it is possible that denominator=0.
	inner_y			-= inner_max_yD
	inner_D 		-= inner_max_yD
	inner_U 		-= inner_max_yD

	# Calculate scores and denominator
	exp_y 			= np.exp(inner_y)
	exp_D 			= np.exp(inner_D)
	exp_U 			= np.exp(inner_U)
	Z 				= exp_y + np.sum(exp_D) * (K-1)/n  # exp_y * n / (K-1) + np.sum(exp_D) Does terribly!

	# Add y index and gradient
	indices = np.concatenate( ([int(y)] , indices_U) ) #indices_D , 
	#grad 	= np.concatenate( ([sigma(log_Z - inner_y)] , - sigma(inner_D - log_Z) , - sigma(inner_U - log_Z)  / len(indices_U) * (K-1-len(indices_D)) )) 
	grad 	= np.concatenate( ( [1-exp_y / Z] , - exp_U / (exp_U  + Z) * (K-1)/n ) ) #* (K-1)/n

	return [ indices , grad , 2*n ]

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
	print "Gradual sample estimate: %s" %str(np.mean([multilevel(X,y,w,q,p,base) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact(X,y,w))
	"""
	# Note this calculation is approximate and is only accurate for large K
	base = int(W_p * (2*p-1) / (2*p))
	K = X.shape[0]
	# Maximum value of the geometric random variable
	J_max = np.ceil(np.log2(float(K)/base))
	# Sample from truncated geometric distribution with weter p
	J = truncated_geometric_sample(p,J_max)
	# R_base, R_J_minus and R_J
	R_base = importance(X,y,w,sample_range=xrange(base))#dot_base.dot(X[range_base]) / sum(dot_base)#sum(q[range_base]) * 
	R_J_minus = importance(X,y,w,sample_range=xrange(int(min(base*2**(J-1) , K)) ))#dot_J_minus.dot(X[range_J_minus]) / sum(dot_J_minus)#sum(q[range_J_minus]) * 
	R_J = importance(X,y,w,sample_range=xrange(int(min(base*2**J , K)) ))#dot_J.dot(X[range_J]) / sum(dot_J)#sum(q[range_J]) * 
	# Calculate R
	R = R_base + (R_J - R_J_minus ) / truncated_geometric_pmf(p,J_max,J)
	# Return softmax approximation
	return X[y] - R

def importance_gradient(X,y,w,q,n_samples=100):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w:		As for the exact_gradient function
		q:			Sampling probabilities of the indices. q[i] = Prob(y=i)
		n_samples:	Number of indices to sample

	Output:
		Importance sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**2
	print "Importance sample estimate: %s" %str(np.mean([importance_gradient(X,y,w,q,n_samples) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	K = X.shape[0]
	indices = choice(range(K), n_samples, p = q)
	vec = np.exp(X[indices].dot(w)) / q[indices]
	return X[y] - vec.dot(X[indices])/np.sum(vec)

def unbiased_importance_gradient(X,y,w,q,p_2,n_samples=100):
	"""
	Calculate the softmax gradient using the importance sampling method
	Input:
		X,y,w,q, n_samples:		As for the importance_gradient function
		p_2:					Probability of sampling exact gradient

	Output:
		Unbiased importance sampling estimate for the gradient of the softmax

	To test whether this function works as intended, check its gradients vs the true gradient:
	rep = 10**5
	print "Unbiased importance sample estimate: %s" %str(np.mean([unbiased_importance_gradient(X,y,w,q,n_samples,p_2) for _ in xrange(rep)],axis=0))
	print "Exact: %s" % str(exact_gradient(X,y,w))
	"""
	R = importance_gradient(X,y,w,q,n_samples)
	if random() > p_2 :  return R
	else : return (exact_gradient(X,y,w) - (1-p_2)*R)/p_2


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

