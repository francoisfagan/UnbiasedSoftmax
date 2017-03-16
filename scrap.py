


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

