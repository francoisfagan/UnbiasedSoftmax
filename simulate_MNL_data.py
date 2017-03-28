# Simulate_data for Multinomial Logistic Regression, aka softmax 

import numpy as np
from numpy.random import randn, choice

# General Hyperparameters
K = 3													# Number of classes
dim = 2													# Dimension
n_datapoints = 10**5										# Number of datapoints
train_test_split = 0.7									# Fraction that is training data
print_or_save = "save"#"print"									# Whether to print or save the data

# Simulate data
W = randn(K,dim)*10										# True W value
X = randn(n_datapoints,dim)								# Covariates, one for each datapoint
q = [np.exp(W.dot(X[i])) for i in xrange(n_datapoints)]	# Unnormalized sampling probabilities for each datapoint
q = [q[i]/sum(q[i]) for i in xrange(n_datapoints)]		# Normalized sampling probabilities for each datapoint
Y = [choice(range(K), 1, p=q)[0] for q in q]			# Sample a class for each datapoint

# Save or print simulated data
str_header = 'simulated_data_K_%d_dim_%d_n_datapoints_%d'%(K, dim, n_datapoints)+'_'	# Gives details on parameters for how data simulated
for data in ["train","test"]:
	# Set training or testing data rante
	data_range = xrange(int(train_test_split*n_datapoints)) if data == "train" else xrange(int(train_test_split*n_datapoints), n_datapoints)
	# String of data
	data_str = "\n".join([ str(", ".join(str(x) for x in X[i]))+", "+str(Y[i]) for i in data_range])

	# Either print or save the results
	if print_or_save == "print":
		print(data_str)
		print("")
	elif print_or_save == "save":
		with open(str_header+data+'.csv', 'w') as the_file:
				the_file.write(data_str)
	else:
		print("Either print or save the data!")

# Either print or save the results
if print_or_save == "print":
	print(W)
	print("")
elif print_or_save == "save":
	np.savetxt(str_header+"W"+'.csv',W, delimiter=' , ')
else:
	print("Either print or save the data!")

