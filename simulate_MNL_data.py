# Simulate_data for Multinomial Logistic Regression, aka softmax 

import numpy as np
from numpy.random import randn, choice

np.random.seed(1)
# General Hyperparameters
K = 10**3													# Number of classes
dim = 2													# Dimension
n_datapoints = 10**6										# Number of datapoints
train_test_split = 0.7									# Fraction that is training data
print_or_save = "save"#"print"									# Whether to print or save the data

# Simulate data
sigma = 10
W = randn(K,dim)*sigma										# True W value
X = randn(n_datapoints,dim)								# Covariates, one for each datapoint
Y = np.zeros((n_datapoints,1))
for i in xrange(n_datapoints):	
	q = np.exp(W.dot(X[i]))								# Unnormalized sampling probabilities for each datapoint
	q = q/sum(q)										# Normalized sampling probabilities for each datapoint
	Y[i]= int(choice(range(K), 1, p=q)[0])					# Sample a class for each datapoint
#print (Y)
X_Y = np.hstack((X,Y))

# Save or print simulated data
str_header = '../UnbiasedSoftmaxData/Simulated/simulated_data_K_%d_dim_%d_n_datapoints_%d_sigma_%d'%(K, dim, n_datapoints, sigma)+'_'	# Gives details on parameters for how data simulated
for data in ["train","test"]:
	# Set indices for training or testing data
	data_range = range(int(train_test_split*n_datapoints)) if data == "train" else range(int(train_test_split*n_datapoints), n_datapoints)
	# String of data
	np.savetxt(str_header+data+'.csv', X_Y[data_range,:], delimiter=",")
	# data_str = "\n".join([ str(", ".join(str(x) for x in X[i]))+", "+str(Y[i]) for i in data_range])

\

