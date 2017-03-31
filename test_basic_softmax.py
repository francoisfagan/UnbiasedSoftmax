#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import scipy.sparse
from numpy.random import randn, choice
import pdb
from numpy.linalg import norm


data_path = "../UnbiasedSoftmaxData/Simulated/simulated_data_K_100_dim_2_n_datapoints_100000"

train_data = np.genfromtxt(data_path + "_train.csv", delimiter=',')
test_data = np.genfromtxt(data_path + "_test.csv", delimiter=',')
np.random.seed(1)


K = 3#10**1													# Number of classes
dim = 2													# Dimension
n_datapoints = 10**4										# Number of datapoints

# Simulate data
print ("Simulating data")
sigma = 10
W_true = randn(K,dim)*sigma										# True W value
X = randn(n_datapoints,dim)								# Covariates, one for each datapoint
Y = np.array(xrange(n_datapoints))
for i in xrange(n_datapoints):
	inner_prod = W_true.dot(X[i])
	inner_prod  -= np.max(inner_prod)	
	q = np.exp(inner_prod)								# Unnormalized sampling probabilities for each datapoint
	q = q/sum(q)										# Normalized sampling probabilities for each datapoint
	Y[i]= int(choice(range(K), 1, p=q)[0])					# Sample a class for each datapoint

x = X
y = Y


def EXACT_gradient(x,y,W):
	K 			= W.shape[0]
	inner_prod 	= W.dot(x)
	inner_prod  -= np.max(inner_prod)
	grad 		= np.exp(inner_prod) 
	grad 		= -grad/sum(grad)
	grad[y] 	+= 1
	return grad#[ range(K) , grad , K ]

def oneHotIt(Y):
    m = Y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def getGrad(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    #loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return grad

def getLoss(w,x,y):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) #We then find the loss of the probabilities
    return loss

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm



lam = 0.0
iterations = 10**6
learningRate = 0.1

print ("Fitting W_fit")
W_fit = np.array(W_true)#np.zeros([len(np.unique(y)),x.shape[1]])
W_fit2 = np.array(W_true*0.1)
for it in range(iterations):
	i_data = it%n_datapoints
	x_i = X[i_data,:]
	y_i = Y[i_data]
	grad = EXACT_gradient(x_i,y_i,W_fit)
	W_fit += learningRate*(0.99*(iterations-it) +0.01)/float(iterations) *(np.outer(grad,x_i))# - lam*W_fit)
	grad2 = EXACT_gradient(x_i,y_i,W_fit2)
	W_fit2 = W_fit2 + learningRate*(0.99*(iterations-it) +0.01)/float(iterations) *(np.outer(grad2,x_i))# - lam*W_fit)
	# print (W_fit - W_true)
	# print (W_fit2 - W_true)


print "W_true used to generate the data loss: %f with gradient of norm: %f" % (getLoss(W_true.T,x,y) , norm(getGrad(W_true.T,x,y,0)))
print W_true
print "W fit by EXACT: %f with gradient of norm: %f" % (getLoss(W_fit.T,x,y) , norm(getGrad(W_fit.T,x,y,0)))
print W_fit
print "W fit2 by EXACT: %f with gradient of norm: %f" % (getLoss(W_fit2.T,x,y) , norm(getGrad(W_fit2.T,x,y,0)))
print W_fit2







# print "W fit by other method: %f" %getLoss(w,x,y)
# print w.T
# print ("Fitting w")
# w = np.zeros([x.shape[1],len(np.unique(y))])
# for i in range(0,iterations):
# 	grad = getGrad(w,x,y,lam)
# 	w -=  (learningRate * grad)

