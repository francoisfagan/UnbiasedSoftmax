# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn import linear_model
from Dgradients import *

def readHyperParameters():
	data_path = "simulated_data_K_3_dim_2_n_datapoints_100000"

	hyper_param = 0
	repetitions = 1
	work_total = 10*3
	eval_delay = 10
	work_frac = 0.9

	return [data_path , hyper_param , repetitions , work_total, work_frac, eval_delay]
	# # Load hyperparameters from the terminal
 #    if len(sys.argv) > 1:
 #        data_set = sys.argv[1]
 #        max_iter = int(sys.argv[2])
 #        hyper_param = float(sys.argv[3])
 #        repetitions = int(sys.argv[4])


 #        return [train_path, test_path , max_iter , hyper_param , repetitions]
 #    else:
 #        print"""
 #        ------------------------------------
 #        Recommended defaults:
 #        max_iter = 5000
 #        hyper_param = 0.01
 #        repetitions = 1
 #        ------------------------------------
 #        """
 #        exit(0)


class Solver:

	def __init__(self, data_path, hyper_param, repetitions=1, work_total=100, work_frac=0.1, n_eval_loss=10):

		# Set hyperparameters
		self.hyper_param = hyper_param
		self.repetitions = repetitions
		self.work_total = work_total
		self.work_frac = work_frac
		self.n_eval_loss = n_eval_loss

		# Load training and test data
		train_data = np.genfromtxt(data_path + "_train.csv", delimiter=',')
		test_data = np.genfromtxt(data_path + "_test.csv", delimiter=',')

		# Store training data
		self.X_train = train_data[:,:-1]
		self.Y_train = train_data[:,-1]
		self.n_samples_train = int(train_data.shape[0])

		# Store test data
		self.X_test = test_data[:,:-1]
		self.Y_test = test_data[:,-1]
		self.n_samples_test = int(test_data.shape[0])

		# Set parameter values using training data
		self.dim = int(train_data.shape[1]-1)
		self.K = int(max(self.Y_train)+1)
		# self.q = np.array([1.0/self.K]*self.K)

		# Set algorithm constants
		self.NS_n = 2
		self.OVE_n = 2

	def score(self,W):
		return np.mean(np.argmax(W.dot(self.X_test.T),axis=0)==self.Y_test)
		

	def scikit_learn(self):
		logreg = linear_model.LogisticRegression(C=1e5,solver ='newton-cg',multi_class='multinomial',fit_intercept=False)
		logreg.fit(self.X_train, self.Y_train)
		print np.mean(logreg.predict(self.X_test)==self.Y_test)
		

	def fit(self, name):
		print(name)
		if name == 'scikit_learn':
			self.scikit_learn()
		else:
			max_iter = self.work_total
			#W_history = np.zeros(self.repetitions , (self.K,self.dim) , max_iter/self.n_eval_loss )

			for repeat in range(repetitions):
				print(repeat)
				W = np.zeros((self.K,self.dim))
				for i in xrange(0,max_iter):
					data_index = randint(0,self.n_samples_train-1)
					#pdb.set_trace()
					#eval(name + "(self.X_train[data_index],self.Y_train[data_index],W,work_frac)")
					if name == 'exact':
						grad_indices , grad = exact(self.X_train[data_index],self.Y_train[data_index],W)
					elif name == 'NS':
						grad_indices , grad = NS(self.X_train[data_index],self.Y_train[data_index],W,self.NS_n)
					elif name == 'OVE':
						grad_indices , grad = OVE(self.X_train[data_index],self.Y_train[data_index],W,self.OVE_n)
					elif name == 'DNS':
						grad_indices , grad = DNS(self.X_train[data_index],self.Y_train[data_index],W,self.work_frac,self.NS_n)
					elif name == 'DOVE':
						grad_indices , grad = DOVE(self.X_train[data_index],self.Y_train[data_index],W,self.work_frac,self.OVE_n)
					else:
						print('Error, not a valid method. Check the method name.')

					W[grad_indices] += 0.1*np.outer(grad,self.X_train[data_index])
					print self.score(W)
					#if (i%round(self.n_eval_loss+1) == 0):
						#print self.tester.test(W)
						#self.param_history.append(numpy.array(params))



if __name__ == "__main__":

	# Read hyperparameters from the terminal
	data_path , hyper_param , repetitions , work_total, work_frac, eval_delay = readHyperParameters()

	# Create trainer class to run the Trains in
	solver = Solver(data_path , hyper_param, repetitions , work_total, work_frac, eval_delay)
	for method in ['scikit_learn','exact','NS','OVE','DNS','DOVE']:#
		solver.fit(method)


