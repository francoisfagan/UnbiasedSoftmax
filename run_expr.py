# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
from sklearn import linear_model
from unbiased_gradients import *
import pickle
import os





def readHyperParameters():
	data_path = "../UnbiasedSoftmaxData/Simulated/simulated_data_K_100_dim_2_n_datapoints_100000"

	hyper_param = 0.1
	repetitions = 10
	time_total = 10**5
	n_eval_loss = 10
	NS_n = 5
	OVE_n = 5
	p2_scale = 1

	return [data_path , hyper_param , repetitions , time_total, n_eval_loss, NS_n, OVE_n, p2_scale]
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


	# Plot figures


class Solver:

	def __init__(self, data_path, hyper_param, repetitions, time_total, n_eval_loss, NS_n, OVE_n, p2_scale):

		# Set hyperparameters
		self.hyper_param = hyper_param
		self.repetitions = repetitions
		self.time_total = time_total
		self.n_eval_loss = n_eval_loss

		# Load training and test data
		self.data_path = data_path
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
		self.NS_n = NS_n
		self.OVE_n = OVE_n
		self.p2_scale = p2_scale


		self.parameter_save_name = "_hyper_%d_rep_%d_time_%d_n_eval_loss_%d_NS_n_%d_OVE_n_%d_p2_scale_%d_"%(hyper_param,repetitions,time_total,n_eval_loss,NS_n,OVE_n,p2_scale)+data_path[data_path.rfind("/")+1:]

		# store results of running methods
		self.method_test_scores = {}
		self.method_train_scores = {}

	def test_score(self,W):
		return np.mean(np.argmax(W.dot(self.X_test.T),axis=0)==self.Y_test)

	def train_score(self,W):
		return np.mean(np.argmax(W.dot(self.X_train.T),axis=0)==self.Y_train)
		
	def scikit_learn(self):
		logreg = linear_model.LogisticRegression(C=1e5,solver ='newton-cg',multi_class='multinomial',fit_intercept=False)
		logreg.fit(self.X_train, self.Y_train)
		print np.mean(logreg.predict(self.X_test)==self.Y_test)
		
	def fit(self, method):
		print(method)
		if method == 'scikit_learn':
			self.scikit_learn()
			return

		if   method 	== 'EXACT':		gradient_calculator = EXACT()
		elif method 	== 'NS':		gradient_calculator = NS(self.NS_n)
		elif method 	== 'OVE':		gradient_calculator = OVE(self.OVE_n)
		elif method 	== 'DNS':		gradient_calculator = DNS(self.NS_n, self.K, self.p2_scale)
		elif method 	== 'DOVE':		gradient_calculator = DOVE(self.OVE_n, self.K, self.p2_scale)
		else:						raise ValueError('Not a valid method method.')
		

		self.method_test_scores[method] = []
		self.method_train_scores[method] = []
		for rep in xrange(repetitions):
			#print(rep)
			W = np.zeros((self.K,self.dim))
			time = 0
			prev_time = 0
			test_scores_cum_work = []
			train_scores_cum_work = []
			iteration = 0
			while time < self.time_total:
				data_index = randint(0,self.n_samples_train-1)
				grad_indices , grad , work = gradient_calculator.calculate_gradient(self.X_train[data_index],self.Y_train[data_index],W)
				W[grad_indices] += self.hyper_param*np.outer(grad,self.X_train[data_index])*(0.9*(self.time_total-time)/float(self.time_total) +0.1)
				time += work
				iteration += 1
				if (time > (prev_time + float(self.time_total) / self.n_eval_loss)):
					prev_time = time
					#print self.tester.test(W)
					#self.param_history.append(numpy.array(params))
					test_scores_cum_work.append([self.test_score(W) , time])
					train_scores_cum_work.append([self.train_score(W) , time])
			self.method_test_scores[method].append(test_scores_cum_work)
			self.method_train_scores[method].append(train_scores_cum_work)

	def save_results(self):
		# Pickle results
		pickle_name = os.getcwd() + "/Data/Pickled_scores/"+"_".join(self.method_scores.keys())+self.parameter_save_name+".p"
		with open(pickle_name, 'wb') as f:
			pickle.dump(self.method_scores, f)

	def plot_results(self,test_or_train):


		if test_or_train == 'Test':
			method_scores = self.method_test_scores
		elif test_or_train == 'Train':
			method_scores = self.method_train_scores
		else:
			print("Select 'Test' or 'Train' to plot results.")

		if 'EXACT' not in method_scores.keys():
			print("Exact method was not run so cannot plot")
			return

		times = [score_time[1] for score_time in method_scores['EXACT'][0]]
		inter_time = (times[1] - times[0])*0.9 # Window around time periods. Multiplied by 0.9 so times[i+1] and times[i] do not overlap
		fig, ax = plt.subplots()
		for method in ['EXACT','OVE','DOVE','NS','DNS']:
			if method in method_scores.keys():
				rep_scores_cum_work = method_scores[method]
				flattened_score_times = [score_time for rep in rep_scores_cum_work for score_time in rep]
				flattened_score_times.sort(key = lambda x: x[1]) # So that they are ordered in increasing times
				scores_times = [[score_time[0] for score_time in flattened_score_times if abs(score_time[1]-time)<=inter_time] for time in times]
				scores_time_mean = [np.mean(scores_time) for  scores_time in scores_times]
				scores_time_std = [np.std(scores_time) for  scores_time in scores_times]
				ax.errorbar(times,scores_time_mean, yerr = scores_time_std, label=method)

		legend = ax.legend(loc='lower right', shadow=True)

		plt.xlabel('Time')
		plt.ylabel('Score')
		plt.title(test_or_train+' accuracy')
		plot_save_name = os.getcwd() + "/Data/Plots/"+"_".join(method_scores.keys())+"_"+test_or_train+self.parameter_save_name+".png"
		plt.savefig(plot_save_name)
		plt.show()

		for method , rep_scores_cum_work in method_scores.iteritems():
			final_scores = [ scores_cum_work[-1][0] for scores_cum_work in rep_scores_cum_work]
			print "Mean final score %.2f with std %.2f" %(np.mean(final_scores) , np.std(final_scores))



if __name__ == "__main__":
	np.random.seed(1)

	# Read hyperparameters from the terminal
	data_path , hyper_param , repetitions , time_total, n_eval_loss, NS_n, OVE_n, p2_scale = readHyperParameters()

	# Create trainer class to run the Trains in
	solver = Solver(data_path , hyper_param, repetitions , time_total, n_eval_loss, NS_n, OVE_n, p2_scale)
	for method in ['EXACT','DNS','DOVE','OVE','NS']:#'scikit_learn',
		solver.fit(method)
	#solver.save_results()
	solver.plot_results('Test')
	solver.plot_results('Train')




