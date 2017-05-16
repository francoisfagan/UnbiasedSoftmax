# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import sys
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy.random import random
from sklearn import linear_model
from unbiased_gradients import *
import pickle
import os
from sklearn.datasets import load_svmlight_file


def loadLIBSVMdata(data_path):
	train_test_split = 0.7
	data = load_svmlight_file(data_path, multilabel=True)
	Y = data[1]
	i_Y_not_empty = [i for i,y in enumerate(Y) if y!=()]
	X = data[0].toarray()[i_Y_not_empty,:]
	Y = [Y[i][0] for i in i_Y_not_empty]

	n_samples = len(Y)
	X_train = X[:int(train_test_split*n_samples),:]
	Y_train = Y[:int(train_test_split*n_samples)]
	X_test = X[int(train_test_split*n_samples):,:]
	Y_test = Y[int(train_test_split*n_samples):]
	return X_train , Y_train , X_test , Y_test

class Solver:

	def __init__(self, data_path, hyper_param, repetitions, time_total, n_eval_loss, NS_n, OVE_n, IS_n, p2_scale, alpha):

		# Set hyperparameters
		self.hyper_param = hyper_param
		self.repetitions = repetitions
		self.time_total = time_total
		self.n_eval_loss = n_eval_loss

		# Load training and test data
		self.data_path = data_path
		if data_path[:29] == "../UnbiasedSoftmaxData/LIBSVM":
			self.X_train , self.Y_train , self.X_test , self.Y_test = loadLIBSVMdata(data_path)
		else:
			train_data = np.genfromtxt(data_path + "_train.csv", delimiter=',')
			test_data = np.genfromtxt(data_path + "_test.csv", delimiter=',')

			# Store training data
			self.X_train = train_data[:,:-1]
			self.Y_train = train_data[:,-1]

			# Store test data
			self.X_test = test_data[:,:-1]
			self.Y_test = test_data[:,-1]

		self.n_samples_train = len(self.Y_train)
		self.n_samples_test = len(self.Y_test)

		# Set parameter values using training data
		self.dim = int(len(self.X_train[0]))
		self.K = int(max(self.Y_train)+1)

		# Set algorithm constants
		self.NS_n = NS_n
		self.OVE_n = OVE_n
		self.IS_n = IS_n
		self.p2_scale = p2_scale
		self.alpha = alpha
		self.start_indices = randint(0,self.n_samples_train,repetitions)

		self.parameter_save_name = "_hyper_%.3f_rep_%d_time_%d_n_eval_loss_%d_NS_n_%d_OVE_n_%d_p2_scale_%d_alpha_%.2f_"%(hyper_param,repetitions,time_total,n_eval_loss,NS_n,OVE_n,p2_scale,alpha)+data_path[data_path.rfind("/")+1:]

		# store results of running methods
		self.method_test_scores = {}
		self.method_train_scores = {}

	def test_score(self,W):
		return np.mean(np.argmax(W.dot(self.X_test.T),axis=0)==self.Y_test)#[::10,:][::10]

	def train_score(self,W):
		return np.mean(np.argmax(W.dot(self.X_train[::50,:].T),axis=0)==self.Y_train[::50])
		
	def scikit_learn(self):
		logreg = linear_model.LogisticRegression(C=1e5,solver ='newton-cg',multi_class='multinomial',fit_intercept=False)
		logreg.fit(self.X_train, self.Y_train)
		print np.mean(logreg.predict(self.X_test)==self.Y_test)
		
	def fit(self, method):
		print(method)
		if method == 'scikit_learn':
			self.scikit_learn()
			return

		

		self.method_test_scores[method] = []
		self.method_train_scores[method] = []
		for rep in xrange(repetitions):


			if   method 	== 'EXACT':		gradient_calculator = EXACT()
			elif method 	== 'NS':		gradient_calculator = NS(self.NS_n)
			elif method 	== 'OVE':		gradient_calculator = OVE(self.OVE_n)
			elif method 	== 'IS':		gradient_calculator = IS(self.IS_n)
			elif method 	== 'IS_RB':		gradient_calculator = IS_RB(self.IS_n)
			elif method 	== 'DNS':		gradient_calculator = DNS(self.NS_n, self.K, self.p2_scale, self.alpha, self.time_total)
			elif method 	== 'DNS_nonRB':	gradient_calculator = DNS_nonRB(self.NS_n, self.K, self.p2_scale)
			elif method 	== 'DOVE':		gradient_calculator = DOVE(self.OVE_n, self.K, self.p2_scale, self.alpha, self.time_total)
			elif method 	== 'DOVE_nonRB':gradient_calculator = DOVE_nonRB(self.OVE_n, self.K, self.p2_scale)
			elif method 	== 'DIS':		gradient_calculator = DIS(self.IS_n, self.K, self.p2_scale, self.alpha, self.time_total)
			else:						raise ValueError('Not a valid method method.')

			print(rep)
			W = np.zeros((self.K,self.dim))
			time = 0
			prev_time = 0
			test_scores_cum_work = []
			train_scores_cum_work = []
			iteration = self.start_indices[rep]
			while time < self.time_total:
				data_index = iteration % self.n_samples_train #data_index = randint(0,self.n_samples_train) this would have more variance in the results, therefore don't use
				grad_indices , grad , work = gradient_calculator.calculate_gradient(self.X_train[data_index],int(self.Y_train[data_index]),W)
				W[grad_indices] = W[grad_indices] + self.hyper_param*np.outer(grad,self.X_train[data_index])*(0.9*(self.time_total-time)/float(self.time_total) +0.1)
				time += work
				iteration += 1
				if (time > (prev_time + float(self.time_total) / self.n_eval_loss)):
					prev_time = time
					test_scores_cum_work.append([self.test_score(W) , time])
					#train_scores_cum_work.append([self.train_score(W) , time])
			self.method_test_scores[method].append(test_scores_cum_work)
			self.method_train_scores[method].append(train_scores_cum_work)

	def save_results(self):
		# Pickle results
		pickle_name = os.getcwd() + "/Data/Pickled_scores/"+"_".join(self.method_test_scores.keys())+self.parameter_save_name+".p"
		with open(pickle_name, 'wb') as f:
			pickle.dump(self.method_test_scores, f)

	def plot_results(self,test_or_train):

		if test_or_train == 'Test':
			method_scores = self.method_test_scores
		elif test_or_train == 'Train':
			method_scores = self.method_train_scores
		else:
			print("Select 'Test' or 'Train' to plot results.")
			return

		if 'EXACT' not in method_scores.keys():
			print("Exact method was not run so cannot plot")
			return

		times = [score_time[1] for score_time in method_scores['EXACT'][0]]
		inter_time = (times[1] - times[0])*0.9 # Window around time periods. Multiplied by 0.9 so times[i+1] and times[i] do not overlap
		fig, ax = plt.subplots()
		for method in ['EXACT','OVE','DOVE','DOVE_nonRB','NS','DNS','DNS_nonRB','IS','IS_RB','DIS']:
			if method in method_scores.keys():
				rep_scores_cum_work = method_scores[method]
				flattened_score_times = [score_time for rep in rep_scores_cum_work for score_time in rep]
				flattened_score_times.sort(key = lambda x: x[1]) # So that they are ordered in increasing times
				scores_times = [[score_time[0] for score_time in flattened_score_times if abs(score_time[1]-time)<=inter_time] for time in times]
				scores_time_mean = [np.mean(scores_time) for  scores_time in scores_times]
				scores_time_std = [np.std(scores_time) for  scores_time in scores_times]
				ax.errorbar(times,scores_time_mean, yerr = scores_time_std, label=method)

		legend = ax.legend(loc='lower right', shadow=True)

		plt.xlabel('No. inner products')
		plt.ylabel('Score')
		plt.title(test_or_train+' accuracy')
		plot_save_name = os.getcwd() + "/Data/Plots/"+"_".join(method_scores.keys())+"_"+test_or_train+self.parameter_save_name#+".png"
		#plt.savefig(plot_save_name)
		fig.set_canvas(plt.gcf().canvas)
		fig.savefig(plot_save_name + ".pdf", format='pdf')
		plt.show()


if __name__ == "__main__":
	np.random.seed(1)

	data_path = "../UnbiasedSoftmaxData/LIBSVM/Delicious_data.txt"
	"""
	"../UnbiasedSoftmaxData/Simulated/simulated_data_K_100_dim_2_n_datapoints_100000"
	"../UnbiasedSoftmaxData/Simulated/simulated_data_K_1000_dim_2_n_datapoints_1000000_sigma_1"
	"../UnbiasedSoftmaxData/LIBSVM/Delicious_data.txt"
	"""
	hyper_param = 0.01
	repetitions = 5
	time_total = 10**8# 10**5/5 for alpha
	n_eval_loss = 20
	NS_n = 5
	OVE_n = 5
	IS_n = 30
	p2_scale = 1
	alpha = 1.0

	# Create trainer class to run the Trains in
	solver = Solver(data_path , hyper_param, repetitions , time_total, n_eval_loss, NS_n, OVE_n, IS_n, p2_scale, alpha)
	for method in ['DOVE']:#'scikit_learn','DOVE_nonRB','DNS_nonRB','IS','IS_RB','IS','OVE','NS','EXACT',,'DNS','DIS'
		solver.fit(method)
	solver.save_results()
	solver.plot_results('Test')
	#solver.plot_results('Train')




