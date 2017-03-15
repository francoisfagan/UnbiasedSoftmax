# -*- coding: utf-8 -*-

"""
    Main driver script for experiments
"""


import pdb
import numpy, sys
import matplotlib.pyplot as plt
from sklearn import linear_model

def readHyperParameters():
	# Load hyperparameters from the terminal
    if len(sys.argv) > 1:
        data_set = sys.argv[1]
        max_iter = int(sys.argv[2])
        hyper_param = float(sys.argv[3])
        repetitions = int(sys.argv[4])

	    # Set path to training and testing data
	    train_path = data_set + "_train.csv"
	    test_path  = data_set + "_test.csv"

        return [train_path, test_path , max_iter , hyper_param , repetitions]
    else:
        print"""
        ------------------------------------
        Recommended defaults:
        max_iter = 5000
        hyper_param = 0.01
        repetitions = 1
        ------------------------------------
        """
        exit(0)

def numpy_read(fpath):
    return numpy.genfromtxt(fpath, delimiter=',')

class Train:

    def __init__(self, train_path, hyper_param, repetitions=1, max_iter=1000, eval_delay=100):
        self.train_data = numpy_read(train_path)
        self.hyper_param = hyper_param
        self.repetitions = repetitions
        self.max_iter = max_iter
        self.eval_delay = eval_delay

        self.param_shape = (self.train_data.shape[1],1)
        self.K = #range of K from 0 to K-1

    def train(self, name):
    	print(name)
	    param_history = numpy.zeros(self.repetitions , self.param_shape , self.max_iter/self.eval_delay )

        trainer_repetitions = [] 
        for repeat in range(repetitions):
    		print(repeat)
        	params = numpy.zeros(self.param_shape)
	        for i in xrange(0,self.max_iter):
	        	data_point = random.rand(); # CORRECT THIS! Instead of doing this, maybe just cycle through the data?
	        	gradient = update(name, self.train_data[data_point])
	            params -= gradient
	            if (i%round(self.eval_delay+1) == 0):
	                self.param_history.append(numpy.array(params))

        # Export results

class Test:

    def __init__(self, test_path, opt_results):
    	self.test_data = numpy_read(test_path)

	def plot(self):


	    


if __name__ == "__main__":

	# Read hyperparameters from the terminal
	train_path, test_path , max_iter , hyper_param , repetitions = readHyperParameters()

    # Create trainer class to run the Trains in
    trainer = Train(train_path , hyper_param, repetitions, max_iter, eval_delay)

    # Run the trainers
    opt_choices = ["Softmax"]
    opt_results = { opt_choice : trainer.train(opt_choice) for opt_choice in opt_choices } # Stores the results for each trainer in an array

	# Solve using in-built scikit-learn methods
	opt_results["Scikit" : ]

    # Create tester class to test the algorithms
    tester = Test(test_path , opt_results)
    tester.evaluate_historical_loss()
    tester.plot()


