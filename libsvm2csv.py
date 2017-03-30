# Put libsvm data files into csv format
from sklearn.datasets import load_svmlight_file
import os
import pdb

data_path = "../UnbiasedSoftmaxData/ALOI/aloi"
def get_data():
    data = load_svmlight_file(data_path)
    return data[0], data[1]

X, y = get_data()
pdb.set_trace()