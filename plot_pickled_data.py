
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(method_scores,test_or_train):


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

def unpickle_data(data_path):

	return method_scores

if __name__ == "__main__":
	data_path = "DOVE_EXACT_OVE_hyper_0_rep_1_time_100000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_simulated_data_K_1000_dim_2_n_datapoints_1000000_sigma_10"
	test_or_train = "Test"
	method_scores = pickle.load( open( data_path, "rb" ) )
	plot_results(method_scores,test_or_train)

