
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(method_scores,test_or_train,data_path):


	if 'EXACT' not in method_scores.keys():
		print("Exact method was not run so cannot plot")
		return

	times = [score_time[1] for score_time in method_scores['EXACT'][0]]
	inter_time = (times[1] - times[0])*0.9 # Window around time periods. Multiplied by 0.9 so times[i+1] and times[i] do not overlap
	fig, ax = plt.subplots()
	for method in ['EXACT','OVE','DOVE','NS','DNS','IS','DIS','DOVE_nonRB','DNS_nonRB']:
		if method in method_scores.keys():
			rep_scores_cum_work = method_scores[method]
			flattened_score_times = [score_time for rep in rep_scores_cum_work for score_time in rep]
			flattened_score_times.sort(key = lambda x: x[1]) # So that they are ordered in increasing times
			scores_times = [[score_time[0] for score_time in flattened_score_times if abs(score_time[1]-time)<=inter_time] for time in times]
			scores_time_mean = [np.mean(scores_time) for  scores_time in scores_times]
			scores_time_std = [np.std(scores_time) for  scores_time in scores_times]
			method_label = method
			# if method_label == 'OVE':
			# 	method_label = 'OVE_unscaled'
			# elif method_label == 'DOVE':
			# 	method_label = 'DOVE_unscaled'
			linestyle = 'solid' if (method[0] == "D") else 'dashed'
			ax.errorbar(times,scores_time_mean, yerr = scores_time_std, label=method_label, linestyle = linestyle)

	legend = ax.legend(loc='lower right', shadow=True)

	plt.xlabel('No. inner products')
	plt.ylabel('Score')
	plt.title('Delicious '+test_or_train+' accuracy')#Eurlex#Simulated with $\eta = 0.1$
	plot_save_name = "_".join(method_scores.keys())+"_"+test_or_train+data_path#+".png"
	#plt.savefig(plot_save_name)
	fig.set_canvas(plt.gcf().canvas)
	fig.savefig(plot_save_name + ".pdf", format='pdf')
	plt.show()

def unpickle_data(data_path):

	return method_scores

if __name__ == "__main__":
	data_path = "IS_EXACT_DIS_hyper_0.100000_rep_5_time_100000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_simulated_data_K_100_dim_2_n_datapoints_100000.p"
	#"aDOVE_NS_EXACT_DNS_OVE_hyper_0.010000_rep_5_time_1000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_eurlex_train.txt"
	#"aDOVE_NS_EXACT_DNS_OVE_hyper_0.005000_rep_3_time_10000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_Bibtex_data.txt"
	#"aDOVE_NS_EXACT_DNS_OVE_hyper_0.010000_rep_3_time_10000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_Delicious_data.txt"
	#"aaDOVE_NS_EXACT_DNS_OVE_hyper_0.010000_rep_5_time_10000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_eurlex_train.txt"

	test_or_train = "test"
	method_scores = pickle.load( open( data_path, "rb" ) )

	# data_path_updated = "IS_EXACT_DIS_hyper_0.010000_rep_10_time_10000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_Delicious_data.txt.p"
	# method_scores_updated = pickle.load( open( data_path_updated, "rb" ) )
	# for method in ['IS','DIS',"EXACT"]:
	# 	method_scores[method] = method_scores_updated[method]

	plot_results(method_scores,test_or_train,data_path)
