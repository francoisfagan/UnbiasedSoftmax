
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(method_scores,test_or_train,data_path):


	if 'EXACT' not in method_scores.keys():
		print("Exact method was not run so cannot plot")
		return

	times = [score_time[1] for score_time in method_scores['EXACT'][0]]
	inter_time = 2*(times[1] - times[0])*0.9 # Window around time periods. Multiplied by 0.9 so times[i+1] and times[i] do not overlap
	fig, ax = plt.subplots()
	#for method in method_scores.keys():
	for method in ['EXACT','IS','NS','OVE','DIS','DOVE','DNS']:#
#['EXACT','NS','DNS','DNS_0->1']
	#'OVE','DOVE','NS','DNS','IS','DIS','DOVE_nonRB','DNS_nonRB'
	#['EXACT','IS','NS','OVE','DIS','DOVE','DNS']
	#['EXACT','DIS','DNS_nonRB','DOVE']
	#['EXACT','DIS','DOVE','DNS']

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
			# if method_label == 'DOVE':
			# 	method_label = 'DOVE_unscaled'
			# if method_label == 'DIS':
			# 	method_label = 'DIS_big_step'
			linestyle = 'solid' if (method[0] == "D") else 'dashed'
			if method_label == 'EXACT':
				ax.errorbar(times,scores_time_mean, yerr = scores_time_std, label=method_label, linestyle = 'dotted', marker = "^")
			else:
				ax.errorbar(times,scores_time_mean, yerr = scores_time_std, label=method_label, linestyle = linestyle)

	legend = ax.legend(loc='lower right', shadow=True)


	plt.xlabel('No. inner products')
	plt.ylabel('Score')
	plt.title(r'Eurlex '+test_or_train+' accuracy')#Eurlex#Simulated with $\eta = 0.1$  $\alpha$ effect
	plt.xticks([10000000,55000000,100000000])#[1000000,5500000,10000000][10000,55000,100000]
	#plt.yticks([0.58,0.65,0.72])
	plot_save_name = "_".join(method_scores.keys())+"_"+test_or_train+data_path#+".png"
	#plt.savefig(plot_save_name)
	#fig.tight_layout()
	fig.set_canvas(plt.gcf().canvas)
	fig.set_size_inches(5,4)#(8, 5),(4,5)
	fig.savefig(plot_save_name + ".pdf", format='pdf', bbox_inches='tight')
	plt.show()

def unpickle_data(data_path):

	return method_scores

if __name__ == "__main__":
	test_or_train = "test"


	data_path = "EXACT_hyper_0.010_rep_10_time_100000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_eurlex_train.txt.p"
	method_scores = pickle.load( open( data_path, "rb" ) )


	data_paths_updated = ["NS_hyper_0.010_rep_10_time_100000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_eurlex_train.txt.p",
	"DNS_hyper_0.010_rep_10_time_100000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_eurlex_train.txt.p",
	"IS_hyper_0.010_rep_10_time_100000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_eurlex_train.txt.p",
	"DIS_hyper_0.010_rep_10_time_100000000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_eurlex_train.txt.p"
	]
	for data_path_updated in data_paths_updated:
		method_scores_updated = pickle.load( open( data_path_updated, "rb" ) )
		for method in method_scores_updated.keys():#
			#if not method == 'DOVE': 
			method_scores[method] = method_scores_updated[method]

	# ## For NS alpha plots
	# method_scores = {}
	# method_scores['EXACT'] = pickle.load( open( "EXACT_DNS_hyper_0.100_rep_50_time_20000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_0.00_simulated_data_K_100_dim_2_n_datapoints_100000.p", "rb" ))['EXACT']
	# method_scores['NS'] = pickle.load( open( "EXACT_DNS_hyper_0.100_rep_50_time_20000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_0.00_simulated_data_K_100_dim_2_n_datapoints_100000.p", "rb" ))['DNS']#'DNS_0'
	# method_scores['DNS'] = pickle.load( open( "EXACT_DNS_hyper_0.100_rep_50_time_20000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_simulated_data_K_100_dim_2_n_datapoints_100000.p", "rb" ))['DNS']
	# method_scores['DNS_0->1'] = pickle.load( open( "gradualEXACT_DNS_hyper_0.100_rep_50_time_20000_n_eval_loss_10_NS_n_5_OVE_n_5_p2_scale_1_alpha_1.00_simulated_data_K_100_dim_2_n_datapoints_100000.p", "rb" ))['DNS']
	# data_path = "alpha_values"

	plot_results(method_scores,test_or_train,data_path)
