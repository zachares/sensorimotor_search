import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':

	# total_objects = 5
	# mode_results = OrderedDict()
	# mode_results[0] = [[6.5, 3.6, 3.5, 2.5, 1.45],[4.5, 2.72, 2.39, 1.5, 0.74]]
	# mode_results[1] = [[5.95, 6.55, 4.2, 3.45, 1.25],[4.76, 4.08, 5.88, 2.13, 0.536]]
	# mode_results[2] = [[4.8, 3.95, 2.15, 1.75, 1.6],[2.67, 3.88, 1.31, 1.37, 0.735]]
	# mode_results[4] = [[5.6, 4.5, 4.3, 2.35, 1.6],[4.53, 3.34, 4.51, 1.90, 1.56]]
	# step_totals = [[],[]]

	# for k,v in mode_results.items():
	# 	step_totals[0].append(sum(v[0]))
	# 	step_totals[1].append(np.sqrt(sum(np.square(np.array(v[1])))))

	# def get_mode_name(mode):
	# 	if mode == 0:
	# 		return 'iterator'
	# 	elif mode == 1:
	# 		return 'prior sampling'
	# 	elif mode == 2:
	# 		return 'greedy'
	# 	elif mode == 4:
	# 		return 'POMDP'
	# 	else:
	# 		raise Exception('unsupported decision making mode')

	# # Create lists for the plot
	# mode_names = [ get_mode_name(mode) for mode in mode_results.keys() ]
	# mode_names += ['Total Number of Steps for Consecutive Task']
	
	# fig, axes = plt.subplots(2, 3)

	# mode_pos = np.arange(len(mode_names) - 1)

	# axes_idx = 0

	# # step_totals = [[],[]]	
	# # for key, step_counts in mode_results.items():
	# # 	total_counts = np.zeros(num_trials)
	# # 	mode_means = []
	# # 	mode_stds = []
	# # 	x_labels = []
	# # 	x_pos = np.arange(total_objects)
	# # 	for i, counts in enumerate(step_counts):
	# # 		total_counts += np.array(counts)
	# # 		mode_means.append(sum(counts) / len(counts))
	# # 		mode_stds.append(np.std(np.array(counts)))
	# # 		x_labels.append(str(total_objects - i))


	# # 	step_totals[0].append(np.mean(total_counts))
	# # 	step_totals[1].append(np.std(total_counts))

	# for key, value in mode_results.items():
	# 	mode_means, mode_stds = value[0], value[1]
	# 	x_pos = np.arange(total_objects)
	# 	x_labels = [ str(total_objects - i) for i in range(total_objects) ]

	# 	x_idx = axes_idx % 2
	# 	y_idx = axes_idx // 2

	# 	axes[x_idx, y_idx].bar(x_pos, mode_means, yerr=mode_stds, align='center', alpha=0.5, ecolor='black', capsize=10)
	# 	axes[x_idx, y_idx].set_ylabel("Average Number of Outer Loop Steps to Completion")
	# 	axes[x_idx, y_idx].set_xticks(x_pos)
	# 	axes[x_idx, y_idx].set_xticklabels(x_labels)
	# 	axes[x_idx, y_idx].set_xlabel("Number of Objects Left in the Robot's workspace")
	# 	axes[x_idx, y_idx].set_title(mode_names[axes_idx])
	# 	axes[x_idx, y_idx].yaxis.grid(True)

	# 	axes_idx += 1

	# print(step_totals[0])
	# print(mode_pos)
	# print(step_totals[1])		

	# x_idx = axes_idx % 2
	
	# y_idx = axes_idx // 2

	# axes[x_idx, y_idx].bar(mode_pos, step_totals[0], yerr=step_totals[1], align='center', alpha=0.5, ecolor='black', capsize=10)
	# axes[x_idx, y_idx].set_ylabel("Average Number of Outer Loop Steps to Completion")
	# axes[x_idx, y_idx].set_xticks(mode_pos)
	# axes[x_idx, y_idx].set_xticklabels(mode_names[:-1])
	# axes[x_idx, y_idx].set_title(mode_names[axes_idx])
	# axes[x_idx, y_idx].yaxis.grid(True)

	# axes[-1,-1].axis('off')

	# # plt.tight_layout()
	# plt.savefig('5_objects_3_object_types.png')
	# plt.show()


	fig, axes = plt.subplots(1, 2)
	axes_idx = 0

	names = ['transformer', 'cnn', 'lstm']
	name_pos = np.arange(len(names))
	mean_pos_change = [0.0047, 0.0014, 0.0019]
	std_pos_change = [0.0041, 0.0013, 0.0015]
	classification_accuracy = [0.66, 0.48, 0.45]

	axes[0].bar(name_pos,mean_pos_change, yerr=std_pos_change, align='center', alpha=0.5, ecolor='black',capsize=10)
	axes[0].set_ylabel("Average Decrease in Position Error Per Step")
	axes[0].set_xticks(name_pos)
	axes[0].set_xticklabels(names)
	axes[0].set_title('Position Estimation Performance')
	axes[0].set_ylim(0, 0.01)
	axes[0].yaxis.grid(True)

	axes[1].bar(name_pos, classification_accuracy, align='center', alpha=0.5, ecolor='black',capsize=10)
	axes[1].set_ylabel("Classification Accuracy")
	axes[1].set_xticks(name_pos)
	axes[1].set_xticklabels(names)
	axes[1].set_title('Object Type Estimation Performance')
	axes[1].yaxis.grid(True)
	axes[1].set_ylim(0,1.0)

	plt.savefig('architecture_performance.png')
	plt.show()