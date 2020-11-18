import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('seaborn-talk')
plt.style.use('ggplot')


import numpy as np
from collections import OrderedDict

from matplotlib.pyplot import plot, show, savefig, xlim, figure,\
 ylim, legend, boxplot, setp, axes, grid, title, ylabel, xlabel, scatter, tight_layout

if __name__ == '__main__':
	################################################################################
	############## ms = method stepcount, nss = not sampling stepcount, ss = sampling stepcount #frame by frame stepcount
# 	path_to_load = '/scr-ssd/sens_search_logging/evaluation/success_rate/'

# 	ms_total = []
# 	nss_total = []
# 	ss_total = []
# 	fbf_total = []

# 	with open(path_to_load + '20201026_full_success_rate00.pkl', 'rb') as f:
# 		ms_total.append(pickle.load(f))

# 	with open(path_to_load + '20201026_full_success_rate01.pkl', 'rb') as f:
# 		ms_total.append(pickle.load(f))

# 	with open(path_to_load + '20201026_full_success_rate02.pkl', 'rb') as f:
# 		ms_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_full_success_rate00.pkl', 'rb') as f:
# 		ms_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_full_success_rate01.pkl', 'rb') as f:
# 		ms_total.append(pickle.load(f))

# 	###################################################################################

# 	with open(path_to_load + '20201025_noupdate_woutsampling_success_rate00.pkl', 'rb') as f:
# 		nss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201025_noupdate_woutsampling_success_rate01.pkl', 'rb') as f:
# 		nss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201025_noupdate_woutsampling_success_rate02.pkl', 'rb') as f:
# 		nss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_noupdate_woutsampling_success_rate00.pkl', 'rb') as f:
# 		nss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_noupdate_woutsampling_success_rate01.pkl', 'rb') as f:
# 		nss_total.append(pickle.load(f))

# 	######################################################################################

# 	with open(path_to_load + '20201027_noupdate_wsampling_success_rate00.pkl', 'rb') as f:
# 		ss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201027_noupdate_wsampling_success_rate01.pkl', 'rb') as f:
# 		ss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201027_noupdate_wsampling_success_rate02.pkl', 'rb') as f:
# 		ss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_noupdate_wsampling_success_rate00.pkl', 'rb') as f:
# 		ss_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_noupdate_wsampling_success_rate01.pkl', 'rb') as f:
# 		ss_total.append(pickle.load(f))

# 	#######################################################################################

# 	with open(path_to_load + '20201028_framebyframe_success_rate00.pkl', 'rb') as f:
# 		fbf_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_framebyframe_success_rate01.pkl', 'rb') as f:
# 		fbf_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_framebyframe_success_rate02.pkl', 'rb') as f:
# 		fbf_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_framebyframe_success_rate03.pkl', 'rb') as f:
# 		fbf_total.append(pickle.load(f))

# 	with open(path_to_load + '20201028_framebyframe_success_rate04.pkl', 'rb') as f:
# 		fbf_total.append(pickle.load(f))	

# 	mc_total = []
# 	nsc_total = []
# 	sc_total = []
# 	fbfc_total = []

# 	for j in range(4):
# 		method_count = [0,0,0,0,0,0,0,0,0,0]
# 		nsamp_count = [0,0,0,0,0,0,0,0,0,0]
# 		samp_count = [0,0,0,0,0,0,0,0,0,0]
# 		fbf_count = [0,0,0,0,0,0,0,0,0,0]

# 		method_stepcount = ms_total[j]
# 		not_sampling_stepcount = nss_total[j]
# 		sampling_stepcount = ss_total[j]
# 		framebyframe_stepcount = fbf_total[j]

# 		for i in range(len(method_stepcount)):
# 			m_sc = method_stepcount[i]
# 			ns_sc = not_sampling_stepcount[i]
# 			s_sc = sampling_stepcount[i]
# 			fbf_sc = framebyframe_stepcount[i]

# 			if m_sc != -1:
# 				method_count[m_sc - 1] += 1
# 			if ns_sc != -1:
# 				nsamp_count[ns_sc - 1] += 1
# 			if s_sc != -1:
# 				samp_count[s_sc - 1] += 1
# 			if fbf_sc != -1:
# 				fbf_count[fbf_sc -1] += 1

# 		for i in range(len(method_count)):
# 			if i == 0:
# 				continue
# 			method_count[i] += method_count[i-1]
# 			nsamp_count[i] += nsamp_count[i-1]
# 			samp_count[i] += samp_count[i-1]
# 			fbf_count[i] += fbf_count[i-1]

# 		for i in range(len(method_count)):
# 			method_count[i] /= len(method_stepcount)
# 			nsamp_count[i] /= len(method_stepcount)
# 			samp_count[i] /= len(method_stepcount)
# 			fbf_count[i] /= len(method_stepcount)

# 		mc_total.append(method_count)
# 		nsc_total.append(nsamp_count)
# 		sc_total.append(samp_count)
# 		fbfc_total.append(fbf_count)

# 	mc_total = np.array(mc_total)
# 	nsc_total = np.array(nsc_total)
# 	sc_total = np.array(sc_total)
# 	fbfc_total = np.array(fbfc_total)

# 	# print(mc_total)
# 	# print(np.std(mc_total, axis = 0))

# 	step_count = [1,2,3,4,5,6,7,8,9,10]

# 	plt.plot(step_count, np.mean(mc_total, axis = 0), 'blue')
# 	plt.fill_between(step_count, np.mean(mc_total, axis = 0) - np.std(mc_total, axis = 0),\
# 	 np.mean(mc_total, axis = 0) + np.std(mc_total, axis = 0), alpha=0.3)
# 	plt.plot(step_count, np.mean(nsc_total, axis = 0), 'red')
# 	plt.fill_between(step_count, np.mean(nsc_total, axis = 0) - np.std(nsc_total, axis = 0),\
# 	 np.mean(nsc_total, axis = 0) + np.std(nsc_total, axis = 0), alpha=0.3, color = 'red')
# 	plt.plot(step_count, np.mean(sc_total, axis = 0), 'green')
# 	plt.fill_between(step_count, np.mean(sc_total, axis = 0) - np.std(sc_total, axis = 0),\
# 	 np.mean(sc_total, axis = 0) + np.std(sc_total, axis = 0), alpha=0.3, color = 'green')
# 	plt.plot(step_count, np.mean(fbfc_total, axis = 0), 'k')
# 	plt.fill_between(step_count, np.mean(fbfc_total, axis = 0) - np.std(fbfc_total, axis = 0),\
# 	 np.mean(fbfc_total, axis = 0) + np.std(fbfc_total, axis = 0), alpha=0.3, color = 'grey')
# 	plt.grid(True)
# 	plt.xlim(0,11)
# 	plt.ylim(0,1.1)
# 	plt.xlabel('Number of High Level Steps (Number of Attempts)')
# 	plt.ylabel('Success Rate over 150 Trials')
# 	plt.legend(['Full Approach', 'No Update', 'No Update + Sampling', 'Frame-by-Frame Update'], loc=4)
# 	plt.tight_layout()
# 	plt.savefig(path_to_load + 'Prior_vs_Posterior.png')
# 	plt.show()

# ########################################################################
# 	path_to_load = "/scr-ssd/sens_search_logging/evaluation/perception_accuracy/"

# 	with open(path_to_load + '20201026_full_position_estimation.pkl', 'rb') as f:
# 		med = pickle.load(f)

# 	with open(path_to_load + '20201111_full_withoutresidual_position_estimation00.pkl', 'rb') as f:
# 		mwored = pickle.load(f)

# 	with open(path_to_load + '20201028_framebyframe_position_estimation.pkl', 'rb') as f:
# 		fbfed = pickle.load(f)

# 	fig = figure()
# 	ax = axes()
# 	# hold(True)

# 	# first boxplot pair
# 	bp0 = plt.boxplot(fbfed, positions = [1, 4, 7, 10, 13, 16], widths = 0.6, patch_artist = True, showfliers=False)
# 	for patch in bp0['boxes']:
# 		patch.set(facecolor='pink') 

# 	# setBoxColors(bp)

# 	# second boxplot pair
# 	bp1 = plt.boxplot(mwored, positions = [2, 5, 8, 11, 14, 17], widths = 0.6, patch_artist = True, showfliers=False)
# 	for patch in bp1['boxes']:
# 		patch.set(facecolor='lightgreen') 
# 	# setBoxColors(bp)


# 	# second boxplot pair
# 	bp2 = plt.boxplot(med, positions = [3, 6, 9, 12, 15, 18], widths = 0.6, patch_artist = True, showfliers=False)
# 	for patch in bp2['boxes']:
# 		patch.set(facecolor='lightblue') 

# 	print(np.mean(med[-1]))
# 	# setBoxColors(bp)


# 	# set axes limits and labels
# 	xlim(0,19)
# 	ylim(0,0.05)
# 	grid(True)
# 	ax.set_xticks([2,5,8,11,14,17])
# 	ax.set_xticklabels(['Initial Error', '1 Step', '2 Steps', '3 Steps', '4 Steps', '5 Steps'])


# 	# draw temporary red and blue lines and use them to create a legend
# 	hB, = plot([1,1],'pink')
# 	hG, = plot([1,1],'lightgreen')
# 	hR, = plot([1,1],'lightblue')
# 	legend((hB, hG, hR),('Frame-by-Frame', 'Relative Position Sensor', 'Full Approach'))
# 	hB.set_visible(False)
# 	hR.set_visible(False)
# 	hG.set_visible(False)

# 	# title('Average Performance over 150 Trials on Full Task')
# 	ylabel("Distance Error (m)")

# 	tight_layout()
# 	savefig(path_to_load + 'Sequential_vs_FramebyFrame.png')
# 	show()

# #####################################################################################
# 	path_to_load = "/scr-ssd/sens_search_logging/evaluation/full_task/"

# 	with open(path_to_load + '20201117_failure_wposition_update_stepcounts00.pkl', 'rb') as f:
# 		sc_failure = pickle.load(f)

# 	with open(path_to_load + '20201117_failure_only_stepcounts00.pkl', 'rb') as f:
# 		sc_baseline = pickle.load(f)

# 	with open(path_to_load + '20201117_full_seperate_stepcounts00.pkl', 'rb') as f:
# 		sc_trace = pickle.load(f)

# 	step_counts_baseline = [[],[],[],[],[]]
# 	step_counts_failure = [[],[],[],[],[]]
# 	step_counts_trace = [[],[],[],[],[]]
# 	b_count = 0
# 	f_count = 0
# 	t_count = 0

# 	for i in range(5):
# 		bcount_list = sc_baseline[i]
# 		fcount_list = sc_failure[i]
# 		tcount_list = sc_trace[i]

# 		for num in bcount_list:
# 			if num == -1:
# 				# continue
# 				b_count += 1
# 				step_counts_baseline[i].append(30)
# 			else:
# 				step_counts_baseline[i].append(num)

# 		for num in fcount_list:
# 			if num == -1:
# 				# continue
# 				f_count += 1
# 				step_counts_failure[i].append(30)
# 			else:
# 				step_counts_failure[i].append(num)

# 		for num in tcount_list:
# 			if num == -1:
# 				t_count += 1
# 				step_counts_trace[i].append(30)
# 			else:
# 				step_counts_trace[i].append(num)

# 	counts_all =  [[],[],[],[],[]]

# 	print("B count: ", b_count/ (100 * 5))
# 	print("F count: ", f_count / (100 * 5))
# 	print("T count: ", t_count / (100 * 5))

# 	for i in range(5):
# 		sc_b = []
# 		sc_f = []
# 		sc_t = []
# 		print(len(step_counts_baseline[i]))
# 		print(len(step_counts_failure[i]))
# 		print(len(step_counts_trace[i]))
# 		for j in range(len(step_counts_baseline[i])):
# 			sc_b.append(min(step_counts_baseline[i][j], 30))
# 			sc_f.append(min(step_counts_failure[i][j], 30))
# 			sc_t.append(min(step_counts_trace[i][j], 30))

# 		if i == 0:
# 			counts_all[i].append(np.array(sc_b))
# 			counts_all[i].append(np.array(sc_f))
# 			counts_all[i].append(np.array(sc_t))
# 		else:
# 			counts_all[i].append(np.array(sc_b) + counts_all[i-1][0])
# 			counts_all[i].append(np.array(sc_f) + counts_all[i-1][1])
# 			counts_all[i].append(np.array(sc_t) + counts_all[i-1][2])

# 	for i in range(5):
# 		for j in range(3):
# 			counts_all[i][j] = np.clip(counts_all[i][j], 0, 60)

# 	fig, axes = plt.subplots(5, 1)
# 	counts, bins, patches = axes[0].hist(counts_all[0],10,histtype='bar', cumulative=1, range=(0,60), color=['seagreen', 'pink', 'lightblue'])
# 	axes[0].set_xticks(bins)
# 	axes[0].set_yticks([0,25,50,75,100])
# 	axes[0].set_yticklabels(['0','0.25','0.5','0.75','1.0'])
# 	axes[0].set_ylabel("n=1",rotation='horizontal')
# 	axes[0].yaxis.set_label_coords(1.05,0.4)
# 	axes[0].set_ylim(0.75)
# 	axes[0].set_xlim(0,60)
# 	axes[0].axvline(np.mean(counts_all[0][0]), color = 'darkgreen')
# 	axes[0].axvline(np.mean(counts_all[0][1]), color = 'red')
# 	axes[0].axvline(np.mean(counts_all[0][2]), color = 'blue')

# 	counts, bins, patches = axes[1].hist(counts_all[1],10,histtype='bar', cumulative=1, range=(0,60), color=['seagreen', 'pink', 'lightblue'])
# 	axes[1].set_xticks(bins)
# 	axes[1].set_xlim(0,60)
# 	axes[1].set_ylim(0.75)
# 	axes[1].set_yticks([0,25,50,75,100])
# 	axes[1].set_yticklabels(['0','0.25','0.5','0.75','1.0'])
# 	axes[1].set_ylabel("n=2", rotation='horizontal')
# 	axes[1].yaxis.set_label_coords(1.05,0.4)
# 	axes[1].axvline(np.mean(counts_all[1][0]), color = 'darkgreen')
# 	axes[1].axvline(np.mean(counts_all[1][1]), color = 'red')
# 	axes[1].axvline(np.mean(counts_all[1][2]), color = 'blue')

# 	counts, bins, patches = axes[2].hist(counts_all[2],10,histtype='bar', cumulative=1, range=(0,60), color=['seagreen', 'pink', 'lightblue'])
# 	axes[2].set_xticks(bins)
# 	axes[2].set_xlim(0,60)
# 	axes[2].set_ylim(0.75)
# 	axes[2].set_yticks([0,25,50,75,100])
# 	axes[2].set_yticklabels(['0','0.25','0.5','0.75','1.0'])
# 	axes[2].set_ylabel("n=3", rotation='horizontal')
# 	axes[2].yaxis.set_label_coords(1.05,0.4)
# 	axes[2].axvline(np.mean(counts_all[2][0]), color = 'darkgreen')
# 	axes[2].axvline(np.mean(counts_all[2][1]), color = 'red')
# 	axes[2].axvline(np.mean(counts_all[2][2]), color = 'blue')

# 	counts, bins, patches = axes[3].hist(counts_all[3],10,histtype='bar', cumulative=1,  range=(0,60), color=['seagreen', 'pink', 'lightblue'])
# 	axes[3].set_xticks(bins)
# 	axes[3].set_xlim(0,60)
# 	axes[3].set_ylim(0.75)
# 	axes[3].set_yticks([0,25,50,75,100])
# 	axes[3].set_yticklabels(['0','0.25','0.5','0.75','1.0'])
# 	axes[3].set_ylabel("n=4", rotation='horizontal')
# 	axes[3].yaxis.set_label_coords(1.05,0.4)
# 	axes[3].axvline(np.mean(counts_all[3][0]), color = 'darkgreen')
# 	axes[3].axvline(np.mean(counts_all[3][1]), color = 'red')
# 	axes[3].axvline(np.mean(counts_all[3][2]), color = 'blue')

# 	counts, bins, patches = axes[4].hist(counts_all[4],10,histtype='bar', cumulative=1,  range=(0,60), color=['seagreen', 'pink', 'lightblue'])
# 	axes[4].set_xticks(bins)
# 	axes[4].set_xlim(0,60)
# 	axes[4].set_ylim(0.75)
# 	axes[4].set_yticks([0,25,50,75,100])
# 	axes[4].set_yticklabels(['0','0.25','0.5','0.75','1.0'])
# 	axes[4].set_ylabel("n=5", rotation='horizontal')
# 	axes[4].yaxis.set_label_coords(1.05,0.4)
# 	axes[4].axvline(np.mean(counts_all[4][0]), color = 'darkgreen')
# 	axes[4].axvline(np.mean(counts_all[4][1]), color = 'red')
# 	axes[4].axvline(np.mean(counts_all[4][2]), color = 'blue')


# 	hK, = plt.plot([1,1],'seagreen')
# 	hB, = plt.plot([1,1],'pink')
# 	hR, = plt.plot([1,1],'lightblue')
# 	hj, = plt.plot([1,1],'darkgreen')
# 	hg, = plt.plot([1,1],'red')
# 	hl, = plt.plot([1,1],'blue')
# 	lgd = plt.legend((hK, hB, hR),('Failure Only',\
# 	 'Failure + Position', 'Full Approach', 'No Update Based on Trace Mean',\
# 	 'Update Position Only Based on Trace Mean', 'Update Position and Type Based on Trace Mean'),\
# 	  loc='lower center', bbox_to_anchor=(0.13, 0.0825), framealpha=0)
# 	hK.set_visible(False)
# 	hB.set_visible(False)
# 	hR.set_visible(False)
# 	hj.set_visible(False)
# 	hl.set_visible(False)
# 	hg.set_visible(False)
# 	axes[0].get_xaxis().set_visible(False)
# 	axes[1].get_xaxis().set_visible(False)
# 	axes[2].get_xaxis().set_visible(False)
# 	axes[3].get_xaxis().set_visible(False)
# 	# axes[4].set_xlim(0,31)
# 	axes[0].xaxis.grid(False)
# 	axes[1].xaxis.grid(False)
# 	axes[2].xaxis.grid(False)
# 	axes[3].xaxis.grid(False)
# 	axes[4].xaxis.grid(False)
# 	txt = fig.text(-0.01, 0.5, 'Success Rate', size=12, color='dimgrey', va='center', rotation='vertical')
# 	fig.tight_layout()
# 	plt.xlabel('Number of High Level Steps (Number of Attempts)')
# 	# plt.ylabel('Number of Trials which Completed Task within Bin Range')
# 	# axes[0].x_label('5 Objects Left')
# 	plt.savefig(path_to_load + 'failure_vs_trace_cdf.png', bbox_extra_artists=(lgd,txt,), bbox_inches='tight')
# 	plt.show()

#####################################################################
	path_to_load = "/scr-ssd/sens_search_logging/evaluation/perception_accuracy/"

	with open(path_to_load + '20201114_full_withresidual_innovation00.pkl', 'rb') as f:
		med = pickle.load(f)

	with open(path_to_load + '20201114_full_withoutresidual_innovation00.pkl', 'rb') as f:
		mwored = pickle.load(f)

	dm = []
	dmwor = []
	mm = [[],[],[],[],[],[]]
	mmwor = [[],[],[],[],[],[]]
	ranges = [0, np.pi / 6, 2 * np.pi / 6, 3 * np.pi / 6, 4 * np.pi / 6,  5 * np.pi / 6, np.pi]

	for i in range(len(med)):
		m_err = med[i][0]
		m_pu = med[i][1]

		unit_vector_1 = m_err / np.linalg.norm(m_err)
		unit_vector_2 = m_pu / np.linalg.norm(m_pu)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product)

		bin_idx = 0

		for j in range(1,len(ranges)):
			if angle >= ranges[j-1] and angle < ranges[j]:
				bin_idx = j - 1

		# print(bin_idx)
		mm[bin_idx].append(np.linalg.norm(m_pu))

		mwor_err = mwored[i][0]
		mwor_pu = mwored[i][1]

		unit_vector_1 = mwor_err / np.linalg.norm(mwor_err)
		unit_vector_2 = mwor_pu / np.linalg.norm(mwor_pu)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product)

		bin_idx = 0

		for j in range(1,len(ranges)):
			if angle >= ranges[j-1] and angle < ranges[j]:
				bin_idx = j - 1

		mmwor[bin_idx].append(np.linalg.norm(mwor_pu))

	fig = figure()
	ax = axes()
	# hold(True)

	bp0 = boxplot(mmwor, positions = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5], widths = 0.6, patch_artist = True, showfliers=False)
	for patch in bp0['boxes']:
		patch.set(facecolor='lightgreen') 

	bp1 = boxplot(mm, positions = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5], widths = 0.6, patch_artist = True, showfliers=False)
	for patch in bp1['boxes']:
		patch.set(facecolor='lightblue') 

	# set axes limits and labels
	xlim(0,12)
	ylim(0,0.025)
	grid(True)
	ax.set_xticks([0,2,4,6,8,10,12])
	ax.set_xticklabels(['0', '\u03C0/6', '2\u03C0/6', '3\u03C0/6', '4\u03C0/6', '5\u03C0/6', '\u03C0'])


	# draw temporary red and blue lines and use them to create a legend
	hG, = plot([1,1],'lightgreen')
	hR, = plot([1,1],'lightblue')
	legend((hG, hR),('Relative Position Sensor', 'Full Approach'))
	hG.set_visible(False)
	hR.set_visible(False)

	# title('Update Magnitude as a Function of the Angle between the Position Update and the true Position Error')
	ylabel("Update Magnitude (m)")
	xlabel("Angle between Position Update and True Position Error")

	tight_layout()
	savefig(path_to_load + 'Relative_Positon_Sensor_vs_Position_Error_Sensor.png')
	show()

##################################################
	## buggy results
	# path_to_load = "/scr-ssd/sens_search_logging/evaluation/full_task/"

	# failure_trial_list = []
	# trace_trial_list = []
	# baseline_trial_list = []

	# #### failure + position update trials
	# with open(path_to_load + '20201028_failure_wposition_update_stepcounts00.pkl', 'rb') as f:
	# 	failure_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_wposition_update_stepcounts01.pkl', 'rb') as f:
	# 	failure_trial_list.append(pickle.load(f))

	# #### trace trials
	# # joint state space
	# with open(path_to_load + '20201028_full_stepcounts00.pkl', 'rb') as f:
	# 	trace = pickle.load(f)

	# # seperate state space
	# with open(path_to_load + '20201027_full_seperate_stepcounts00.pkl', 'rb') as f:
	# 	trace_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201029_full_seperate_stepcounts01.pkl', 'rb') as f:
	# 	trace_trial_list.append(pickle.load(f))

	# #### failure only trials
	# with open(path_to_load + '20201028_failure_only_stepcounts00.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_only_stepcounts01.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_only_stepcounts02.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_only_stepcounts03.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_only_stepcounts04.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# with open(path_to_load + '20201028_failure_only_stepcounts05.pkl', 'rb') as f:
	# 	baseline_trial_list.append(pickle.load(f))

	# sc_failure = [[],[],[],[],[]]

	# for i in range(5):
	# 	for j in range(len(failure_trial_list)):
	# 		sc_failure[i] += failure_trial_list[j][i]

	# sc_trace = [[],[],[],[],[]]

	# for i in range(5):
	# 	for j in range(len(trace_trial_list)):
	# 		sc_trace[i] += trace_trial_list[j][i]

	# sc_baseline = [[],[],[],[],[]]

	# for i in range(5):
	# 	for j in range(len(baseline_trial_list)):
	# 		sc_baseline[i] += baseline_trial_list[j][i]