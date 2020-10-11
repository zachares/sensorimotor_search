import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools

import torch
import torch.nn.functional as F

import perception_learning as pl
from logger import Logger
from agent import Joint_POMDP, Separate_POMDP
import project_utils as pu
import utils_sl as sl

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
import rlkit
from rlkit.torch import pytorch_util as ptu

import matplotlib.pyplot as plt
from collections import OrderedDict

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("evaluation_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["task_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']
	seed = cfg['task_params']['seed']
	ctrl_freq = cfg['control_params']['control_freq']
	noise_scale  = 0.5
	cfg['task_params']['noise_scale'] = noise_scale
	logging_folder = cfg["logging_params"]["logging_folder"]
	num_samples = cfg['logging_params']['num_samples']

	name = "estimate_observation_params.yml"

	print("Saving ", name, " to: ", logging_folder + name)

	with open(logging_folder + name, 'w') as ymlfile2:
		yaml.dump(cfg, ymlfile2)

	##########################################################
	### Setting up hardware for loading models
	###########################################################
	use_cuda = cfg['use_cuda']
	device = torch.device("cuda:0" if use_cuda else "cpu")
	random.seed(seed)
	np.random.seed(seed)
	ptu.set_gpu_mode(use_cuda, gpu_id=0)
	ptu.set_device(0)

	if use_cuda:
	    torch.cuda.manual_seed(seed)
	else:
	    torch.manual_seed(seed)

	if use_cuda:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")
	
	##############################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	robo_env = robosuite.make("PandaPegInsertion",\
		has_renderer= display_bool,\
		ignore_done=True,\
		use_camera_obs= not display_bool and collect_vision_bool,\
		has_offscreen_renderer = not display_bool and collect_vision_bool,\
		gripper_visualization=False,\
		control_freq=ctrl_freq,\
		gripper_type ="CrossPegwForce",\
		controller='position',\
		camera_name=camera_name,\
		camera_depth=collect_depth,\
		camera_width=image_size,\
		camera_height=image_size,\
		horizon = horizon)

	if 'info_flow' in cfg.keys():
		ref_model_dict = pl.get_ref_model_dict()
		model_dict = sl.declare_models(ref_model_dict, cfg, device)	

		if 'SAC_Policy' in cfg['info_flow'].keys():
			data = torch.load(cfg['info_flow']['SAC_Policy']["model_folder"] + "itr_" + str(cfg['info_flow']['SAC_Policy']["epoch"]) + ".pkl")
			model_dict['SAC_Policy'] = data['exploration/policy'].to(device)
			model_dict['policy'] = model_dict['SAC_Policy']
			cfg['policy_keys'] = cfg['info_flow']['SAC_Policy']['policy_keys']
			cfg['state_size'] = cfg['info_flow']['SAC_Policy']['state_size'] 
		# else:
		# 	raise Exception('No policy provided to perform evaluaiton')

		for model_name in cfg['info_flow'].keys():
			if model_name == 'SAC_Policy':
				continue

			if 'success_params' in cfg['info_flow'][model_name].keys():
				success_params = cfg['info_flow'][model_name]['success_params']

			if cfg['info_flow'][model_name]['sensor']:
				model_dict['sensor'] = model_dict[model_name]
				print("Sensor: ", model_name)

			if cfg['info_flow'][model_name]['regression_sensor']:
				model_dict['regression_sensor'] = model_dict[model_name]
				print("Regression Sensor", model_name)

			if cfg['info_flow'][model_name]['encoder']:
				model_dict['encoder'] = model_dict[model_name]
				print("Encoder: ", model_name)

	else:
		raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode=2, **model_dict)
	############################################################
	### Declaring decision model
	############################################################
	# if cfg['info_flow']['SAC_Policy']['unconstrained'] == 1:
	# 	decision_model = Separate_POMDP(env, mode = cfg['info_flow']['SAC_Policy']['mode'], device = device,\
	# 	 success_params = cfg['info_flow']['SAC_Policy']['success_params'], horizon = cfg['info_flow']['SAC_Policy']['horizon'])	
	# else: 

	# if 'SAC_Policy' in cfg['info_flow'].keys() and\
	#  'success_params' in cfg['info_flow']['SAC_Policy'].keys():
	# 	success_params = cfg['info_flow']['SAC_Policy']['success_params']
	# elif 'StatePosSensor_wConstantUncertainty' in cfg['info_flow'].keys() and\
	#  'success_params' in cfg['info_flow']['StatePosSensor_wConstantUncertainty'].keys():
	# 	success_params = cfg['info_flow']['StatePosSensor_wConstantUncertainty']['success_params']
	# elif 'success_params' in cfg.keys():
	# 	success_params = cfg['success_params']
	# else:
	# 	success_params = len(self.robo_env.peg_names) * [0.5]

	mode_results = OrderedDict()
	decision_model = Joint_POMDP(env, mode = 0, device = device,\
		 success_params = success_params, horizon = 3)

	mode_list = [0,1,2,3,4]
	mode_list = [2,3,4]

	for mode in mode_list:
		# if mode == 3:
		# 	decision_model.env.sensor = model_dict['regression_sensor']
		# else:
		# 	decision_model.env.sensor = model_dict['sensor']

		decision_model.mode = mode
	    ##################################################################################
	    #### Logging tool to save scalars, images and models during training#####
	    ##################################################################################
		# if not debugging_flag:
		# 	logger = Logger(cfg, debugging_flag, False, run_description)
		# 	logging_dict = {}

	    ############################################################
	    ### Starting tests
	    ###########################################################
		
		step_counts = [[],[],[],[],[]]
		pos_diff_list = []
		prob_diff_list = []
		failure_count = 0
		completion_count = 0
		pos_diverge_count = 0
		state_diverge_count = 0

		num_trials = 20
		trial_num = 20

		while trial_num > 0:

			print("\n")
			print("############################################################")
			print("######                BEGINNING NEW TRIAL              #####")
			print("######         Number of Trials Left: ", trial_num, "       #####")
			print("############################################################")
			print("\n")

			# if not debugging_flag:
			# 	logging_dict['scalar'] = {}

			###########################################################
			decision_model.env.robo_env.reload = False
			decision_model.env.robo_env.random_seed = 0
			decision_model.reset(config_type = '3_small_objects_fit')
			num_boxes = decision_model.env.robo_env.num_boxes
			total_objects = sum(num_boxes)
			pegs = []

			for i, nb in enumerate(num_boxes):
				for _ in range(nb):
					pegs.append(i)

			print(pegs)
			np.random.shuffle(pegs)
			print(pegs)
			peg_idx = pegs[-1]

			decision_model.env.robo_env.reload = True
			decision_model.env.robo_env.prev_cand_idx = None
			decision_model.reset(config_type = '3_small_objects', peg_idx = peg_idx)

			ref_ests = copy.deepcopy(decision_model.env.cand_ests)
			num_complete = 0

			while num_complete < total_objects:	
				# decision_model.print_hypothesis()
				continue_bool = True
				step_count = 0
				# a = input("Continue?")

				while continue_bool:
					action_idx = decision_model.choose_action()

					pos_init = copy.deepcopy(decision_model.env.cand_ests[action_idx][0])

					got_stuck = decision_model.env.big_step(action_idx, ref_ests = ref_ests)

					pos_final = decision_model.env.cand_ests[action_idx][0]			
					
					# print("Step Count ", step_count)

					if not got_stuck:
						step_count += 1

						if decision_model.env.done_bool:				
							obs_idx = 0

							if decision_model.env.obs_state_logprobs is not None:
								new_logprobs = torch.zeros_like(pu.toTorch(decision_model.env.obs_state_logprobs,\
								 decision_model.env.device))

								new_logprobs[0, decision_model.env.robo_env.peg_idx] = 1.0

								new_logprobs = torch.log(new_logprobs / torch.sum(new_logprobs))

								decision_model.new_obs(action_idx, obs_idx, new_logprobs.cpu().numpy())

								step_counts[num_complete].append(step_count)

								if len(pegs) > 1:
									prob_diff_list.append(decision_model.prob_diff / step_count)
									print("shape prob difference ", decision_model.prob_diff / step_count)
							else:
								decision_model.new_obs(action_idx, obs_idx, None)

							continue_bool = False

						else:
							decision_model.new_obs(action_idx, decision_model.env.obs_idx, decision_model.env.obs_state_logprobs)

							pos_actual = decision_model.env.robo_env.hole_sites[action_idx][-3][:2]

							error_init = np.linalg.norm(pos_init - pos_actual)
							error_final = np.linalg.norm(pos_final - pos_actual)

							error_diff = error_final - error_init
							
							error_diff_per_step = error_init - error_final

							if mode != 3:
								pos_diff_list.append(error_diff_per_step)

							print("initial error - final error ", error_diff_per_step)
					else:
						decision_model.env.cand_ests[action_idx][0] += np.random.uniform(low=-0.002, high=0.002, size =  2)
						decision_model.env.robo_env.hole_sites[action_idx][-1][:2] = decision_model.env.cand_ests[action_idx][0]
						decision_model.env.robo_env.reload = True
						decision_model.env.prev_cand_idx = None
						decision_model.reset(config_type = '3_small_objects', peg_idx = pegs[-1])

						print("\n\n GOT STUCK \n\n")

				if decision_model.env.done_bool and len(pegs) > 1:
					prev_peg = pegs[-1]
					pegs = pegs[:-1]
					current_peg = pegs[-1]

					if current_peg != prev_peg:
						decision_model.states = 1 - decision_model.states

					decision_model.env.robo_env.reload = True
					decision_model.env.robo_env.prev_cand_idx = action_idx
					decision_model.reset(config_type = '3_small_objects', peg_idx = pegs[-1])
					decision_model.completed_tasks[action_idx] = 1.0
					num_complete += 1
					decision_model.env.robo_env.random_seed += 1

					# print("State Probs:", decision_model.state_probs.cpu().numpy())
				elif decision_model.env.done_bool:
					num_complete += 1

				print("Action IDX Final", action_idx)
				print(decision_model.completed_tasks)

			trial_num -=1

				# if step_count > 11:
				# 	continue_bool = False
				# 	failure_count += 1
				# 	trial_num -= 1		
		mode_results[mode] = copy.deepcopy(step_counts)

	for k, v in mode_results.items():
		print("Mode: ", k)
		for i, counts in enumerate(v):
			print("Mean Number of Steps Per Trial: ", sum(counts) / len(counts), 'for', str(total_objects - i), ' objects left')
			counts_array = np.array(counts)
			print("Standard Deviation of Number of Steps Per Trial: ", np.std(counts_array))

	print("Mean Change in Position Error: ", sum(pos_diff_list) / len(pos_diff_list))
	print("Mean Correct Prob Change: ", sum(prob_diff_list) / len(prob_diff_list))

	def get_mode_name(mode):
		if mode == 0:
			return 'iterator'
		elif mode == 1:
			return 'vision prior sampling'
		elif mode == 2:
			return 'greedy'
		elif mode == 3:
			return 'regression'
		elif mode == 4:
			return 'POMDP'
		else:
			raise Exception('unsupported decision making mode')

	# Create lists for the plot
	mode_names = [ get_mode_name(mode) for mode in mode_results.keys() ]
	mode_names += ['Total Number of Steps for Consecutive Task']
	
	fig, axes = plt.subplots(2, 3)

	mode_pos = np.arange(len(mode_names) - 1)

	axes_idx = 0

	step_totals = [[],[]]

	for key, step_counts in mode_results.items():
		total_counts = np.zeros(num_trials)
		mode_means = []
		mode_stds = []
		x_labels = []
		x_pos = np.arange(total_objects)

		for i, counts in enumerate(step_counts):
			total_counts += np.array(counts)
			mode_means.append(sum(counts) / len(counts))
			mode_stds.append(np.std(np.array(counts)))
			x_labels.append(str(total_objects - i))

		step_totals[0].append(np.mean(total_counts))
		step_totals[1].append(np.std(total_counts))

		x_idx = axes_idx % 2
		y_idx = axes_idx // 2

		axes[x_idx, y_idx].bar(x_pos, mode_means, yerr=mode_stds, align='center', alpha=0.5, ecolor='black', capsize=10)
		axes[x_idx, y_idx].set_ylabel("Average Number of Outer Loop Steps to Completion")
		axes[x_idx, y_idx].set_xticks(x_pos)
		axes[x_idx, y_idx].set_xticklabels(x_labels)
		axes[x_idx, y_idx].set_xlabel("Number of Objects Left in the Robot's workspace")
		axes[x_idx, y_idx].set_title(mode_names[axes_idx])
		axes[x_idx, y_idx].yaxis.grid(True)

		axes_idx += 1

	print(step_totals[0])
	print(mode_pos)
	print(step_totals[1])		

	x_idx = axes_idx % 2
	
	y_idx = axes_idx // 2

	axes[x_idx, y_idx].bar(mode_pos, step_totals[0], yerr=step_totals[1], align='center', alpha=0.5, ecolor='black', capsize=10)
	axes[x_idx, y_idx].set_ylabel("Average Number of Outer Loop Steps to Completion")
	axes[x_idx, y_idx].set_xticks(mode_pos)
	axes[x_idx, y_idx].set_xticklabels(mode_names[:-1])
	axes[x_idx, y_idx].set_title(mode_names[axes_idx])
	axes[x_idx, y_idx].yaxis.grid(True)

	# plt.tight_layout()
	plt.savefig('5_objects_3_object_types.png')
	plt.show()
