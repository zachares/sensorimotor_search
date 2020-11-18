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
from agent import Joint_POMDP, Seperate_POMDP
import project_utils as pu
import utils_sl as sl

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
# import rlkit
# from rlkit.torch import pytorch_util as ptu

import matplotlib.pyplot as plt
from collections import OrderedDict

import datetime
import time
import pickle

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("full_task_evaluation_params.yml", 'r') as ymlfile:
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
	logging_folder = cfg["logging_params"]["logging_folder"]
	num_samples = cfg['logging_params']['num_samples']

	##########################################################
	### Setting up hardware for loading models
	###########################################################
	use_cuda = cfg['use_cuda']
	device = torch.device("cuda:0" if use_cuda else "cpu")
	# random.seed(seed)
	# np.random.seed(seed)
	# ptu.set_gpu_mode(use_cuda, gpu_id=0)
	# ptu.set_device(0)

	if use_cuda:
	    torch.cuda.manual_seed(seed)
	else:
	    torch.manual_seed(seed)

	if use_cuda:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")

	t_now = time.time()
	date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d')
	results_name = date + '_' + cfg['logging_params']['experiment_name']

	logging_path = logging_folder + results_name + '.pkl'
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

	success_params = cfg['success_params']

	if 'info_flow' in cfg.keys():
		ref_model_dict = pl.get_ref_model_dict()
		model_dict = sl.declare_models(ref_model_dict, cfg, device)	

		# if 'SAC_Policy' in cfg['info_flow'].keys():
		# 	data = torch.load(cfg['info_flow']['SAC_Policy']["model_folder"] + "itr_" + str(cfg['info_flow']['SAC_Policy']["epoch"]) + ".pkl")
		# 	model_dict['SAC_Policy'] = data['exploration/policy'].to(device)
		# 	model_dict['policy'] = model_dict['SAC_Policy']
		# 	cfg['policy_keys'] = cfg['info_flow']['SAC_Policy']['policy_keys']
		# 	cfg['state_size'] = cfg['info_flow']['SAC_Policy']['state_size'] 
		# # else:
		# # 	raise Exception('No policy provided to perform evaluaiton')

		for model_name in cfg['info_flow'].keys():
			# if model_name == 'SAC_Policy':
			# 	continue

			if cfg['info_flow'][model_name]['sensor']:
				model_dict['sensor'] = model_dict[model_name]
				loglikelihood_matrix_default = model_dict['sensor'].get_loglikelihood_model()[0]
				print("Sensor: ", model_name)

			# if cfg['info_flow'][model_name]['encoder']:
			# 	model_dict['encoder'] = model_dict[model_name]
			# 	print("Encoder: ", model_name)

	else:
		raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode=2, **model_dict)

	seperate = False

	if seperate:
		decision_model = Joint_POMDP(env, mode = 0, device = device,\
			 success_params = success_params, horizon = 3)
	else:
		decision_model = Seperate_POMDP(env, mode = 0, device = device,\
			 success_params = success_params, horizon = 3)

	############################################################
	### Starting tests
	###########################################################
	# mode_list = [0,1,2]
	decision_model.mode = 0
	step_counts = [[],[],[],[],[]]
	num_trials = 100
	trial_num = num_trials
	decision_model.env.random_seed = 1
	decision_model.env.eval_idx = 0
	decision_model.env.robo_env.random_seed = 0

	while trial_num > 0:
		print("\n")
		print("############################################################")
		print("######                BEGINNING NEW TRIAL              #####")
		print("######         Number of Trials Left: ", trial_num, "       #####")
		print("############################################################")
		print("\n")

		decision_model.env.robo_env.reload = False

		decision_model.reset(config_type = '3_small_objects_fit')
		num_boxes = decision_model.env.robo_env.num_boxes
		total_objects = sum(num_boxes)
		pegs = []

		for i, nb in enumerate(num_boxes):
			for _ in range(nb):
				pegs.append(i)

		np.random.shuffle(pegs)

		full_pegs = copy.deepcopy(pegs)

		decision_model.env.robo_env.reload = True
		decision_model.reset(config_type = '3_small_objects_fit', peg_idx = pegs[-1])
		num_complete = 0

		while num_complete < total_objects:	
			# decision_model.print_hypothesis()
			continue_bool = True
			step_count = 0
			# a = input("Continue?")

			while continue_bool and step_count < 30:
				# print("Current Pegs: ",  pegs)
				# print("Full Pegs: ", full_pegs)
				action_idx = decision_model.choose_action()
				got_stuck = decision_model.env.big_step(action_idx)		
				
				# print("Step Count ", step_count)

				if not got_stuck:
					step_count += 1

					if decision_model.env.done_bool:				
						obs_idx = 0
						step_counts[num_complete].append(step_count)

						if decision_model.mode == 0:
							decision_model.mode = 1
							mode_change = True
						else:
							mode_change = False

						new_logprobs = torch.zeros_like(pu.toTorch(loglikelihood_matrix_default, decision_model.env.device))
						new_logprobs[obs_idx, decision_model.env.robo_env.hole_sites[action_idx][1]] = 1.0
						new_logprobs = torch.log(new_logprobs / torch.sum(new_logprobs))
						decision_model.new_obs(action_idx, 0, new_logprobs.cpu().numpy()) 
						continue_bool = False

						if mode_change:
							decision_model.mode = 0

					else:
						decision_model.new_obs(action_idx, decision_model.env.obs_idx, decision_model.env.obs_state_logprobs)
						decision_model.reset(config_type = '3_small_objects_fit', peg_idx = pegs[-1])
				else:
					decision_model.env.cand_ests[action_idx][0] += 100 * np.random.uniform(low=-0.002, high=0.002, size =  2)
					decision_model.env.robo_env.hole_sites[action_idx][-1][:2] = 0.01 * decision_model.env.cand_ests[action_idx][0]
					decision_model.reset(config_type = '3_small_objects_fit', peg_idx = pegs[-1])
					print("\n\n GOT STUCK \n\n")

			if decision_model.env.done_bool and len(pegs) > 1:
				pegs = pegs[:-1]
				decision_model.completed_tasks[action_idx] = 1.0
				decision_model.env.robo_env.completed_tasks = decision_model.completed_tasks.cpu().numpy()
				decision_model.reset(config_type = '3_small_objects_fit', peg_idx = pegs[-1])
				num_complete += 1

				# print("State Probs:", decision_model.state_probs.cpu().numpy())
			elif decision_model.env.done_bool:
				num_complete += 1
				decision_model.completed_tasks[:] = 0
				decision_model.env.robo_env.completed_tasks = decision_model.completed_tasks.cpu().numpy()

			elif step_count >= 30 and len(pegs) > 1:
				step_counts[num_complete].append(-1)
				pegs = pegs[:-1]
				decision_model.completed_tasks[decision_model.corr_action_idx_list[0]] = 1.0
				decision_model.env.robo_env.completed_tasks = decision_model.completed_tasks.cpu().numpy()
				decision_model.reset(config_type = '3_small_objects_fit', peg_idx = pegs[-1])
				decision_model.reset_probs()
				num_complete += 1

			elif step_count >= 30:
				step_counts[num_complete].append(-1)
				num_complete += 1
				decision_model.completed_tasks[:] = 0
				decision_model.env.robo_env.completed_tasks = decision_model.completed_tasks.cpu().numpy()

			print("Action IDX Final", action_idx)
			print("Completed Fitting Tasks", decision_model.completed_tasks.cpu().numpy())

		trial_num -=1

	print("Saving results_dict to: ", logging_path)
	print("Step Counts: ", step_counts)
	with open(logging_path, 'wb') as f:
		pickle.dump(step_counts, f, pickle.HIGHEST_PROTOCOL)






