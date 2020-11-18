import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

import project_utils as pu

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
import models_modules as mm
import torch

# import rlkit
# from rlkit.torch import pytorch_util as ptu

import perception_learning as pl
import utils_sl as sl

import pickle

import datetime
import time

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("pe_eval_params.yml", 'r') as ymlfile:
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
	test_success = cfg['test_success']
	logging_folder = cfg["logging_params"]["logging_folder"]
	num_samples = cfg['logging_params']['num_samples']
	
	t_now = time.time()
	date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d')
	results_name = date + '_' + cfg['logging_params']['experiment_name']

	if test_success: # testing policy effectiveness
		experiment_type_folder = 'success_rate/'
		print("TESTING POLICY")
		env_mode = 0
		num_samples = 150
		num_steps = []

	else: # testing network accuracy
		experiment_type_folder = 'perception_accuracy/'
		print("TESTING PERCEPTION NETWORKS")
		env_mode = 2
		classification_accuracy = np.zeros(2)
		errors = [[],[],[],[],[],[]]
		innovation = []
		num_samples = 100

	logging_path = logging_folder + experiment_type_folder + results_name + '.pkl'
	##########################################################
	### Setting up hardware for loading models
	###########################################################
	use_cuda = cfg['use_cuda']
	device = torch.device("cuda:0" if use_cuda else "cpu")
	# ptu.set_gpu_mode(use_cuda, gpu_id=0)
	# ptu.set_device(0)

	# if use_cuda:
	#     torch.cuda.manual_seed(seed)
	# else:
	#     torch.manual_seed(seed)

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

		# if 'SAC_Policy' in cfg['info_flow'].keys():
		# 	data = torch.load(cfg['info_flow']['SAC_Policy']["model_folder"] + "itr_" + str(cfg['info_flow']['SAC_Policy']["epoch"]) + ".pkl")
		# 	model_dict['SAC_Policy'] = data['exploration/policy'].to(device)
		# 	model_dict['policy'] = model_dict['SAC_Policy']
		# 	cfg['policy_keys'] = cfg['info_flow']['SAC_Policy']['policy_keys']
		# 	cfg['state_size'] = cfg['info_flow']['SAC_Policy']['state_size'] 
		# else:
		# 	raise Exception('No policy provided to perform evaluaiton')

		for model_name in cfg['info_flow'].keys():
			# if model_name == 'SAC_Policy':
			# 	continue

			if cfg['info_flow'][model_name]['sensor']:
				model_dict['sensor'] = model_dict[model_name]
				print("Sensor: ", model_name)

			# if cfg['info_flow'][model_name]['encoder']:
			# 	model_dict['encoder'] = model_dict[model_name]
			# 	print("Encoder: ", model_name)

	else:
		model_dict = {}
		# raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode = env_mode, **model_dict)

	trial_num = 0
	num_attempts = cfg['num_attempts']

	while trial_num < num_samples:
		print("Trial Num ", trial_num + 1, " out of ", num_samples, " trials")
		env.mode = env_mode
		env.robo_env.reload = False
		env.reset(initialize=False, config_type = '3_small_objects_fit')
		action_idx = env.robo_env.cand_idx
		substate_idx = env.robo_env.hole_idx
		peg_idx = env.robo_env.peg_idx
		pos_init = copy.deepcopy(env.robo_env.hole_sites[action_idx][-1][:2])

		attempt_count = 0

		if test_success:
			while not env.done_bool and attempt_count < 10:
				attempt_count += 1
				# pos_temp_init = copy.deepcopy(env.robo_env.hole_sites[action_idx][-1][:2])

				got_stuck = env.big_step(action_idx)

				if env.done_bool:
					break

				# pos_final = env.robo_env.hole_sites[action_idx][-1][:2]	

				# pos_actual = env.robo_env.hole_sites[action_idx][-3][:2]

				# error_init = np.linalg.norm(pos_temp_init - pos_actual)
				# error_final = np.linalg.norm(pos_final - pos_actual)
				
				# error_diff_per_step = error_init - error_final

				# print("Error Change: ", error_diff_per_step)

				env.robo_env.reload = True
				env.reset(initialize=False, config_type = '3_small_objects_fit')
		else:
			for attempt in range(num_attempts):
				attempt_count += 1

				pos_temp_init = copy.deepcopy(env.robo_env.hole_sites[action_idx][-1][:2])

				got_stuck = env.big_step(action_idx)

				if env.done_bool or got_stuck:
					break	

				pos_final = env.robo_env.hole_sites[action_idx][-1][:2]	
				pos_actual = env.robo_env.hole_sites[action_idx][-3][:2]

				error_init = np.linalg.norm(pos_temp_init - pos_actual)
				error_final = np.linalg.norm(pos_final - pos_actual)

				if attempt_count == 1 and not env.done_bool:
					errors[attempt_count-1].append(min(error_init, 0.04))
					errors[attempt_count].append(min(error_final, 0.04))

					innovation.append([pos_actual - pos_temp_init, pos_final - pos_temp_init])

				elif not env.done_bool:
					errors[attempt_count].append(error_final)
				
				error_diff_per_step = error_init - error_final

				print("Error Change: ", error_diff_per_step)

				env.robo_env.reload = True
				env.reset(initialize=False, config_type = '3_small_objects_fit')

		if test_success and not got_stuck:
			if env.done_bool:
				num_steps.append(attempt_count)
			else:
				num_steps.append(-1)
			trial_num += 1

		elif not got_stuck and (attempt_count > 1 or (num_attempts == 1 and not env.done_bool)):
			pos_final = env.robo_env.hole_sites[action_idx][-1][:2]	

			pos_actual = env.robo_env.hole_sites[action_idx][-3][:2]

			error_init = np.linalg.norm(pos_init - pos_actual)
			error_final = np.linalg.norm(pos_final - pos_actual)
			
			error_diff_per_step = error_init - error_final

			print("Error Change: ", error_diff_per_step)

			if (env.obs_idx == 0 and env.robo_env.peg_idx == env.robo_env.hole_idx) or\
				(env.obs_idx == 1 and env.robo_env.peg_idx != env.robo_env.hole_idx):
				print("Correct Classification")
				classification_accuracy[0] += 1

			else:
				print("Incorrect Classification")
				classification_accuracy[1] += 1

			trial_num += 1

	if test_success:
		print("Saving results_dict to: ", logging_path)
		with open(logging_path, 'wb') as f:
			pickle.dump(num_steps, f, pickle.HIGHEST_PROTOCOL)
	else:
		print("Mean: ", np.mean(np.array(errors[0]) - np.array(errors[1])))
		print("STD: ", np.std(np.array(errors[0]) - np.array(errors[1])))

		# print("Saving results_dict to: ", logging_path)
		# with open(logging_path, 'wb') as f:
		# 	pickle.dump(errors, f, pickle.HIGHEST_PROTOCOL)

		print("Saving results_dict to: ", logging_path)
		with open(logging_path, 'wb') as f:
			pickle.dump(innovation, f, pickle.HIGHEST_PROTOCOL)



