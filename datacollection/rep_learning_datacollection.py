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


if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("rep_learning_datacollection_params.yml", 'r') as ymlfile:
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
	num_trials = cfg['logging_params']['num_trials']

	name = "rep_learning_datacollection_params.yml"

	print("Saving ", name, " to: ", logging_folder + name)

	with open(logging_folder + name, 'w') as ymlfile2:
		yaml.dump(cfg, ymlfile2)


	##########################################################
	### Setting up hardware for loading models
	###########################################################
	random.seed(seed)
	np.random.seed(seed)
	
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

	env = SensSearchWrapper(robo_env, cfg, selection_mode= 1)

	# spiral_mp = pu.Spiral2D_Motion_Primitive()
	# # best parameters from parameter grid search
	# spiral_mp.rt = 0.026
	# spiral_mp.nr = 2.2
	# spiral_mp.pressure = 0.003
	# env.policy = spiral_mp.trajectory

	# for trial_num in range(num_trials):
		# env.reset(initialize = False)
		# env.big_step(env.robo_env.cand_idx)
	step = 0.04
	offset = 0.05

	# env_config = [
	# ["Cross",[0.5 - offset, - (4 + 0.5) * step, 0.82],False],
	# ["Snake",[0.5 + offset, - (3 + 0.5) * step, 0.82],False],
	# ["3_Square",[0.5 - offset,  - (2 + 0.5) * step, 0.82],False],
	# ["2_Rect",[0.5 + offset, - (1 + 0.5) * step, 0.82],False],
	# ["S_Shape",[0.5 - offset, - (0 + 0.5) * step, 0.82],False],
	# ["U_Shape",[0.5 + offset, (0 + 0.5) * step, 0.82],False],
	# ["T_Shape",[0.5 - offset, (1 + 0.5) * step, 0.82],False],
	# ["Triangle",[0.5 + offset, (2 + 0.5) * step, 0.82],False],
	# ["Key",[0.5 - offset, (3 + 0.5) * step, 0.82],False],
	# ["Wind_Mill",[0.5 + offset, (4 + 0.5) * step, 0.82],False]
	# ]

	while env.file_num <= num_trials:
		if env.mode == 0:
			env.mode = 1
		else:
			env.mode = 0
		# if env.done_bool:
		# 	env_config[env.robo_env.cand_idx][2] = True
		env.reset(config_type = '3_small_objects_fit')

		for i in range(cfg['task_params']['horizon']):
			if env.done_bool:
				continue
			noise = np.random.normal(0.0, [noise_scale,noise_scale,noise_scale] , 3)
			env.step(noise)

		for i in range(cfg['task_params']['horizon']):
			if env.done_bool:
				continue
			noise = np.random.normal(0.0, [noise_scale,noise_scale,noise_scale] , 3)
			noise[2] = - abs(noise[2])
			env.step(noise)

		
		# target = env.gen_initpoint(0.06) + env.robo_env.hole_sites[env.robo_env.cand_idx][-3] 

		# for i in range(cfg['task_params']['horizon']):
		# 	if env.done_bool:
		# 		continue
			
		# 	obs = env.robo_env._get_observation()
		# 	goal = target - obs['eef_pos']

		# 	noise = np.random.normal(0.0, [noise_scale,noise_scale,noise_scale] , 3)
		# 	noise[2] = - abs(noise[2])
		# 	env.step(goal + noise)

		env.robo_env.random_seed += 1

			
			# mode = random.choice([0,1,2,3])

			# if mode == 0: 
				
			# elif mode == 1:
			# 	noise[2] = -abs(noise[2])
			# 	env.step(noise)
			# elif mode == 2:
			# 	env.step(goal + noise  * 0.5)
			# elif mode == 3:
			# 	noise[2] = -abs(noise[2])				
			# 	env.step(goal + noise * 0.5)

		# if random.choice([0,0,0,0,0,0,0,0,0,1]) == 1:
		# 	for i in range(cfg['task_params']['horizon']):
		# 	if env.done_bool:
		# 		continue
		# 		noise = np.random.normal(0.0, [noise_scale, noise_scale, noise_scale] , 3)
		# 		obs = env.robo_env._get_observation()
		# 		# obs['rel_pos'][2] += 0.000014 * i
		# 		goal = -200 * obs['rel_pos'][:]
		# 		env.step(goal + 0.5 * noise)
		# else:


		
				# mode = random.choice([0,1,2,3])

				# if mode == 0: 
				# 	env.step(noise)
				# elif mode == 1:
				# 	noise[2] = -abs(noise[2])
				# 	env.step(noise)
				# elif mode == 2:
				# 	
				# elif mode == 3:
				# 	noise[2] = -abs(noise[2])				
				# 	env.step(goal + noise * 0.5)
