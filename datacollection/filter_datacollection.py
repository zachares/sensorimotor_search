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
	with open("filter_datacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["task_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']
	ctrl_freq = cfg['control_params']['control_freq']
	
	noise_scale  = 0.025
	cfg['task_params']['noise_scale'] = noise_scale

	logging_folder = cfg["logging_params"]["logging_folder"]
	num_trials = cfg['logging_params']['num_trials']

	name = "filter_datacollection_params.yml"

	print("Saving ", name, " to: ", logging_folder + name)

	with open(logging_folder + name, 'w') as ymlfile2:
		yaml.dump(cfg, ymlfile2)
	
	#########################################################
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

	robo_env.random_seed = 1234 # so the random seed is different than during evaluation trials
	env.random_seed = 1234 # so the random seed is different than during evaluation trials
	env = SensSearchWrapper(robo_env, cfg, selection_mode=0)

	while not hasattr(env, 'file_num') or env.file_num <= num_trials:
		if env.mode == 0:
			env.mode = 1
		elif env.mode == 1:
			env.mode = 0

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
