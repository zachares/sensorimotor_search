import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools

from project_utils import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
			
if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("senssearchdatacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	num_trials = cfg['logging_params']['num_trials']
	seed = cfg['control_params']['seed']
	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["control_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']

	logging_folder = cfg["logging_params"]["logging_folder"]
	name = "datacollection_params"

	print("Saving ", name, " to: ", logging_folder + name + ".yml")

	with open(logging_folder + name + ".yml", 'w') as ymlfile2:
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

	env = SensSearchWrapper(robo_env, cfg, mode = 'choose_iter')
	############################################################
	### Starting tests
	###########################################################
	for trial_num in range(num_trials + 1):
		env.reset()
		env.mp_action()


