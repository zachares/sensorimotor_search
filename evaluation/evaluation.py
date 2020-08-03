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
from agent import Outer_Loop
import project_utils as pu
import utils_sl as sl

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
import rlkit
from rlkit.torch import pytorch_util as ptu

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("evaluation_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	constraint_type = cfg['task_params']['constraint_type']
	use_cuda = cfg['model_params']['use_cuda'] and torch.cuda.is_available()

	print("CUDA availability", torch.cuda.is_available())

	# run_mode = cfg['logging_params']['run_mode']
	num_trials = 20 #cfg['logging_params']['num_trials']

	seed = cfg['task_params']['seed']
	ctrl_freq = cfg['control_params']['control_freq']
	horizon = cfg['task_params']['horizon']
	##########################################################
	### Setting up hardware for loading models
	###########################################################
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
    ##########################################################
    ### Initializing and loading model
	kwargs = {
	'use_bandit': cfg['task_params']['use_bandit'],
	'use_state_est': cfg['task_params']['use_state_est'],
	}

	if 'info_flow' in cfg.keys():
		ref_model_dict = pl.get_ref_model_dict()
		model_dict = sl.declare_models(ref_model_dict, cfg, device)	
		
		if 'History_Encoder' in model_dict.keys():
			model_dict['History_Encoder'].eval()
			kwargs['sensor'] = model_dict["History_Encoder"]
			kwargs['encoder'] = model_dict["History_Encoder"]

		if 'Observation_Likelihood_Matrix' in model_dict.keys():
			model_dict['Observation_Likelihood_Matrix'].eval()
			kwargs['likelihood_model'] = model_dict["Observation_Likelihood_Matrix"]

	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)

	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["task_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']

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

	env = SensSearchWrapper(robo_env, cfg, mode_vect = cfg['task_params']['mode_vect'], **kwargs)
	############################################################
	### Declaring decision model
	############################################################
	task_dict = pu.gen_task_dict(len(robo_env.hole_sites.keys()), robo_env.hole_names, robo_env.hole_names, constraint_type = 1)

	decision_model = Outer_Loop(env, task_dict, device = device, **kwargs)
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	# if not debugging_flag:
	# 	logger = Logger(cfg, debugging_flag, False, run_description)
	# 	logging_dict = {}

    ############################################################
    ### Starting tests
    ###########################################################

	step_counts = []

	for trial_num in range(num_trials + 1):
		print("\n")
		print("############################################################")
		print("######                BEGINNING NEW TRIAL              #####")
		print("############################################################")
		print("\n")

		# if not debugging_flag:
		# 	logging_dict['scalar'] = {}

		###########################################################
		decision_model.reset()
		# decision_model.print_hypothesis()
		step_count = decision_model.step_count
		# a = input("Continue?")

		while step_count < cfg['task_params']['max_big_steps']:
			action_idx = decision_model.choose_action()
			
			obs, insertion_bool = decision_model.env.big_step(action_idx, goal = pu.circ_mp2D)
			# print("Step Count ", step_count)
			obs_idxs = decision_model.env.get_obs(obs)

			if insertion_bool:
				# print("Insertion Bool", insertion_bool)
				step_count = cfg['task_params']['max_big_steps']
				continue
			
			decision_model.new_obs(action_idx, obs_idxs)

			step_count = decision_model.step_count


		step_counts.append(step_count)

	print("Mean Number of Steps Per Trial: ", sum(step_counts) / len(step_counts))

		# if not debugging_flag:

		# 	decision_model.calc_metrics()
		# 	# logging_dict['scalar']["Number of Steps"] = num_steps
		# 	logging_dict['scalar']["Probability of Correct Configuration" ] = decision_model.curr_state_prob
		# 	logging_dict['scalar']["Probability of Correct Fit" ] = decision_model.curr_fit_prob
		# 	logging_dict['scalar']['Entropy of State Distribution'] = decision_model.curr_state_entropy
		# 	logging_dict['scalar']['Entropy of Fit Distribution'] = decision_model.curr_fit_entropy
		# 	# logging_dict['scalar']["Insertion"] = done_bool * 1.0
		# 	# logging_dict['scalar']["Intentional Insertion"] = decision_model.insert_bool

		# 	logger.save_scalars(logging_dict, trial_num, 'evaluation/')
