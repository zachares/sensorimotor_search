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
		else:
			spiral_mp = pu.Spiral2D_Motion_Primitive()
			# best parameters from parameter grid search
			spiral_mp.rt = 0.026
			spiral_mp.nr = 2.2
			spiral_mp.pressure = 0.003
			model_dict['policy'] = spiral_mp.trajectory

		if 'History_Encoder_Transformer' in model_dict.keys():
			model_dict['encoder'] = model_dict['History_Encoder_Transformer']
			model_dict['sensor'] = model_dict['History_Encoder_Transformer']
			print(model_dict['sensor'].loading_folder)
	else:
		raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode= 2, **model_dict)
	############################################################
	### Declaring decision model
	############################################################
	task_dict = pu.gen_task_dict(len(robo_env.hole_sites.keys()), robo_env.hole_names, [['not_insert', 'insert'], robo_env.hole_names],\
	env.sensor.likelihood_model.model.p.detach().cpu().numpy(), env.sensor.success_params.model.p.detach().cpu().numpy(), constraint_type = 1)

	decision_model = Outer_Loop(env, task_dict, device = device)
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
	num_trials = 10

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
			obs_idxs = decision_model.env.big_step(action_idx)
			# print("Step Count ", step_count)

			if decision_model.env.done_bool:
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
