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
			cfg['policy_keys'] = cfg['info_flow']['SAC_Policy']['policy_keys']
			cfg['state_size'] = cfg['info_flow']['SAC_Policy']['state_size'] 
		else:
			raise Exception('No policy provided to perform evaluaiton')

		if 'History_Encoder_wUncertainty' in model_dict.keys():
			model_dict['encoder'] = model_dict['History_Encoder_wUncertainty']
			model_dict['sensor'] = model_dict['History_Encoder_wUncertainty']
			print(model_dict['sensor'].loading_folder)

		if 'History_Encoder_wEstUncertainty' in model_dict.keys():
			model_dict['encoder'] = model_dict['History_Encoder_wEstUncertainty']
			model_dict['sensor'] = model_dict['History_Encoder_wEstUncertainty']
			print(model_dict['sensor'].loading_folder)

		if 'History_Encoder_Baseline' in model_dict.keys():
			model_dict['encoder'] = model_dict['History_Encoder_Baseline']
			model_dict['sensor'] = model_dict['History_Encoder_Baseline']
			print(model_dict['sensor'].loading_folder)
	else:
		raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode=2, **model_dict)
	############################################################
	### Declaring decision model
	############################################################
	if env.sensor.num_obs == len(robo_env.fit_names):
		task_dict = pu.gen_task_dict(len(robo_env.hole_sites.keys()), robo_env.hole_names,\
		 robo_env.fit_names, constraint_type = cfg['info_flow']['SAC_Policy']['constraint'])
	elif env.sensor.num_obs == len(robo_env.hole_names):
		task_dict = pu.gen_task_dict(len(robo_env.hole_sites.keys()), robo_env.hole_names,\
		 robo_env.hole_names, constraint_type = cfg['info_flow']['SAC_Policy']['constraint'])
	else:
		raise Exception('unsupported observation space')

	decision_model = Outer_Loop(env, task_dict, mode = cfg['info_flow']['SAC_Policy']['mode'], device = device)
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
	pos_diff_list = []
	prob_diff_list = []
	completion_count = 0
	pos_diverge_count = 0
	state_diverge_count = 0

	num_trials = 81
	trial_num = 81

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
		decision_model.reset()
		# decision_model.print_hypothesis()
		continue_bool = True
		step_count = 0
		# a = input("Continue?")

		while continue_bool:
			action_idx = decision_model.choose_action()
			obs_idxs, obs_logprobs, bad_bool = decision_model.env.big_step(action_idx)
			step_count += 1
			if not bad_bool:
				pos_diff_list.append(decision_model.env.pos_diff)
			# print("Step Count ", step_count)

			if not bad_bool:
				if decision_model.env.done_bool:
					completion_count += 1
					step_counts.append(step_count)
					continue_bool = False
					trial_num -=1
				elif decision_model.env.pos_diverge_bool:
					pos_diverge_count += 1
					continue_bool = False
					trial_num -=1
				elif decision_model.env.state_diverge_bool:
					state_diverge_count += 1
					continue_bool = False
					trial_num -=1
				else:
					decision_model.new_obs(action_idx, obs_idxs, obs_logprobs)
					prob_diff_list.append(decision_model.prob_diff)
			else:
				continue_bool = False
				print("\n\n GOT STUCK \n\n")

	print("Mean Number of Steps Per Trial: ", sum(step_counts) / len(step_counts))
	print("Mean Change in Position Error: ", sum(pos_diff_list) / len(pos_diff_list))
	print("Mean Correct Prob Change: ", sum(prob_diff_list) / len(prob_diff_list))
	print("Completion Rate: ", completion_count / num_trials)
	print("Pos Divergence Rate: ", pos_diverge_count / num_trials)
	print("State Divergence Rate: ", state_diverge_count / num_trials)

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
