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

sys.path.insert(0, "../../robosuite/") 
sys.path.insert(0, "../learning/") 
sys.path.insert(0, "../../supervised_learning/")
sys.path.insert(0, "../")

from models import *
from logger import Logger
from decision_model import *
from project_utils import *
from supervised_learning_utils import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
import rlkit
from rlkit.torch import pytorch_util as ptu

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("perception_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg['logging_params']['display_bool']
	collect_vision_bool = cfg['logging_params']['collect_vision_bool']
	converge_thresh = cfg['decision_params']['converge_thresh']
	constraint_type = cfg['decision_params']['constraint_type']

	use_cuda = cfg['model_params']['use_GPU'] and torch.cuda.is_available()

	print("CUDA availability", torch.cuda.is_available())

	run_mode = cfg['logging_params']['run_mode']
	num_trials = cfg['logging_params']['num_trials']

	seed = cfg['control_params']['seed']
	ctrl_freq = cfg['control_params']['control_freq']
	horizon = cfg['control_params']['horizon']
    ##################################################################################
    ### Setting Debugging Flag
    ##################################################################################
	# if run_mode == 0:
	# 	debugging_flag = True
	# 	run_description = "development"
	# else:
	# 	var = input("Run code in debugging mode? If yes, no Results will be saved.[yes,no]: ")
	# 	debugging_flag = False
	# 	if var == "yes":
	# 		debugging_flag = True
	# 		run_description = "evaluation"
	# 	elif var == "no":
	# 		debugging_flag = False
	# 		run_description = "evaluation"
	# 	else:
	# 		raise Exception("Sorry, " + var + " is not a valid input for determine whether to run in debugging mode")

	# if debugging_flag:
	# 	print("Currently Debugging")
	# else:
	# 	print("Training with debugged code")
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
	ref_model_dict = get_ref_model_dict()
	model_dict = declare_models(ref_model_dict, cfg, device)	

	sensor = model_dict["Options_Sensor"]
	likelihood_model = model_dict["Options_LikelihoodNet"]

	sensor.eval()
	likelihood_model.eval()
	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	robo_env = robosuite.make("PandaPegInsertion",\
	 has_renderer=display_bool,\
	ignore_done=True,\
	 use_camera_obs= not display_bool and collect_vision_bool,\
	 has_offscreen_renderer = not display_bool and collect_vision_bool,\
	gripper_visualization=False,\
	 control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce",\
	   controller='position',\
	camera_depth=True,\
	 camera_width=128,\
	 camera_height=128,\
	  horizon = horizon)

	env = SensSearchWrapper(robo_env, cfg, mode = "test", sensor = sensor, likelihood = likelihood_model)
	############################################################
	### Declaring decision model
	############################################################
	state_dict = gen_state_dict(len(robo_env.hole_sites.keys()), robo_env.hole_names, robo_env.fit_names, constraint_type)

	decision_model = Decision_Model(state_dict, env)
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	# if not debugging_flag:
	# 	logger = Logger(cfg, debugging_flag, False, run_description)
	# 	logging_dict = {}

    ############################################################
    ### Starting tests
    ###########################################################
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
		decision_model.print_hypothesis()
		step_count = decision_model.step_count
		# a = input("Continue?")

		while step_count < converge_thresh:
			macro_action = decision_model.choose_action()
			
			obs, insertion_bool = decision_model.env.perform_macroaction(macro_action)

			if insertion_bool:
				step_count = converge_thresh
				continue
			
			decision_model.new_obs(macro_action, obs)

			decision_model.print_hypothesis()
			step_count = decision_model.step_count


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
