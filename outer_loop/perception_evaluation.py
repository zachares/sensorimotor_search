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
from task_models import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("perception_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg['logging_params']['display_bool']

	policy_idxs = cfg['decision_params']['policy_idxs']
	num_samples = cfg['decision_params']['num_samples']
	converge_thresh = cfg['decision_params']['converge_thresh']
	constraint_type = cfg['decision_params']['constraint_type']


	use_cuda = cfg['model_params']['use_GPU'] and torch.cuda.is_available()
	info_flow = cfg['info_flow']

	run_mode = cfg['logging_params']['run_mode']
	num_trials = cfg['logging_params']['num_trials']
	step_by_step = cfg['logging_params']['step_by_step']

	dataset_keys = cfg['dataset_keys']

	dataset_path = cfg['dataset_path']

	with open(dataset_path + "datacollection_params.yml", 'r') as ymlfile:
		cfg1 = yaml.safe_load(ymlfile)

	workspace_dim = cfg1['control_params']['workspace_dim']
	kp = np.array(cfg1['control_params']['kp'])
	ctrl_freq = np.array(cfg1['control_params']['control_freq'])
	step_threshold = cfg1['control_params']['step_threshold']
	tol = cfg1['control_params']['tol']
	seed = cfg1['control_params']['seed']

	cfg['control_params'] = cfg1['control_params']

	plus_offset = np.array(cfg1['datacollection_params']['plus_offset'])

	peg_names = cfg1['peg_names']
	hole_names = cfg1['hole_names']
	fit_names = cfg1['fit_names']

	tool_names = peg_names
	substate_names = hole_names
	option_names = fit_names

	num_tools = len(tool_names)
	num_substates = len(substate_names)
	num_options = len(option_names)
    ##################################################################################
    ### Setting Debugging Flag
    ##################################################################################
	if run_mode == 0:
		debugging_flag = True
		run_description = "development"
	else:
		var = input("Run code in debugging mode? If yes, no Results will be saved.[yes,no]: ")
		debugging_flag = False
		if var == "yes":
			debugging_flag = True
			run_description = "evaluation"
		elif var == "no":
			debugging_flag = False
			run_description = "evaluation"
		else:
			raise Exception("Sorry, " + var + " is not a valid input for determine whether to run in debugging mode")

	if debugging_flag:
		print("Currently Debugging")
	else:
		print("Training with debugged code")
	##########################################################
	### Setting up hardware for loading models
	###########################################################
	device = torch.device("cuda" if use_cuda else "cpu")
	random.seed(seed)
	np.random.seed(seed)

	if use_cuda:
	    torch.cuda.manual_seed(seed)
	else:
	    torch.manual_seed(seed)

	if use_cuda:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")
    ##########################################################
    ### Initializing and loading model
	model_dict = declare_models(cfg, "", device)

	sensor = model_dict["Options_Sensor"]
	confusion = model_dict["Options_ConfNet"]
	# confusion = model_dict["Options_ConfMat"]

	sensor.eval()
	confusion.eval()
	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=False, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	env.viewer.set_camera(camera_id=2)

	num_cand = len(list(env.mujoco_objects.keys()))
	############################################################
	### Declaring decision model
	############################################################
	act_model = Action_PegInsertion(workspace_dim, tol, plus_offset, num_samples)

	prob_model = Probability_PegInsertion(sensor, confusion, num_tools, num_substates)

	state_dict = gen_state_dict(num_cand, substate_names, option_names, constraint_type)

	decision_model = Decision_Model(state_dict, prob_model, act_model, policy_idxs)
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	if not debugging_flag:
		logger = Logger(cfg, debugging_flag, False, run_description)
		logging_dict = {}

	obs = {}
	tool_idx = -1
    ############################################################
    ### Starting tests
    ###########################################################
	for trial_num in range(num_trials + 1):
		print("\n")
		print("############################################################")
		print("######                BEGINNING NEW TRIAL              #####")
		print("############################################################")
		print("\n")

		if not debugging_flag:
			logging_dict['scalar'] = {}

		tool_idx += 1
		tool_idx = tool_idx % len(tool_names)
		# tool_idx = 0
		decision_model.tool_idx = tool_idx

		gripper_type = tool_names[tool_idx] + "PegwForce" 

		env.gripper_type =  gripper_type
		env.reset()
		env.viewer.set_camera(camera_id=2)

		if display_bool:
			env.render()

		num_steps = 0

		correct_options = []

		### code necessary for this specific problem, but not the general problem
		if "Fit" not in option_names or "Not_fit" not in option_names:
			print(option_types)
			raise Exception("The code below does not reflect the problem")

		for i in env.hole_sites.keys():
			name = env.hole_sites[i][0]
			tool_name = tool_names[tool_idx]

			if name == tool_name:
				correct_options.append("Fit")
			else:
				correct_options.append("Not_fit")
		###########################################################

		decision_model.reset_probs(substate_names, correct_options)
		decision_model.prob_model.set_tool_idx(tool_idx)
		decision_model.print_hypothesis()
		# a = input("Continue?")

		while True and num_steps < converge_thresh:
			num_steps += 1
			points_list = decision_model.choose_action()

			point_idx = 0
			point_idx, done_bool, obs = movetogoal(env, decision_model.cand_idx, cfg, points_list, point_idx, obs)
			point_idx, done_bool, obs = movetogoal(env, decision_model.cand_idx, cfg, points_list, point_idx, obs)

			# print("moved to initial position")

			obs_dict = {}

			while point_idx < len(points_list):
				point_idx, done_bool, obs, obs_dict = movetogoal(env, decision_model.cand_idx, cfg, points_list, point_idx, obs, obs_dict)

				# if done_bool:
				# 	point_idx = len(points_list)

			# if done_bool:
			# 	for i in range(20):
			# 		move_down(env, display_bool)

			# 	print("\n")
			# 	print("############################################################")
			# 	print("######                INSERTED                         #####")
			# 	print("############################################################")
			# 	print("\n")			
			# 	break

			decision_model.new_obs(obs_dict)
			decision_model.print_hypothesis()
			if step_by_step == 1.0:
				a = input("Continue?")

		if not debugging_flag:

			decision_model.calc_metrics()
			# logging_dict['scalar']["Number of Steps"] = num_steps
			logging_dict['scalar']["Probability of Correct Configuration" ] = decision_model.curr_state_prob
			logging_dict['scalar']["Probability of Correct Fit" ] = decision_model.curr_fit_prob
			logging_dict['scalar']['Entropy of State Distribution'] = decision_model.curr_state_entropy
			logging_dict['scalar']['Entropy of Fit Distribution'] = decision_model.curr_fit_entropy
			# logging_dict['scalar']["Insertion"] = done_bool * 1.0
			# logging_dict['scalar']["Intentional Insertion"] = decision_model.insert_bool

			logger.save_scalars(logging_dict, trial_num, 'evaluation/')
