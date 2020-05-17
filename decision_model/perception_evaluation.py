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

	policy_number = cfg['decision_params']['policy_number']
	num_samples = cfg['decision_params']['num_samples']
	converge_thresh = cfg['decision_params']['converge_thresh']
	constraint_type = cfg['decision_params']['constraint_type']


	use_cuda = cfg['model_params']['use_GPU'] and torch.cuda.is_available()
	force_size =cfg['model_params']['force_size'] 
	action_size =cfg['model_params']['action_size']
	proprio_size = cfg['model_params']['proprio_size']
	pose_size = cfg['model_params']['pose_size']

	info_flow = cfg['info_flow']

	run_mode = cfg['logging_params']['run_mode']
	num_trials = cfg['logging_params']['num_trials']
	step_by_step = cfg['logging_params']['step_by_step']

	dataset_keys = cfg['dataset_keys']

	dataset_path = cfg['dataset_params']['dataset_path']

	with open(dataset_path + "datacollection_params.yml", 'r') as ymlfile:
		cfg1 = yaml.safe_load(ymlfile)

	workspace_dim = cfg1['control_params']['workspace_dim']
	kp = np.array(cfg1['control_params']['kp'])
	ctrl_freq = np.array(cfg1['control_params']['control_freq'])
	step_threshold = cfg1['control_params']['step_threshold']
	tol = cfg1['control_params']['tol']
	seed = cfg1['control_params']['seed']

	plus_offset = np.array(cfg1['datacollection_params']['plus_offset'])

	tool_types = cfg1['peg_names']
	hole_shapes = cfg1['hole_names']
	option_types = cfg1['fit_names']

	num_tools = len(tool_types)
	num_options = len(option_types)
    ##################################################################################
    ### Setting Debugging Flag
    ##################################################################################
	if run_mode == 0:
		debugging_flag = True
		run_description = "development"
	else:
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
    ##########################################################
	sensor = Options_Sensor("", "Options_Sensor", info_flow, force_size, proprio_size,\
	 action_size, num_tools, num_options, device = device).to(device)
	confusion = Options_ConfNet("", "Options_ConfNet", info_flow, pose_size,\
	 num_tools, num_options, device = device).to(device)
	# confusion = Options_ConfMat("", "Options_ConfMat", info_flow, num_options, device = device).to(device)
	
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

	hole_info = {}
	for i, hole_shape in enumerate(hole_shapes):
		top_site = hole_shape + "Peg_top_site"
		top = env._get_sitepos(top_site)
		print(top)
		top_height = top[2]
		hole_info[i] = {}
		hole_info[i]["pos"] = top
		hole_info[i]["name"] = hole_shape

	num_cand = len(list(hole_info.keys()))
	############################################################
	### Declaring decision model
	############################################################
	act_model = Action_PegInsertion(hole_info, workspace_dim, tol, plus_offset, num_samples)

	prob_model = Probability_PegInsertion(sensor, confusion, num_tools, num_options)

	state_dict = gen_state_dict(num_cand, option_types, constraint_type)

	decision_model = Decision_Model(state_dict, prob_model, act_model, policy_number)
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	if not debugging_flag:
		logger = Logger(cfg, debugging_flag, False, run_description)
		logging_dict = {}

	obs = {}
	tool_idx = -1
	fixed_params = (0, kp, tol, step_threshold, display_bool,top_height, dataset_keys)
    ############################################################
    ### Starting tests
    ###########################################################
	for trial_num in range(num_trials):
		print("\n")
		print("############################################################")
		print("######                BEGINNING NEW TRIAL              #####")
		print("############################################################")
		print("\n")

		if not debugging_flag:
			logging_dict['scalar'] = {}

		# tool_idx += 1
		# tool_idx = tool_idx % len(tool_types)

		gripper_type = tool_types[tool_idx] + "PegwForce" 

		env.reset(gripper_type)
		env.viewer.set_camera(camera_id=2)

		if display_bool:
			env.render()

		num_steps = 0

		correct_option = []

		#### code necessary for this specific problem, but not the general problem
		if "Fit" not in option_types or "Not Fit" not in option_types:
			raise Exception("The code below does not reflect the problem")

		for key in hole_info.keys():
			name = hole_info[key]["name"]
			tool_name = tool_types[tool_idx]

			if name == tool_name:
				correct_option.append("Fit")
			else:
				correct_option.append("Not Fit")
		############################################################

		decision_model.reset_probs(correct_option)
		decision_model.prob_model.set_tool_idx(tool_idx)
		decision_model.print_hypothesis()
		# a = input("Continue?")

		while True and num_steps < converge_thresh:
			num_steps += 1
			points_list = decision_model.choose_action()
			top_goal = decision_model.act_model.hole_info[decision_model.cand_idx]["pos"]

			point_idx = 0
			point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs)
			point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs)

			# print("moved to initial position")

			obs_dict = {}

			while point_idx < len(points_list):
				point_idx, done_bool, obs, obs_dict = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs, obs_dict)

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
			# logging_dict['scalar']["Number of Steps"] = num_steps
			logging_dict['scalar']["Probability of Correct Configuration" ] = decision_model.state_dis[0,decision_model.state_dict["correct_idx"]]
			logging_dict['scalar']['Entropy'] = decision_model.curr_entropy
			logging_dict['scalar']['Num_Misclassified'] = decision_model.num_misclassifcations
			# logging_dict['scalar']["Insertion"] = done_bool * 1.0
			# logging_dict['scalar']["Intentional Insertion"] = decision_model.insert_bool

			logger.save_scalars(logging_dict, trial_num, 'evaluation/')
