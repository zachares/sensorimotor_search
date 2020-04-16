import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import copy
import os
import matplotlib.pyplot as plt
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tensorboardX import SummaryWriter

sys.path.insert(0, "../robosuite/") 
sys.path.insert(0, "../models/") 
sys.path.insert(0, "../robosuite/") 
sys.path.insert(0, "../datalogging/") 
sys.path.insert(0, "../supervised_learning/")
sys.path.insert(0, "../data_collection/")

from models import *
from logger import Logger
from decision_model import *
from datacollection_util import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

def move_down(env, display_bool):
	obs, reward, done, info = env.step(np.array([0,0, -0.1]))
	
	if display_bool:
		env.render()

if __name__ == '__main__':

	###################################################
	### Declaring options and required modalities
	###################################################
	peg_types = ["Cross", "Rect", "Square"] # this ordering is very important, it must be the same as the ordering the network was trained using
	obs_keys = [ "force_hi_freq", "proprio", "action", "contact", "joint_pos", "joint_vel" ]
	
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("perception_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)
	display_bool = cfg['perception_params']['display_bool']
	kp = np.array(cfg['perception_params']['kp'])
	noise_parameters = np.array(cfg['perception_params']['noise_parameters'])
	ctrl_freq = np.array(cfg['perception_params']['control_freq'])
	num_trials = cfg['perception_params']['num_trials']

	workspace_dim = cfg['perception_params']['workspace_dim']
	seed = cfg['perception_params']['seed']

	step_threshold = cfg['perception_params']['step_threshold']

	use_cuda = cfg['perception_params']['use_GPU'] and torch.cuda.is_available()

	debugging_val = cfg['debugging_params']['debugging_val']

	info_flow = cfg['info_flow']

	force_size =info_flow['dataset']['outputs']['force_hi_freq'] 
	action_dim =info_flow['dataset']['outputs']['action']
	proprio_size = info_flow['dataset']['outputs']['proprio']
	num_options = info_flow['dataset']['outputs']['num_options']

	pose_size = 3 ### magic number

	num_samples = cfg['perception_params']['num_samples']

	converge_thresh = cfg['perception_params']['converge_thresh']
    ##################################################################################
    ### Setting Debugging Flag
    ##################################################################################
	if debugging_val == 1.0:
		debugging_flag = True
		var = input("Debugging flag activated. No Results will be saved. Continue with debugging [y,n]: ")
		if var != "y":
			debugging_flag = False
	else:
		debugging_flag = False

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
	sensor = Options_Classifier("", "Options_Classifier", info_flow,\
		 force_size, proprio_size, action_dim, num_options, device = device).to(device)
	confusion = Options_Net("", "Options_Confusion", info_flow, pose_size, num_options, device = device).to(device)
	sensor_pred = Options_Net("", "Options_Prediction", info_flow, pose_size, num_options, device = device).to(device)
	confusion_mat = Options_Mat("", "Options_Confusion_Mat", info_flow, pose_size, num_options, device = device).to(device)
	insert_model = Insertion_PredictionResNet("", "Insertion_PredictionResNet", info_flow, pose_size, device = device).to(device)

	sensor.eval()
	confusion.eval()
	confusion_mat.eval()
	sensor_pred.eval()
	insert_model.eval()
	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	env.viewer.set_camera(camera_id=2)

	tol = 0.01 # position tolerance
	tol_ang = 100 #this is so high since we are only doing position control

	ori_action = np.array([np.pi, 0, np.pi])
	plus_offset = np.array([0, 0, 0.04, 0,0,0])
	peg_idx = 0

	hole_poses = []
	for idx, peg_type in enumerate(peg_types):
		peg_top_site = peg_type + "Peg_top_site"
		top = np.concatenate([env._get_sitepos(peg_top_site) - np.array([0, 0, 0.01]), ori_action])
		top_height = top[2]
		hole_poses.append((top, peg_type))

	############################################################
	### Declaring decision model
	############################################################
	decision_model = DecisionModel(hole_poses, num_options, workspace_dim, num_samples,\
	 ori_action, plus_offset, sensor, confusion, confusion_mat, sensor_pred, insert_model)

    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	# logger = Logger(cfg, debugging_flag, False)
	# logging_dict = {}
	obs = {}
	fixed_params = (kp, noise_parameters, tol, tol_ang, step_threshold, display_bool,top_height, obs_keys)
    ############################################################
    ### Starting tests
    ###########################################################
	for trial_num in range(num_trials):
		#choosing random peg
		decision_model.reset_memory()
		decision_model.set_pegidx(peg_idx)
		# logging_dict['scalar'] = {}

		gripper_type = peg_types[peg_idx] + "PegwForce" 

		env.reset(gripper_type)
		env.viewer.set_camera(camera_id=2)
		if display_bool:
			env.render()

		peg_idx += 1
		peg_idx = peg_idx % len(peg_types)

		num_steps = 0

		print("\n")
		print("############################################################")
		print("######                BEGINNING NEW TRIAL              #####")
		print("############################################################")
		print("\n")
		decision_model.print_hypothesis()
		# a = input("Continue?")

		while True and num_steps < 20:
			# decision_model.choose_hole()
			# points_list = decision_model.choose_action()
			points_list = decision_model.choose_both()

			point_idx = 0
			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]

			point_idx, done_bool, obs = movetogoal(env, fixed_params, points_list, point_idx, obs)
			point_idx, done_bool, obs = movetogoal(env, fixed_params, points_list, point_idx, obs)

			# print("moved to initial position")

			obs_dict = {}

			while point_idx < len(points_list):
				point_idx, done_bool, obs, obs_dict = movetogoal(env,fixed_params, points_list, point_idx, obs, obs_dict)

				if done_bool:
					point_idx == len(points_list)

			if done_bool:
				for i in range(20):
					move_down(env, display_bool)

				print("\n")
				print("############################################################")
				print("######                INSERTED                         #####")
				print("############################################################")
				print("\n")			
				break

			decision_model.new_obs(obs_dict)
			decision_model.print_hypothesis()
			num_steps += 1

			# a = input("Continue?")

		# logging_dict['scalar']["Number of Steps"] = num_steps
		# logging_dict['scalar']["Probability of Correct Configuration" ] = decision_model.hole_memory[decision_model.correct_idx]
		# logging_dict['scalar']["Insertion"] = done_bool * 1.0
		# logging_dict['scalar']["Intentional Insertion"] = decision_model.insert_bool

		# logger.save_scalars(logging_dict, trial_num, 'evaluation/')
