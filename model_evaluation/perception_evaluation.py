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

from models import *
from logger import Logger
from decision_model import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4

def productofgauss(mean0, mean1, cov0, cov1):
	inv_cov0 = torch.inverse(cov0)
	inv_cov1 = torch.inverse(cov1)
	cov = torch.inverse(inv_cov0 + inv_cov1)
	mean = torch.mm(cov, torch.mm(inv_cov0, mean0.unsqueeze(1))) + torch.mm(cov, torch.mm(inv_cov1, mean1.unsqueeze(1)))
	return mean.squeeze(), cov

def save_obs(obs_dict, keys, tensor_dict):
	for key in keys:
		if key == "rgbd":
			obs0 = np.rot90(obs_dict['image'], k = 2).astype(np.unint8)
			obs0 = resize(obs, (128, 128))
			obs0 = np.transpose(obs, (2, 1, 0))
			obs0 = np.expand_dims(obs, axis = 0)
			obs1 = np.rot90(obs_dict['depth'], k = 2).astype(np.unint8)
			obs1 = resize(obs, (128, 128))
			obs1 = np.expand_dims(np.expand_dims(obs, axis = 0), axis = 0)
			obs = np.concatenate([obs0, obs1], dim = 1)
		else:
			obs = np.expand_dims(obs_dict[key], axis = 0)

		if key in tensor_dict.keys():
			tensor_dict[key].append(torch.from_numpy(obs).float().to(device).unsqueeze(0))
		else:
			tensor_dict[key] = [torch.from_numpy(obs).float().to(device).unsqueeze(0)]

def movetogoal(env, points_list, tol, tol_ang, step_threshold, display_bool, point_idx, obs, obs_dict = None):
	step_count = 0
	goal = points_list[point_idx][0]
	point_type = points_list[point_idx][1]

	while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
		action, action_euler = env._pose_err(goal)
		pos_err = kp * action_euler[:3]
		noise = np.random.normal(0.0, 0.1, pos_err.size)
		noise[2] = -1.0 * abs(noise[2])
		pos_err += noise_parameter * noise

		obs['action'] = env.controller.transform_action(pos_err)

		if point_type == 1 and obs_dict is not None:
			save_obs(copy.deepcopy(obs), obs_keys, obs_dict)
			# curr_pose = env._get_eepos()

		obs, reward, done, info = env.step(pos_err)
		# obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal
		
		if display_bool:
			env.render()


		step_count += 1

	point_idx += 1

	if obs_dict is not None:
		return point_idx, obs, obs_dict
	else:
		return point_idx, obs

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
	noise_parameter = np.array(cfg['perception_params']['noise_parameter'])
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
	eval_model = Options_ClassifierTransformer("", "Options_ClassifierTransformer", info_flow, force_size, proprio_size,\
	 action_dim, num_options, 0, device = device).to(device)
	pred_model = Options_PredictionResNet("", "Options_PredictionResNet", info_flow, pose_size, num_options, device = device).to(device)
	# origin_model = Origin_DetectionTransformer("", "Origin_DetectionTransformer", info_flow,\
	# 	 force_size, proprio_size, action_dim, num_options, 0, device = device).to(device)
	eval_model.eval()
	pred_model.eval()
	# origin_model.eval()
	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	obs = env.reset()
	env.viewer.set_camera(camera_id=2)

	tol = 0.01 # position tolerance
	tol_ang = 100 #this is so high since we are only doing position control

	ori_action = np.array([np.pi, 0, np.pi])
	plus_offset = np.array([0, 0, 0.04, 0,0,0])

	hole_poses = []
	for idx, peg_type in enumerate(peg_types):
		peg_top_site = peg_type + "Peg_top_site"
		top = np.concatenate([env._get_sitepos(peg_top_site) - np.array([0, 0, 0.01]), ori_action])
		hole_poses.append((top, peg_type))

	############################################################
	### Declaring decision model
	############################################################
	decision_model = DecisionModel(hole_poses, num_options, workspace_dim, num_samples,\
	 ori_action, plus_offset, pred_model, eval_model)

    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
	save_model_flag = False
	logger = Logger(cfg, debugging_flag, save_model_flag)
	logging_dict = {}
    ############################################################
    ### Starting tests
    ###########################################################
	for trial_num in range(num_trials):
		#choosing random peg
		decision_model.reset_memory()
		peg_idx = random.choice(range(len(hole_poses)))
		decision_model.set_pegidx(peg_idx)

		env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
		 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
		  gripper_type = peg_types[peg_idx] + "PegwForce", controller='position', camera_depth=True)

		obs = env.reset()
		env.viewer.set_camera(camera_id=2)

		converge_count = 0
		# glb_cnt = 0
		# logging_dict['scalar'] = {}

		while converge_count < converge_thresh:
			decision_model.choose_hole()
			points_list = decision_model.choose_both()

			point_idx = 0
			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]

			point_idx, obs = movetogoal(env, points_list, tol, tol_ang, step_threshold, display_bool, point_idx, obs)
			point_idx, obs = movetogoal(env, points_list, tol, tol_ang, step_threshold, display_bool, point_idx, obs)

			print("moved to initial position")

			obs_dict = {}

			while point_idx < len(points_list):
				point_idx, obs, obs_dict = movetogoal(env, points_list, tol, tol_ang, step_threshold, display_bool, point_idx, obs, obs_dict)

			decision_model.new_obs(obs_dict)
			prob = decision_model.print_hypothesis()

			if prob == 1:
				converge_count = converge_thresh
			# logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[0] + "_prob" ] = options_probs[0].item()
			# logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[1] + "_prob" ] = options_probs[1].item()
			# logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[2] + "_prob" ] = options_probs[2].item()

			# # logger.save_scalars(logging_dict, glb_cnt, 'evaluation/')
			# glb_cnt += 1

			converge_count += 1