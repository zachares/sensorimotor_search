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

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

def random_point(workspace_dim, peg_top_site):
	xmin = peg_top_site[0] - workspace_dim
	xmax = peg_top_site[0] + workspace_dim

	ymin = peg_top_site[1] - workspace_dim
	ymax = peg_top_site[1] + workspace_dim

	zmin = peg_top_site[2] - 0.005
	zmax = peg_top_site[2] + 2 * workspace_dim

	x = np.random.uniform(low=xmin, high=xmax, size = 1)
	y = np.random.uniform(low=ymin, high=ymax, size= 1)
	z = np.random.uniform(low=zmin, high=zmax, size= 1)

	return np.array([x[0],y[0],z[0]])

def slidepoints(workspace_dim, top_goal):
	zmin = - 0.00
	# print("Zs: ", zmin)

	theta = np.random.uniform(low=0, high=2*np.pi)
	x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	x = workspace_dim * np.cos(theta)
	y = workspace_dim * np.sin(theta)
	point_1 = np.array([x, y, zmin, x_ang, y_ang, z_ang]) + top_goal
	point_2 = np.array([-x, -y, zmin, x_ang, y_ang, z_ang]) + top_goal

	return [(point_1, 1, "sliding"), (point_2, 2, "sliding")]

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

def concatenate(tensor_dict, peg_type):
	for key in tensor_dict.keys():
		tensor_dict[key] = torch.cat(tensor_dict[key], dim = 1)

	tensor_dict['peg_type'] = peg_type

	return tensor_dict 

if __name__ == '__main__':

	###################################################
	### Declaring options and required modalities
	###################################################
	peg_types = ["Cross", "Rect", "Square"]
	obs_keys = [ "force_hi_freq", "proprio", "action", "contact", "joint_pos", "joint_vel"]
	peg_dict = {}
	
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
	model = Options_ClassifierTransformer("", "Options_ClassifierTransformer", info_flow, force_size, proprio_size,\
	 action_dim, num_options, 0, device = device).to(device)

	model.eval()
	#######################################################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	obs = env.reset()
	env.viewer.set_camera(camera_id=2)

	tol = 0.002 # position tolerance
	tol_ang = 100 #this is so high since we are only doing position control

	for idx, peg_type in enumerate(peg_types):
		peg_bottom_site = peg_type + "Peg_bottom_site"
		peg_top_site = peg_type + "Peg_top_site"

		top = np.concatenate([env._get_sitepos(peg_top_site) - np.array([0, 0, 0.01]), np.array([np.pi, 0, np.pi])])
		bottom = np.concatenate([env._get_sitepos(peg_bottom_site) + np.array([0, 0, 0.05]), np.array([np.pi, 0, np.pi])])

		peg_vector = np.zeros(len(peg_types))
		peg_vector[idx] = 1.0

		peg_dict[peg_type] = [top, bottom, peg_vector]
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
		peg_type = random.choice(peg_types)
		peg_type = peg_types[trial_num]

		hole_type = random.choice(peg_types)

		memory = torch.from_numpy(np.zeros((3,3))).float().to(device)

		env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
		 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
		  gripper_type = peg_type + "PegwForce", controller='position', camera_depth=True)

		obs = env.reset()
		env.viewer.set_camera(camera_id=2)
		converge_count = 0
		glb_cnt = 0
		logging_dict['scalar'] = {}

		while converge_count < converge_thresh:
			top_goal = peg_dict[hole_type][0]
			bottom_goal = peg_dict[hole_type][1]
			top_plus = top_goal + np.array([0, 0, 0.02, 0,0,0])

			peg_vector = peg_dict[peg_type][2]
			hole_vector = peg_dict[hole_type][2]

			# moving to first initial position
			points_list = [(top_plus, 0, "top_plus")] + slidepoints(workspace_dim, top_goal) + [(top_plus, 0, "top_plus")] 
			point_idx = 0

			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]

			step_count = 0

			while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
				action, action_euler = env._pose_err(goal)
				pos_err = kp * action_euler[:3]
				noise = np.random.normal(0.0, 0.1, pos_err.size)
				noise[2] = 0.0
				pos_err += noise_parameter * noise
				obs, reward, done, info = env.step(pos_err)
				obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal

				if display_bool:
					env.render()


				step_count += 1

			point_idx += 1
			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]

			step_count = 0

			while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
				action, action_euler = env._pose_err(goal)
				pos_err = kp * action_euler[:3]
				noise = np.random.normal(0.0, 0.1, pos_err.size)
				noise[2] = 0.0
				pos_err += noise_parameter * noise
				obs, reward, done, info = env.step(pos_err)
				obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal

				if display_bool:
					env.render()

				step_count += 1

			print("moved to initial position")

			step_count = 0
			point_idx += 1
			point_num_steps = 0

			obs_dict = {}

			while step_count < step_threshold and point_idx < len(points_list):
				prev_point_type = copy.deepcopy(point_type)
				goal = points_list[point_idx][0]
				point_type = points_list[point_idx][1]
				print("Translation type is ", points_list[point_idx][2])

				while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
					action, action_euler = env._pose_err(goal)
					pos_err = kp * action_euler[:3]
					noise = np.random.normal(0.0, 0.1, pos_err.size)
					noise[2] = 0.0
					pos_err += noise_parameter * noise

					obs['action'] = env.controller.transform_action(pos_err)

					save_obs(copy.deepcopy(obs), obs_keys, obs_dict)

					obs, reward, done, info = env.step(pos_err)
					obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal

					if display_bool:
						env.render()

					step_count += 1

				point_idx += 1

			sample = concatenate(obs_dict,\
			 torch.from_numpy(peg_vector).float().to(device).unsqueeze(0))

			if sample['force_hi_freq'].size(1) < 10:
				continue
			# else:
			# 	print("force size: ", sample['force_hi_freq'].size())

			hole_idx = hole_vector.argmax()
			options_logits = model.process(sample).squeeze(0)
			print(F.softmax(options_logits, dim = 0))
			memory[hole_idx] += options_logits
			options_probs = F.softmax(memory[hole_idx], dim = 0)

			hole_est_idx = options_probs.max(0)[1].item()
			peg_idx = peg_vector.argmax()
			print("Network thinks it is a ", peg_types[hole_est_idx], " hole when it is a ", peg_types[hole_idx], " hole")
			# print(options_probs)
			# print(memory[hole_idx])

			# list_idx = peg_types.index(hole_type)
			# list_idx = (list_idx + 1) % len(peg_types)
			# hole_type = peg_types[list_idx]

			# if hole_idx == hole_vector.argmax():
			# 	logging_dict['scalar'][trial_str + "/Correct"] = 1
			# else:
			# 	logging_dict['scalar'][trial_str + "/Correct"] = 0

			logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[0] + "_prob" ] = options_probs[0].item()
			logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[1] + "_prob" ] = options_probs[1].item()
			logging_dict['scalar'][peg_type + "_" + hole_type + "/" + peg_types[2] + "_prob" ] = options_probs[2].item()

			logger.save_scalars(logging_dict, glb_cnt, 'evaluation/')
			glb_cnt += 1
			converge_count += 1