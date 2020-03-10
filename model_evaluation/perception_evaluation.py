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

# def random_point(workspace_dim, peg_top_site):
# 	xmin = peg_top_site[0] - workspace_dim
# 	xmax = peg_top_site[0] + workspace_dim

# 	ymin = peg_top_site[1] - workspace_dim
# 	ymax = peg_top_site[1] + workspace_dim

# 	zmin = peg_top_site[2] - 0.005
# 	zmax = peg_top_site[2] + 2 * workspace_dim

# 	x = np.random.uniform(low=xmin, high=xmax, size = 1)
# 	y = np.random.uniform(low=ymin, high=ymax, size= 1)
# 	z = np.random.uniform(low=zmin, high=zmax, size= 1)

# 	return np.array([x[0],y[0],z[0]])

# def slidepoints(workspace_dim, top_goal):
# 	zmin = - 0.00
# 	# print("Zs: ", zmin)

# 	theta = np.random.uniform(low=0, high=2*np.pi)
# 	x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
# 	y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
# 	z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
# 	x = workspace_dim * np.cos(theta)
# 	y = workspace_dim * np.sin(theta)
# 	point_1 = np.array([x, y, zmin, x_ang, y_ang, z_ang]) + top_goal
# 	point_2 = np.array([-x, -y, zmin, x_ang, y_ang, z_ang]) + top_goal

# 	return [(point_1, 1, "sliding"), (point_2, 2, "sliding")]

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4

def slidepoints(workspace_dim, num_trajectories = 10):
	zmin = - 0.00
	# print("Zs: ", zmin)

	theta_init = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
	theta_delta = np.random.uniform(low=3 * np.pi / 4, high=np.pi, size = num_trajectories)
	theta_sign = np.random.choice([-1, 1], size = num_trajectories)
	theta_final = T_angle(theta_init + theta_delta * theta_sign)

	c_point_list = []

	for idx in range(theta_init.size):
		x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		theta0 = theta_init[idx]
		theta1 = theta_final[idx]
		x_init = workspace_dim * np.cos(theta0)
		y_init = workspace_dim * np.sin(theta0)
		x_final = workspace_dim * np.cos(theta1)
		y_final = workspace_dim * np.sin(theta1) 

		# print("Initial point: ", x_init, y_init)
		# print("Final point: ", x_final, y_final)
		c_point_list.append(np.expand_dims(np.array([x_init, y_init, zmin, x_final, y_final, zmin]), axis = 0))

	return np.concatenate(c_point_list, axis = 0)

def multinomial_KL(logits_q, logits_p):

	return -(F.softmax(logits_p, dim =1) * (F.log_softmax(logits_q, dim = 1) - F.log_softmax(logits_p, dim = 1))).sum(1)

def productofgauss(mean0, mean1, cov0, cov1):
	inv_cov0 = torch.inverse(cov0)
	inv_cov1 = torch.inverse(cov1)
	cov = torch.inverse(inv_cov0 + inv_cov1)
	mean = torch.mm(cov, torch.mm(inv_cov0, mean0.unsqueeze(1))) + torch.mm(cov, torch.mm(inv_cov1, mean1.unsqueeze(1)))
	return mean.squeeze(), cov

def eval_points(pred_logits):
	hole_0 = pred_logits[:,0]
	hole_1 = pred_logits[:,1]
	hole_2 = pred_logits[:,2]

	kl01 = 0.5 * (multinomial_KL(hole_0, hole_1) + multinomial_KL(hole_1, hole_0))
	kl02 = 0.5 * (multinomial_KL(hole_0, hole_2) + multinomial_KL(hole_2, hole_0))
	kl12 = 0.5 * (multinomial_KL(hole_1, hole_2) + multinomial_KL(hole_2, hole_1))

	kl_sum = kl01 + kl02 + kl12

	return kl_sum.max(0)[1]

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

def concatenate(tensor_dict, peg_type, hole_type, error):
	for key in tensor_dict.keys():
		tensor_dict[key] = torch.cat(tensor_dict[key], dim = 1)

	tensor_dict['pose'] = tensor_dict['proprio'][:,:,:3]
	tensor_dict['init_pose'] = tensor_dict['pose'][:,0]
	tensor_dict['final_pose'] = tensor_dict['pose'][:,-1]
	tensor_dict['pose_delta'] = tensor_dict['final_pose'] - tensor_dict['init_pose']
	tensor_dict['pose_vect'] = torch.cat([tensor_dict['init_pose'] + error, tensor_dict['final_pose'] + error, tensor_dict['pose_delta']], dim = 1)

	tensor_dict['peg_type'] = peg_type
	tensor_dict['hole_type'] = hole_type

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
	pose_size = 3
	num_candidates = 50

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
	options_model = Options_ClassifierTransformer("", "Options_ClassifierTransformer", info_flow, force_size, proprio_size,\
	 action_dim, num_options, 0, device = device).to(device)
	eval_model = Options_PredictionResNet("", "Options_PredictionResNet", info_flow, pose_size, num_options, device = device).to(device)
	origin_model = Origin_DetectionTransformer("", "Origin_DetectionTransformer", info_flow,\
		 force_size, proprio_size, action_dim, num_options, 0, device = device).to(device)

	options_model.eval()
	eval_model.eval()
	origin_model.eval()
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
		first_time = True
		# if trial_num < 3:
		# 	continue
		peg_type = random.choice(peg_types)
		hole_type = random.choice(peg_types)
		peg_num = trial_num % 3
		hole_num = (trial_num - peg_num) // 3
		peg_type = peg_types[peg_num]
		hole_type = peg_types[hole_num]

		options_memory = torch.from_numpy(np.zeros((3,3))).float().to(device)
		error_mean_memory = torch.from_numpy(np.zeros((3,3))).float().to(device)
		error_cov_memory = torch.from_numpy(np.zeros((3,3,3))).float().to(device)

		env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
		 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
		  gripper_type = peg_type + "PegwForce", controller='position', camera_depth=True)

		obs = env.reset()
		env.viewer.set_camera(camera_id=2)
		converge_count = 0
		glb_cnt = 0
		logging_dict['scalar'] = {}
		error = np.random.normal(0.0, 0.0075, 3)
		error_o = error[:]

		while converge_count < converge_thresh:

			top_goal = peg_dict[hole_type][0]
			bottom_goal = peg_dict[hole_type][1]
			top_plus = top_goal + np.array([0, 0, 0.04, 0,0,0])

			peg_vector = peg_dict[peg_type][2]
			hole_vector = peg_dict[hole_type][2]

			cand_points = slidepoints(workspace_dim, num_candidates)
			pred_logits = eval_model.process(torch.from_numpy(cand_points).to(device).float(),\
			 torch.from_numpy(peg_vector).to(device).float().unsqueeze(0).repeat_interleave(num_candidates, dim = 0))
			bps = cand_points[eval_points(pred_logits)] # best point small
			best_init_point = np.concatenate([bps[:3] + top_goal[:3], np.array([np.pi, 0, np.pi]) ])
			best_final_point = np.concatenate([bps[3:] + top_goal[:3], np.array([np.pi, 0, np.pi])])
			# moving to first initial position
			points_list = [(top_plus, 0, "top_plus"), (best_init_point, 0, "init_point"), (best_final_point, 1, "final_point"), (top_plus, 0, "top_plus")] 
			point_idx = 0

			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]

			step_count = 0

			while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
				action, action_euler = env._pose_err(goal)
				pos_err = kp * action_euler[:3]
				noise = np.random.normal(0.0, 0.1, pos_err.size)
				noise[2] = -1.0 * abs(noise[2])
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
				noise[2] = -1.0 * abs(noise[2])
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
					noise[2] = -1.0 * abs(noise[2])
					pos_err += noise_parameter * noise

					obs['action'] = env.controller.transform_action(pos_err)
					if point_type == 1:
						save_obs(copy.deepcopy(obs), obs_keys, obs_dict)
						# curr_pose = env._get_eepos()

					obs, reward, done, info = env.step(pos_err)
					obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal

					if display_bool:
						env.render()

					step_count += 1

				point_idx += 1

			sample = concatenate(obs_dict,\
			 torch.from_numpy(peg_vector).float().to(device).unsqueeze(0),\
			 torch.from_numpy(hole_vector).float().to(device).unsqueeze(0), torch.from_numpy(error).float().to(device).unsqueeze(0))

			if sample['force_hi_freq'].size(1) < 10:
				continue
			# else:
			# 	print("force size: ", sample['force_hi_freq'].size())

			hole_idx = hole_vector.argmax() # gt index

			options_logits = options_model.process(sample).squeeze(0)

			options_memory[hole_idx] += F.log_softmax(options_logits, dim = 0) # updating memory
			options_probs = F.softmax(options_memory[hole_idx], dim = 0)
			hole_est_idx = options_probs.max(0)[1].item()

			error_mean, error_cov = origin_model.process(sample)

			if first_time:
				error_mean_memory[hole_idx] = error_mean.squeeze()
				error_cov_memory[hole_idx] = error_cov.squeeze()
				first_time = False
				# prev_pose = curr_pose
			else:
				# pose_diff = curr_pose - prev_pose
				# prev_pose = curr_pose
				# origin_mean_memory[hole_idx] += torch.from_numpy(pose_diff).float().to(device)
				error_mean_memory[hole_idx], error_cov_memory[hole_idx] = productofgauss(error_mean.squeeze(), error_mean_memory[hole_idx],\
					error_cov.squeeze(), error_cov_memory[hole_idx])

			# error = error - error_mean_memory[hole_idx].detach().cpu().numpy()
			# error_mean_memory[hole_idx] = torch.zeros_like(error_mean_memory[hole_idx])
			print("Current estimate: ", list(options_probs.detach().cpu().numpy()))
			print("Sensor probs:", list(F.softmax(options_logits, dim = 0).detach().cpu().numpy()))
			print("Network thinks it is a ", peg_types[hole_est_idx], " hole when it is a ", peg_types[hole_idx], " hole")
			print("Sensor mean: ", error_mean.squeeze().detach().cpu().numpy())
			print("Network thinks the current pose is", error_mean_memory[hole_idx].detach().cpu().numpy(), "when it is ", error)
			print("Origin estimate error is", 1000 * np.linalg.norm(error_mean_memory[hole_idx].detach().cpu().numpy() - error), " mm, the original error was", 1000 * np.linalg.norm(error_o), " mm")
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