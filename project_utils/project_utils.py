import numpy as np
import time
import h5py
import sys
import copy
import os
import matplotlib.pyplot as plt
import random
import itertools

import torch
import torch.nn.functional as F

# from multinomial import *
from collections import OrderedDict
import time
import copy

'''
Data Collection and Recording Functions
'''
def plot_image(image):
	imgplot = plt.imshow(np.rot90(image, k=2))
	plt.show(block=False)
	plt.pause(5)
	plt.close('all')
    
def save_obs(obs_dict, keys, array_dict):
	for key in keys:
		if key not in obs_dict.keys() and key != "rgbd":
			raise Exception(key, " measurement missing from dictionary")

		if key == "image":
			# plot_image(obs_dict['image'])
			obs = np.expand_dims(obs_dict[key][::-1,...].astype(np.uint8), axis =0)
		elif key == "depth":
			obs = np.expand_dims(obs_dict[key][::-1,...], axis = 0)
		else:
			obs = np.expand_dims(obs_dict[key][:], axis = 0)

		if key in array_dict.keys():
			array_dict[key].append(copy.deepcopy(obs))
		else:
			array_dict[key] = [copy.deepcopy(obs)]

def obs2Torch(numpy_dict, device): #, hole_type, macro_action):
	tensor_dict = OrderedDict()
	for key, value in numpy_dict.items():
		if type(value) == str:
			continue
			
		# print(value)
		if type(value) == list:
			tensor_dict[key] = torch.from_numpy(np.concatenate(value, axis = 0)).float().unsqueeze(0).to(device)
		else:
			tensor_dict[key] = torch.from_numpy(value).float().to(device)

	return tensor_dict

def print_histogram(probs, labels):
	block_length = 10 # magic number
	histogram_height = 10 # magic number
	fill = "#"
	line = "-"
	gap = " "
	num_labels = len(labels)

	counts = torch.round(probs * histogram_height).squeeze()

	for line_idx in range(histogram_height, 0, -1):
		string = "   "

		for i in range(num_labels):
			count = counts[i]
			if count < line_idx:
				string += (block_length * gap)
			else:
				string += (block_length * fill)

			string += "   "

		print(string)

	string = "   "

	for label in labels:
		remainder = block_length - len(label)

		if remainder % 2 == 0:
			offset = int(remainder / 2)
			string += ( offset * line + label + offset * line)
		else:
			offset = int((remainder - 1) / 2)
			string += ( (offset + 1) * line + label + offset * line)

		string += "   "

	print(string)

	string = "   "

	for i in range(num_labels):
		string += (block_length * line)
		string += "   "

	print(string)
	print("\n")

def toTorch(array, device):
    return torch.from_numpy(array).to(device).float()
'''
Outer Loop Functions
'''
def gen_task_dict(num_actions, substate_names, observation_names, likelihood_model, success_params, constraint_type = 1):
	num_substates = len(substate_names)
	task_dict = {}

	task_dict['tool_names'] = substate_names
	task_dict['num_tools'] = len(task_dict['tool_names'])

	if constraint_type == 0: # all possible states
		task_dict["states"] = tuple(itertools.product(range(num_substates), repeat = num_actions))
	elif constraint_type == 1: # states only without copies of the same substate
		assert num_actions <= num_substates
		task_dict["states"] = list(itertools.permutations(range(num_substates), num_actions))
	elif constraint_type == 2: # only for states comprised of fit, not fit and a task where only one object fits
		if num_substates != 2:
			raise Exception("Wrong number of substate types for constraint 2")

		states = []

		for i in range(num_actions):
			state = []
			for j in range(num_actions):
				state.append(1)

			state[i] = 0

			states.append(tuple(state))

		task_dict["states"] = tuple(states)
	else:
		raise Exception("this constraint type is not currently supported")

	task_dict["substate_names"] = substate_names
	task_dict["num_substates"] = num_substates

	task_dict["num_actions"] = num_actions

	task_dict["num_states"] = len(task_dict["states"])

	### multiple observations each step
	if type(observation_names[0]) == list:
		task_dict['observations'] = tuple(itertools.product(*[ range(len(obs_names)) for obs_names in observation_names]))
	### single observation at each step
	else:
		task_dict['observations'] = [ (i) for i in range(len(observation_names)) ]

	task_dict['obs_names'] = observation_names
	task_dict['num_obs'] = len(task_dict['observations'])
	### to avoid search during online execution
	task_dict['obs2idx'] = {} 
	for i, obs in enumerate(task_dict['observations']):
		task_dict['obs2idx'][obs] = i

	task_dict['alpha_vectors'] = np.zeros((num_substates, num_actions, task_dict['num_states'])) # reward vector

	for act_idx in range(num_actions):
		for tool_idx in range(task_dict['num_tools']):
			for state_idx, state in enumerate(task_dict['states']):
				substate_idx = state[act_idx]
				if substate_idx == tool_idx:
					task_dict['alpha_vectors'][tool_idx, act_idx, state_idx] = 1.0

	task_dict['beta_vectors'] =  np.zeros((num_actions, num_substates, task_dict['num_states'])) # marginalizing vector

	for act_idx in range(num_actions):
		for state_idx, state in enumerate(task_dict['states']):
			substate_idx = state[act_idx]
			task_dict['beta_vectors'][act_idx, substate_idx, state_idx] = 1.0

		task_dict['loglikelihood_matrix'] = np.zeros((task_dict['num_tools'], task_dict['num_obs'],\
		 num_actions, task_dict['num_states']))

	for tool_idx in range(task_dict['num_tools']):
		for obs_idx, obs_idxs in enumerate(task_dict['observations']):
			for act_idx in range(num_actions):
				for state_idx, state in enumerate(task_dict['states']):
					substate_idx = state[act_idx]
					probs = likelihood_model[tool_idx, substate_idx]
					prob = probs[obs_idxs]
					task_dict['loglikelihood_matrix'][tool_idx, obs_idx, act_idx, state_idx] = np.log(prob)

	task_dict['success_rate'] = []

	for tool_idx in range(task_dict['num_tools']):
		task_dict['success_rate'].append(success_params[tool_idx, 1])

	return task_dict
'''
Control Functions
'''

class Spiral2D_Motion_Primitive(object):
	def __init__(self, radius_travelled = 0.02, number_rotations = 3, pressure = 0.004):
		self.rt = radius_travelled # 0.02
		self.nr = number_rotations # 3
		self.pressure = pressure

	def trajectory(self, step, env, cfg): # 2D circular motion primitive
		horizon = cfg['control_params']['movement_horizon']
		kp = cfg['control_params']['kp'][2]

		radius_scale = self.rt / horizon

		reference_point = env.hole_sites[env.cand_idx][-1]
		err = env.get_eef_pos_err(reference_point)

		frequency = 2 * np.pi * self.nr / horizon
		radius = radius_scale * step
		time = frequency * step
		fraction = step / horizon 

		return np.array([radius * np.sin(time), radius * np.cos(time), -self.pressure - kp * max(err[2], 0)]) + reference_point

# def movetogoal(env, cfg, goal, recording_dict = None, recording_keys = None):
# 	######### control parameters
# 	kp = np.array(cfg['control_params']['kp'])
# 	tol = cfg['control_params']['tol']
# 	noise_std = cfg['control_params']['noise_std']
# 	horizon = cfg['control_params']['movement_horizon']
# 	display_bool = cfg['logging_params']['display_bool']
# 	horizon_bool = cfg['control_params']['horizon_bool']
# 	tolerance_bool = cfg['control_params']['tolerance_bool']
# 	step = 0

# 	######### determining initial goal
# 	if callable(goal): # motion primitive
# 		goal_pos = goal(step, env.get_eef_pos_err(env.reference_point), cfg)
# 		tol_bool = False
# 		hor_bool = True
# 	elif type(goal) == list: # list of waypoints
# 		for g in goal:
# 			movetogoal(env, cfg, g, recording_dict = recording_dict, recording_keys = recording_keys)

# 			if cfg['control_params']['done_bool'] and not ignore_done:
# 				return 0

# 		tol_bool = False
# 		hor_bool = False

# 	elif type(goal)==np.ndarray: # single waypoint
# 		goal_pos = goal[:]
# 		tol_bool = tolerance_bool
# 		hor_bool = horizon_bool
# 	else:
# 		raise Exception(type(goal), ' is an unsupported goal type at this time')

# 	if tol_bool and hor_bool:
# 		continue_bool = (tol_bool and not env.check_eef_pos_err(goal_pos + env.reference_point, tol))\
# 		and (hor_bool and step < horizon)
# 	else:
# 		continue_bool = (tol_bool and not env.check_eef_pos_err(goal_pos + env.reference_point, tol))\
# 		or (hor_bool and step < horizon)
		
# 	######## checking whether to continue movement
# 	while continue_bool:
# 	 	############# calculating error based on position error
# 		# print("Goal: ", goal_pos)
# 		# print("Reference Point: ", env.reference_point)
# 		# print("Goal: ", goal_pos + env.reference_point)
# 		# print("Error: ", env.get_eef_pos_err(goal_pos + env.reference_point))
# 		# print("Reference Error: ", env.get_eef_pos_err(env.reference_point))
# 		# print("Proportional Error: ", kp * env.get_eef_pos_err(goal + env.reference_point))
# 		action = kp * env.get_eef_pos_err(goal_pos + env.reference_point)
# 		# print(step)
# 		########## adding noise
# 		if all(noise_std) >= 0:
# 			assert len(noise_std) == 1 or len(noise_std) == action.size
# 			noise = np.random.normal(0.0, noise_std, action.size)
# 			action += noise[:]
# 		elif any(noise_std) < 0:
# 			raise Exception('Noise standard deviation can only be a positive value')

# 		######### recording observations
# 		if recording_keys is not None:
# 			obs, reward, done, info = env.step(action)
# 			obs["action"] = action[:]
# 			save_obs(obs, recording_keys, recording_dict)
# 		else:
# 			obs, reward, done, info = env.step(action, ignore_obs = True)            

# 		######## rendering
# 		if display_bool:
# 			env.render()

# 		if info['done'] == 1.0 and not cfg['control_params']['done_bool'] and not cfg['control_params']['ignore_done']:
# 			# print("Done")
# 			cfg['control_params']['done_bool'] = True
# 			cfg['control_params']['steps'] += step
# 			return 0
			
# 		######## plotting code for debugging
#         # if display_bool:
#         #   plt.scatter(step, obs['force'][2])
#         #   # plt.scatter(step, obs['contact'])
#         #   plt.pause(0.001)

#         ######## recording number of steps and updating goal
# 		step += 1

# 		if callable(goal):
# 			goal_pos = goal(step, env.get_eef_pos_err(env.reference_point), cfg)

# 		if tol_bool and hor_bool:
# 			continue_bool = (tol_bool and not env.check_eef_pos_err(goal_pos + env.reference_point, tol))\
# 			and (hor_bool and step < horizon)
# 		else:
# 			continue_bool = (tol_bool and not env.check_eef_pos_err(goal_pos + env.reference_point, tol))\
# 			or (hor_bool and step < horizon)

# 	cfg['control_params']['steps'] += step
# 	##### closing plot from debugging code
# 	# plt.close('all')


# def move_down(env, display_bool):
# 	obs, reward, done, info = env.step(np.array([0,0, -0.1]))
	
# 	if display_bool:
# 		env.render()

# def slidepoints(workspace_dim, tol, num_trajectories = 10):
# 	zmin = -tol
# 	theta_init = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
# 	theta_final = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
# 	r_init = np.random.uniform(low=0, high =workspace_dim, size = num_trajectories)
# 	r_final = np.random.uniform(low=0, high =workspace_dim, size = num_trajectories)
# 	# theta_sign = np.random.choice([-1, 1], size = num_trajectories)
# 	# theta_final = T_angle(theta_init + theta_delta * theta_sign)

# 	c_point_list = []

# 	for idx in range(theta_init.size):
# 		theta0 = theta_init[idx]
# 		theta1 = theta_final[idx]
# 		x_init = r_init[idx] * np.cos(theta0)
# 		y_init = r_init[idx] * np.sin(theta0)
# 		x_final = r_final[idx] * np.cos(theta1)
# 		y_final = r_final[idx] * np.sin(theta1) 

# 		# print("Initial point: ", x_init, y_init)
# 		# print("Final point: ", x_final, y_final)
# 		c_point_list.append(np.expand_dims(np.array([x_init, y_init, zmin, 0, 0, 0, x_final, y_final, zmin]), axis = 0))

# 	return np.concatenate(c_point_list, axis = 0)
# '''
# Processing Model Inputs
# '''
# def filter_depth(depth_image):
#     depth_image = torch.where( depth_image > 1e-7, depth_image, torch.zeros_like(depth_image))
#     return torch.where( depth_image < 2, depth_image, torch.zeros_like(depth_image))

# '''
# Dealing with angles
# '''
# def T_angle_np(angle):
#     TWO_PI = 2 * np.pi
#     ones = np.ones_like(angle)
#     zeros = np.zeros_like(angle)

#     case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
#     case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
#     case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
#     case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

#     return case1 + case2 + case3 + case4

# def calc_angerr(target, current):
#     TWO_PI = 2 * np.pi
#     ones = np.ones_like(target)
#     zeros = np.zeros_like(target)

#     targ = np.where(target < 0, TWO_PI + target, target)
#     curr = np.where(current < 0, TWO_PI + current, current)

#     curr0 = np.where(abs(targ - (curr + TWO_PI)) < abs(targ - curr), ones , zeros) # curr + TWO_PI
#     curr1 = np.where(abs(targ - (curr - TWO_PI)) < abs(targ - curr), ones, zeros) # curr - TWO_PI
#     curr2 = ones - curr0 - curr1

#     curr_fin = curr0 * (curr + TWO_PI) + curr1 * (curr - TWO_PI) + curr2 * curr

#     error = targ - curr_fin

#     error0 = np.where(abs(error + TWO_PI) < abs(error), ones, zeros)
#     error1 = np.where(abs(error - TWO_PI) < abs(error), ones, zeros)
#     error2 = ones - error0 - error1

#     return error * error2 + (error + TWO_PI) * error0 + (error - TWO_PI) * error1

# def T_angle(angle):
#     TWO_PI = 2 * np.pi
#     ones = torch.ones_like(angle)
#     zeros = torch.zeros_like(angle)

#     case1 = torch.where(angle < -TWO_PI, angle + TWO_PI * ((torch.abs(angle) / TWO_PI).floor() + 1), zeros )
#     case2 = torch.where(angle > TWO_PI, angle - TWO_PI * (angle / TWO_PI).floor(), zeros)
#     case3 = torch.where(angle > -TWO_PI, ones, zeros) * torch.where(angle < 0, TWO_PI + angle, zeros)
#     case4 = torch.where(angle < TWO_PI, ones, zeros) * torch.where(angle > 0, angle, zeros)

#     return case1 + case2 + case3 + case4

# '''
# Old Functions
# '''
# def pairwise_divJS(logits): 
# 	'''
# 	takes a set of logits and calculates the sum of the 
# 	jensen-shannon divergences of all pairwise combinations
# 	of the distributions 

# 	logits of size batch_size x num_distributions x num_options
# 	'''
# 	div_js = torch.zeros(logits.size(0)).to(logits.device).float()

# 	for idx0 in range(logits.size(1) - 1):
# 		for idx1 in range(idx0 + 1, logits.size(1)):
# 			if idx0 == idx1:
# 				continue

# 			inputs0 = logits2inputs(logits[:,idx0])
# 			inputs1 = logits2inputs(logits[:,idx1])

# 			djs_distrs = 0.5 * (inputs2KL(inputs0, inputs1) + inputs2KL(inputs1, inputs0))
# 			div_js[:] += djs_distrs 

# 	return div_js

# def gen_conf_mat(num_options, num_actions, uncertainty_range):
# 	if num_actions == 0:
# 		return np.expand_dims(create_confusion(num_options, uncertainty_range), axis = 0)
# 	else:
# 		conf_mat = []
# 		for i in range(num_actions):
# 			conf_mat.append(np.expand_dims(create_confusion(num_options, uncertainty_range), axis = 0))

# 		return np.concatenate(conf_mat, axis = 0)

# def create_confusion(num_options, u_r):
# 	uncertainty = u_r[0] + (u_r[1] - u_r[0]) * np.random.random_sample()
# 	confusion_matrix = uncertainty * np.random.random_sample((num_options, num_options))

# 	confusion_matrix[range(num_options), range(num_options)] = 1.0

# 	confusion_matrix = confusion_matrix / np.tile(np.expand_dims(np.sum(confusion_matrix, axis = 1), axis = 1), (1, num_options))

# 	return confusion_matri

# def movetogoal(env, cand_idx, cfg, points_list, point_idx, obs, obs_dict = None):
# 	kp = np.array(cfg['control_params']['kp'])
# 	step_threshold = cfg['control_params']['step_threshold']
# 	tol = cfg['control_params']['tol']
# 	display_bool = cfg['logging_params']['display_bool']
# 	dataset_keys = cfg['dataset_keys']

# 	top_goal = env.hole_sites[cand_idx][1]
# 	step_count = 0
# 	goal = points_list[point_idx][0] + top_goal
# 	point_type = points_list[point_idx][1]
# 	done_bool = False
# 	top_height = top_goal[2]
# 	# glb_ts = 0

# 	while env.check_eef_pos_err(goal, tol) == False and step_count < step_threshold:
# 		action = kp * env.get_eef_pos_err(goal)

# 		# noise = np.random.normal(0.0, 0.1, action.size)
# 		# noise[2] = -1.0 * abs(noise[2])
# 		# action += noise_parameters * noise

# 		obs['action'] = env.controller.transform_action(action)

# 		if point_type == 1 and obs_dict is not None:
# 			save_obs(copy.deepcopy(obs), dataset_keys, obs_dict)
# 			# curr_pose = env._get_eepos()

# 		obs, reward, done, info = env.step(action)
# 		obs['proprio'][:3] = obs['proprio'][:3] - top_goal
		
# 		if display_bool:
# 			env.render()

# 		# if display_bool:
# 		# 	plt.scatter(glb_ts, obs['force'][2])
# 		# 	# plt.scatter(glb_ts, obs['contact'])
# 		# 	plt.pause(0.001)
# 		# glb_ts += 1

# 		step_count += 1

# 	if point_type == -1:
# 		print("Step count: ", step_count)

# 	point_idx += 1

# 	if obs_dict is not None:
# 		return point_idx, done_bool, obs, obs_dict
# 	else:
# 		return point_idx, done_bool, obs