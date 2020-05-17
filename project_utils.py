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

sys.path.insert(0, "../supervised_learning/") 

from multinomial import *

'''
Data Collection and Recording Functions
'''
def plot_image(image):
	image = np.rot90(image, k =2)
	imgplot = plt.imshow(image)
	plt.show()
    
def save_obs(obs_dict, keys, array_dict):
	for key in keys:
		if key == "rgbd":
			continue
			obs0 = np.rot90(obs_dict['image'], k = 2).astype(np.uint8)
			obs0 = resize(obs0, (128, 128))
			obs0 = np.transpose(obs0, (2, 1, 0))
			obs0 = np.expand_dims(obs0, axis = 0)

			obs1 = np.rot90(obs_dict['depth'], k = 2).astype(np.uint8)
			obs1 = resize(obs1, (128, 128))
			obs1 = np.expand_dims(np.expand_dims(obs1, axis = 0), axis = 0)

			obs = np.concatenate([obs0, obs1], axis = 1)

		else:
			obs = np.expand_dims(obs_dict[key], axis = 0)

		if key in array_dict.keys():
			array_dict[key].append(obs)
		else:
			array_dict[key] = [obs]

def obs2Torch(numpy_dict, device): #, hole_type, macro_action):
	tensor_dict = {}
	for key in numpy_dict.keys():
		tensor_dict[key] = torch.from_numpy(np.concatenate(numpy_dict[key], axis = 0)).float().unsqueeze(0).to(device)
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

'''
Decision Functions
'''
def gen_state_dict(num_cand, options_names, constraint_type):
	num_options = len(options_names)
	state_dict = {}

	if constraint_type == 0:
		state_dict["states"] = tuple(itertools.product(range(num_options), repeat = num_cand))
	elif constraint_type == 1:
		state_dict["states"] = tuple(itertools.permutations(range(num_options), num_cand))
	elif constraint_type == 2:
		if num_options != 2:
			raise Exception("Wrong number of option types for constraint type " + str(2))

		states = []

		for i in range(num_cand):
			state = []
			for j in range(num_cand):
				state.append(1)

			state[i] = 0

			states.append(tuple(state))

		state_dict["states"] = tuple(states)

		print("State space: ", state_dict["states"])

	else:
		raise Exception("this constraint type is not currently supported")

	state_dict["options_names"] = options_names
	state_dict["num_options"] = num_options
	state_dict["num_cand"] = num_cand
	state_dict["num_states"] = len(state_dict["states"])

	return state_dict

def pairwise_divJS(logits): 
    '''
    takes a set of logits and calculates the sum of the 
    jensen-shannon divergences of all pairwise combinations
    of the distributions 

    logits of size batch_size x num_distributions x num_options
    '''
    div_js = torch.zeros(logits.size(0)).to(logits.device).float()

    for idx0 in range(logits.size(1) - 1):
        for idx1 in range(idx0 + 1, logits.size(1)):
            if idx0 == idx1:
                continue

            dis0 = logits[:,idx0]
            dis1 = logits[:,idx1]
            djs_distrs = 0.5 * (logits2KL(dis0, dis1) + logits2KL(dis1, dis0))
            div_js[:] += djs_distrs 
            div_js[:] += djs_distrs

    return div_js

def gen_conf_mat(num_options, num_actions, uncertainty_range):
	if num_actions == 0:
		return np.expand_dims(create_confusion(num_options, uncertainty_range), axis = 0)
	else:
		conf_mat = []
		for i in range(num_actions):
			conf_mat.append(np.expand_dims(create_confusion(num_options, uncertainty_range), axis = 0))

		return np.concatenate(conf_mat, axis = 0)

def create_confusion(num_options, u_r):
	uncertainty = u_r[0] + (u_r[1] - u_r[0]) * np.random.random_sample()
	confusion_matrix = uncertainty * np.random.random_sample((num_options, num_options))

	confusion_matrix[range(num_options), range(num_options)] = 1.0

	confusion_matrix = confusion_matrix / np.tile(np.expand_dims(np.sum(confusion_matrix, axis = 1), axis = 1), (1, num_options))

	return confusion_matrix
'''
Control Functions
'''
def slidepoints(workspace_dim, tol, num_trajectories = 10):
	zmin = -tol
	theta_init = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
	theta_final = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
	r_init = np.random.uniform(low=0, high =workspace_dim, size = num_trajectories)
	r_final = np.random.uniform(low=0, high =workspace_dim, size = num_trajectories)
	# theta_sign = np.random.choice([-1, 1], size = num_trajectories)
	# theta_final = T_angle(theta_init + theta_delta * theta_sign)

	c_point_list = []

	for idx in range(theta_init.size):
		theta0 = theta_init[idx]
		theta1 = theta_final[idx]
		x_init = r_init[idx] * np.cos(theta0)
		y_init = r_init[idx] * np.sin(theta0)
		x_final = r_final[idx] * np.cos(theta1)
		y_final = r_final[idx] * np.sin(theta1) 

		# print("Initial point: ", x_init, y_init)
		# print("Final point: ", x_final, y_final)
		c_point_list.append(np.expand_dims(np.array([x_init, y_init, zmin, 0, 0, 0, x_final, y_final, zmin]), axis = 0))

	return np.concatenate(c_point_list, axis = 0)

def movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs, obs_dict = None):
	glb_ts, kp, tol, step_threshold, display_bool, top_height, obs_keys = fixed_params #noise_parameters,

	step_count = 0
	goal = points_list[point_idx][0]
	point_type = points_list[point_idx][1]
	done_bool = False

	while env.check_eepos_err(goal, tol) == False and step_count < step_threshold:
		action = kp * env.eepos_err(goal)

		# noise = np.random.normal(0.0, 0.1, action.size)
		# noise[2] = -1.0 * abs(noise[2])
		# action += noise_parameters * noise

		obs['action'] = env.controller.transform_action(action)

		if point_type == 1 and obs_dict is not None:
			save_obs(copy.deepcopy(obs), obs_keys, obs_dict)
			# curr_pose = env._get_eepos()

		obs, reward, done, info = env.step(action)
		obs['proprio'][:3] = obs['proprio'][:3] - top_goal
		
		if display_bool:
			env.render()

		# print(obs['proprio'][2])

		if obs['proprio'][2] < - 0.0001 and point_type == 1:
			# print("Inserted")
			obs['insertion'] = np.array([1.0])
			done_bool = True
			# step_count = step_threshold
		else:
			obs['insertion'] = np.array([0.0])

		# if display_bool:
		# 	# plt.scatter(glb_ts, obs['force'][2])
		# 	plt.scatter(glb_ts, obs['contact'])
		# 	plt.pause(0.001)
		# glb_ts += 1

		step_count += 1

	if point_type == -1:
		print("Step count: ", step_count)

	point_idx += 1

	if obs_dict is not None:
		return point_idx, done_bool, obs, obs_dict
	else:
		return point_idx, done_bool, obs

def move_down(env, display_bool):
	obs, reward, done, info = env.step(np.array([0,0, -0.1]))
	
	if display_bool:
		env.render()
'''
Processing Model Inputs
'''
def filter_depth(depth_image):
    depth_image = torch.where( depth_image > 1e-7, depth_image, torch.zeros_like(depth_image))
    return torch.where( depth_image < 2, depth_image, torch.zeros_like(depth_image))

'''
Dealing with angles
'''
def T_angle_np(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4

def calc_angerr(target, current):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(target)
    zeros = np.zeros_like(target)

    targ = np.where(target < 0, TWO_PI + target, target)
    curr = np.where(current < 0, TWO_PI + current, current)

    curr0 = np.where(abs(targ - (curr + TWO_PI)) < abs(targ - curr), ones , zeros) # curr + TWO_PI
    curr1 = np.where(abs(targ - (curr - TWO_PI)) < abs(targ - curr), ones, zeros) # curr - TWO_PI
    curr2 = ones - curr0 - curr1

    curr_fin = curr0 * (curr + TWO_PI) + curr1 * (curr - TWO_PI) + curr2 * curr

    error = targ - curr_fin

    error0 = np.where(abs(error + TWO_PI) < abs(error), ones, zeros)
    error1 = np.where(abs(error - TWO_PI) < abs(error), ones, zeros)
    error2 = ones - error0 - error1

    return error * error2 + (error + TWO_PI) * error0 + (error - TWO_PI) * error1

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)

    case1 = torch.where(angle < -TWO_PI, angle + TWO_PI * ((torch.abs(angle) / TWO_PI).floor() + 1), zeros )
    case2 = torch.where(angle > TWO_PI, angle - TWO_PI * (angle / TWO_PI).floor(), zeros)
    case3 = torch.where(angle > -TWO_PI, ones, zeros) * torch.where(angle < 0, TWO_PI + angle, zeros)
    case4 = torch.where(angle < TWO_PI, ones, zeros) * torch.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4