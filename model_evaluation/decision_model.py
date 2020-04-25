from itertools import permutations
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

# import robosuite
# import robosuite.utils.transform_utils as T
# from robosuite.wrappers import IKWrapper

# def multinomial_KL(logits_q, logits_p, dim):
# 	return -(F.softmax(logits_p, dim =dim) * (F.log_softmax(logits_q, dim = dim) - F.log_softmax(logits_p, dim = dim))).sum(dim)

# def max_avgDivJS(logits, dim):
# 	# logits of size batch_size x num_options x num_options
# 	div_js = torch.zeros(logits.size()).to(logits.device).float().sum(2)

# 	for idx0 in range(logits.size(1) - 1):
# 		for idx1 in range(idx0 + 1, logits.size(1)):
# 			if idx0 == idx1:
# 				continue

# 			dis0 = logits[:,idx0]
# 			dis1 = logits[:,idx1]
# 			djs_distrs = 0.5 * (multinomial_KL(dis0, dis1, dim =dim) + multinomial_KL(dis1, dis0, dim=dim))
# 			div_js[:, idx0] += djs_distrs 
# 			div_js[:, idx1] += djs_distrs

# 	return div_js.max(0)[1][0] # 0 at the end picks the first value in the tensor which is a maximum

# def exp_logprobs(logits, logprobs):

# 	exp_logprobs = torch.zeros_like(logits[:,0])

# 	for i in range(logits.size(1)):
# 		logs = logits[:,i]
# 		lprobs = F.log_softmax((F.log_softmax(logs, dim =1) + logprobs), dim = 1)
# 		exp_logprobs += torch.exp(logprobs)[:,i] * lprobs 

# 	return exp_logprobs

def mat_conf_logprobs(peg_type, hole_type, probs):
	conf_logprobs = torch.zeros_like(peg_type[:,0])
	certainty = 180
	uncertainty = 20

	conf_matrix = np.array([\
		[[certainty, uncertainty, uncertainty],\
		[uncertainty, certainty, uncertainty],\
		[uncertainty, uncertainty, certainty]],\
		[[certainty, uncertainty, uncertainty],\
		[uncertainty, certainty, uncertainty],\
		[uncertainty, uncertainty, certainty]],\
		[[certainty, uncertainty, uncertainty],\
		[uncertainty, certainty, uncertainty],\
		[uncertainty, uncertainty, certainty]],\
		])

	conf_matrix = conf_matrix / np.tile(np.expand_dims(np.sum(conf_matrix, axis = 2), axis = 2), (1,1,peg_type.shape[1]))

	conf_matrix = torch.log(torch.from_numpy(conf_matrix).float().to(peg_type.device))

	for i in range(peg_type.size(0)):
		p_idx = peg_type[i].max(0)[1]
		h_idx = hole_type[i].max(0)[1]
		c_idx = probs[i].max(0)[1]

		conf_logprobs[i] = conf_matrix[p_idx, h_idx, c_idx]

	return conf_logprobs


def calc_entropy(belief):
	return np.where(belief != 0, -1.0 * belief * np.log(belief), 0).sum(-1)

def obs2Torch(numpy_dict, device): #, hole_type, macro_action):
	tensor_dict = {}
	for key in numpy_dict.keys():
		tensor_dict[key] = torch.from_numpy(np.concatenate(numpy_dict[key], axis = 0)).float().unsqueeze(0).to(device)
	return tensor_dict

class DecisionModel(object):
	def __init__(self, hole_poses, num_options, workspace_dim, num_samples, ori_action, plus_offset, model_ensemble, peg_idx = 0):
		sensor, sensor_pred, insert_model, distance_model = model_ensemble

		self.hole_poses = hole_poses # list
		self.ori_action = ori_action
		self.plus_offset = plus_offset

		self.num_options = num_options

		self.workspace_dim = workspace_dim
		self.num_samples = num_samples

		self.sensor = sensor
		self.sensor_pred = sensor_pred
		self.insert_model = insert_model
		self.distance_model = distance_model
		self.random = False
		self.mat = False

		self.set_pegidx(peg_idx)

		self.reset_memory()

	def reset_memory(self):

		self.perms = set(permutations(range(self.num_options)))

		for i, perm in enumerate(self.perms):
			if perm[0] == 0 and perm[1] == 1 and perm[2] == 2:
				self.correct_idx = i

		self.hole_memory = np.expand_dims(np.ones(len(self.perms)) / len(self.perms), axis = 0)

	def set_pegidx(self, peg_idx):
		self.peg_idx = peg_idx
		self.peg_vector = np.zeros(self.num_options)
		self.peg_vector[self.peg_idx] = 1

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.sensor.device).float()

	def transform_action(self, macro_action):
		init_point = np.concatenate([macro_action[0,:3] + self.top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([macro_action[0,6:] + self.top_goal[:3], self.ori_action])

		top_plus = self.top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

	def sample_actions(self):
		return slidepoints(self.workspace_dim, self.num_samples)

	def choose_hole(self):
		max_entropy = 0
		for i in range(self.num_options):
			prob = self.marginalize(self.hole_memory, hole_idx = i)[0]
			entropy = calc_entropy(prob)
			if entropy > max_entropy:
				idx = i
				max_entropy = entropy

		self.hole_idx = idx
		self.hole_type = self.hole_poses[self.hole_idx][1]
		self.top_goal = self.hole_poses[self.hole_idx][0]

	def calc_exp_pred_entropy(self, hole_i, macro_actions):
		exp_pred_entropy = np.zeros(macro_actions.shape[0])

		for i, perm in enumerate(self.perms):
			hole_idx = perm[hole_i]
			probs = self.pred_sensor_probs(hole_idx, macro_actions)
			hole_memory = self.expected_memory(hole_i, macro_actions, probs)
			peg_memory = self.marginalize(hole_memory, shape_idx = self.peg_idx)
			exp_pred_entropy += self.hole_memory[0,i] * calc_entropy(peg_memory)

		return exp_pred_entropy

	def choose_action(self):
		macro_actions = self.sample_actions()

		if self.random:
			max_idx = np.random.choice(range(self.num_samples))
		else:
			distances = self.calc_distances(macro_actions)
			distances = distances - distances.min()

			curr_peg_memory = self.marginalize(self.hole_memory, shape_idx = self.peg_idx)
			curr_entropy = calc_entropy(curr_peg_memory)[0]

			exp_pred_entropy = self.calc_exp_pred_entropy(self.hole_idx, macro_actions)

			print("CHOOSING TO COLLECT INFORMATION")
			print("Current entropy is: " + str(curr_entropy)[:5] + "\nEntropy is predicted to decrease to: " + str(exp_pred_entropy.min())[:5]) 

			max_idx = exp_pred_entropy.argmin()

			print("\n#####################\n")

		self.macro_action = np.expand_dims(macro_actions[max_idx], axis = 0)

		return self.transform_action(self.macro_action)

	def calc_distances(self, macro_actions_np):
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0)
		macro_actions = self.toTorch(macro_actions_np)

		return self.distance_model.distances(peg_type, macro_actions).min(1)[0].detach().cpu().numpy()

	def calc_insert_probs(self, macro_actions_np):
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0)
		macro_actions = self.toTorch(macro_actions_np)
		return torch.sigmoid(self.insert_model.process(macro_actions, peg_type, peg_type))

	# def choose_both(self):
	# 	macro_actions = self.sample_actions()
	# 	distances = self.calc_distances(macro_actions)
	# 	distances = distances - distances.min()

	# 	#### Insertion calculation #####
	# 	insert_probs = self.calc_insert_probs(macro_actions).squeeze()

	# 	curr_peg_memory = self.marginalize(self.hole_memory, shape_idx = self.peg_idx)[0]

	# 	insert_prob = insert_probs.max(0)[0] * curr_peg_memory.max()

	# 	max_insert_idx = insert_probs.max(0)[1]
	# 	hole_insert_idx = curr_peg_memory.argmax()

	# 	curr_entropy = calc_entropy(curr_peg_memory) - distances.min()

	# 	min_pred_entropy = np.inf

	# 	for k in range(len(self.hole_poses)):
	# 		exp_pred_entropy = self.calc_exp_pred_entropy(k, macro_actions)

	# 		if exp_pred_entropy.min() < min_pred_entropy:
	# 			hole_info_idx = copy.deepcopy(k)
	# 			max_info_idx = exp_pred_entropy.argmin()
	# 			min_pred_entropy = exp_pred_entropy.min()

	# 	print("##################")
	# 	print("# Action Choice #\n")
	# 	if self.marginalize(self.hole_memory, shape_idx = self.peg_idx)[0].max() > 0.9:
	# 		print("CHOOSING TO INSERT")
	# 		self.hole_idx = hole_insert_idx
	# 		self.insert_bool = 1.0
	# 		max_idx = max_insert_idx
	# 	else:
	# 		print("CHOOSING TO COLLECT INFORMATION")
	# 		print("Current entropy is: " + str(curr_entropy)[:5] + "\nEntropy is predicted to decrease to: " + str(min_pred_entropy)[:5])
	# 		self.hole_idx = hole_info_idx
	# 		self.insert_bool = 0.0
	# 		max_idx = max_info_idx

	# 	self.hole_type = self.hole_poses[self.hole_idx][1]
	# 	self.top_goal = self.hole_poses[self.hole_idx][0]

	# 	print("\n#####################\n")
	# 	self.macro_action = np.expand_dims(macro_actions[max_idx], axis = 0)

	# 	return self.transform_action(self.macro_action)

	def marginalize(self, hole_memory, hole_idx = None, shape_idx = None):
		margin = np.zeros((hole_memory.shape[0], self.num_options))
		if hole_idx is not None:
			for i, perm in enumerate(self.perms):
				margin[:, perm[hole_idx]] += hole_memory[:,i]
		else:
			for i, perm in enumerate(self.perms):
				idx = list(perm).index(shape_idx)
				margin[:,idx] += hole_memory[:,i]

		return margin

	def update_memory(self, hole_pos_idx, macro_actions_np, obs):
		macro_actions = self.toTorch(macro_actions_np)
		prior_logprobs = torch.log(self.toTorch(self.hole_memory)).repeat_interleave(macro_actions.size(0), dim = 0)
		hole_memory = torch.zeros_like(prior_logprobs)
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

		for j, perm in enumerate(self.perms):
			hole_idx = perm[hole_pos_idx]

			hole_vector = np.zeros(self.num_options)

			hole_vector[hole_idx] = 1

			hole_type = self.toTorch(hole_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

			probs = self.sensor.probs(obs, peg_type, macro_actions)

			if self.mat:
				conf_logprobs = mat_conf_logprobs(peg_type, hole_type, probs)
			else:
				conf_logprobs = self.sensor_pred.conf_logprobs(peg_type, hole_type, macro_actions, probs = probs)

			hole_memory[:,j] = conf_logprobs + prior_logprobs[:,j]

		hole_memory = F.softmax(hole_memory, dim = 1)

		return hole_memory.detach().cpu().numpy()

	def pred_sensor_probs(self, hole_type_idx, macro_actions_np):
		macro_actions = self.toTorch(macro_actions_np)
		hole_vector = np.zeros(self.num_options)
		hole_vector[hole_type_idx] = 1
		hole_type = self.toTorch(hole_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

		return self.sensor_pred.sensor_probs(peg_type, hole_type, macro_actions)

	def expected_memory(self, hole_pos_idx, macro_actions_np, probs):
		macro_actions = self.toTorch(macro_actions_np)
		prior_logprobs = torch.log(self.toTorch(self.hole_memory)).repeat_interleave(macro_actions.size(0), dim = 0)
		hole_memory = torch.zeros_like(prior_logprobs)
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

		for j, perm in enumerate(self.perms):
			hole_idx = perm[hole_pos_idx]
			hole_vector = np.zeros(self.num_options)
			hole_vector[hole_idx] = 1
			hole_type = self.toTorch(hole_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

			if self.mat:
				conf_logprobs = mat_conf_logprobs(peg_type, hole_type, probs)
			else:
				conf_logprobs = self.sensor_pred.conf_logprobs(peg_type, hole_type, macro_actions, probs = probs)

			hole_memory[:,j] = conf_logprobs + prior_logprobs[:,j]

		hole_memory = F.softmax(hole_memory, dim = 1)

		return hole_memory.detach().cpu().numpy()

	def new_obs(self, observations):

		observations['peg_type'] = [self.peg_vector]
		sample = obs2Torch(observations, self.sensor.device)

		if sample['force_hi_freq'].size(1) > 10: # magic number / edge case when num sensor readings is too small
			self.macro_action = np.expand_dims(np.concatenate([np.array(observations['proprio'])[0,0,:6], self.macro_action[0,6:]]), axis = 0)
			self.hole_memory = self.update_memory(self.hole_idx, self.macro_action, sample)

	def print_hypothesis(self):
		block_length = 8 # magic number
		histogram_height = 10 # magic number
		fill = "#"
		line = "-"
		gap = " "

		for i in range(self.num_options):
			probs = self.marginalize(self.hole_memory, hole_idx = i)[0]
			counts = np.round(probs * histogram_height)

			print("Model Hypthesis:", self.hole_poses[np.argmax(probs)][1] , " , Ground Truth:", self.hole_poses[i][1])
			print("With the following histograms:")

			for line_idx in range(histogram_height, 0, -1):
				string = "   "

				for i in range(self.num_options):
					count = counts[i]
					if count < line_idx:
						string += (block_length * gap)
					else:
						string += (block_length * fill)

					string += "   "

				print(string)

			string = "   "

			for hole_pose in self.hole_poses:
				hole_type = hole_pose[1]
				remainder = block_length - len(hole_type)

				if remainder % 2 == 0:
					offset = int(remainder / 2)
					string += ( offset * line + hole_type + offset * line)
				else:
					offset = int((remainder - 1) / 2)
					string += ( (offset + 1) * line + hole_type + offset * line)

				string += "   "

			print(string)

			string = "   "

			for i in range(self.num_options):
				string += (block_length * line)
				string += "   "

			print(string)
			print("\n")

		most_prob_config = 0

		for i in range(self.num_options):
			prob = self.marginalize(self.hole_memory, shape_idx = self.peg_idx)[0]

		print("Robot estimates that the correct position\n is the", self.hole_poses[prob.argmax()][1], "hole with certainty", str(prob.max())[:5], "\n")

