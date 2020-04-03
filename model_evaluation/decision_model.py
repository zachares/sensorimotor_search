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

from models import *
from logger import Logger
from decision_model import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

def T_angles(angle):
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
	theta_final = T_angles(theta_init + theta_delta * theta_sign)

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

def calc_entropy(belief):
	probs = belief[belief.nonzero()]
	return -1.0 * (probs * np.log(probs)).sum()

def concatenate(tensor_dict, peg_type):
	for key in tensor_dict.keys():
		tensor_dict[key] = torch.cat(tensor_dict[key], dim = 1)

	# tensor_dict['pose'] = tensor_dict['proprio'][:,:,:3]
	# tensor_dict['pose_delta'] = tensor_dict['final_pose'] - tensor_dict['init_pose']
	tensor_dict['peg_type'] = peg_type.unsqueeze(0)

	return tensor_dict

class Multinom(object):
	def __init__(self, size = None, values = None):

		if size is None:
			self.probs = values[:]
		else:
			self.probs = np.ones((size))

		self._normalize()
		self._logprob()
		self.entropy = self._entropy()

	def marginalize(self, axes):
		return self.probs.sum(axis = axes)

	def _normalize(self):
		self.probs = self.probs / sum(self.probs)

	def _entropy(self):
		return calc_entropy(self.probs)

	def _logprob(self):
		self.logprobs = -np.inf * np.ones_like(self.probs)
		for i in range(self.probs.size):
			if self.probs[i] != 0:
				self.logprobs[i] = np.log(self.probs[i])

	def update_logprobs(self, logprobs):
		self.logprobs = logprobs
		self.probs = np.exp(self.logprobs)
		self.entropy = self._entropy()

	def update_probs(self, probs, norm = False):
		self.probs = probs
		if norm:
			self._normalize()
		self._logprob()
		self.entropy = self._entropy()

class DecisionModel(object):
	def __init__(self, hole_poses, num_options, workspace_dim, num_samples, ori_action, plus_offset, pred_model, eval_model, peg_idx = 0):
		self.hole_poses = hole_poses # list
		self.ori_action = ori_action
		self.plus_offset = plus_offset

		self.num_options = num_options

		self.workspace_dim = workspace_dim
		self.num_samples = num_samples

		self.pred_model = pred_model
		self.eval_model = eval_model

		self.set_pegidx(peg_idx)

		self.reset_memory()

	def reset_memory(self):

		self.hole_memory = []
		self.obs_memory = []

		for i in range(self.num_options):
			self.hole_memory.append(Multinom(size = (self.num_options)))
			self.obs_memory.append([])

	def set_pegidx(self, peg_idx):
		self.peg_idx = peg_idx
		self.peg_vector = np.zeros(self.num_options)
		self.peg_vector[self.peg_idx] = 1

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.pred_model.device).float()

	def choose_hole(self):
		max_entropy = 0
		for idx, memory in enumerate(self.hole_memory):
			if max_entropy < memory.entropy:
				max_entropy = memory.entropy
				max_idx = idx

		self.hole_idx = max_idx
		self.hole_type = self.hole_poses[self.hole_idx][1]

		return self.hole_idx

	def choose_action(self):

		logprobs = self.toTorch(self.hole_memory[self.hole_idx].logprobs).unsqueeze(0).repeat_interleave(self.num_samples, 0)

		cand_actions = slidepoints(self.workspace_dim, self.num_samples)

		logits = self.pred_model.process(self.toTorch(cand_actions), \
			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

		exp_entropy = np.zeros(self.num_samples)
		probs = self.hole_memory[self.hole_idx].probs

		for i in range(logits.size(0)):
			logit = logits[i]
			for j in range(logit.size(0)):
				logitt = logit[j].unsqueeze(0)
				prob_matrix = self.update_memory(logitt)
				entropy = calc_entropy(prob_matrix)
				exp_entropy[i] += probs[j] * entropy

		max_idx = exp_entropy.argmin()

		action = cand_actions[max_idx]

		top_goal = self.hole_poses[self.hole_idx][0]

		init_point = np.concatenate([action[:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([action[3:] + top_goal[:3], self.ori_action])
		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

	def choose_hole(self):
		max_entropy = 0
		for idx, memory in enumerate(self.hole_memory):
			if max_entropy < memory.entropy:
				max_entropy = memory.entropy
				max_idx = idx

		self.hole_idx = max_idx
		self.hole_type = self.hole_poses[self.hole_idx][1]

	def choose_both(self):
		cand_actions = slidepoints(self.workspace_dim, self.num_samples)

		logits = self.pred_model.process(self.toTorch(cand_actions), \
			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

		min_exp_entropy = np.inf

		for k in range(len(self.hole_poses)):

			self.hole_idx = k

			logprobs = self.toTorch(self.hole_memory[self.hole_idx].logprobs).unsqueeze(0).repeat_interleave(self.num_samples, 0)
			exp_entropy = np.zeros(self.num_samples)
			probs = self.hole_memory[self.hole_idx].probs

			for i in range(logits.size(0)):
				logit = logits[i]
				for j in range(logit.size(0)):
					logitt = logit[j].unsqueeze(0)
					prob_matrix = self.update_memory(logitt)
					entropy = calc_entropy(prob_matrix)
					exp_entropy[i] += probs[j] * entropy

			if exp_entropy.min() < min_exp_entropy:
				hole_idx = k
				max_idx = exp_entropy.argmin()
				min_exp_entropy = exp_entropy.min()

		self.hole_idx = hole_idx

		action = cand_actions[max_idx]

		top_goal = self.hole_poses[self.hole_idx][0]

		init_point = np.concatenate([action[:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([action[3:] + top_goal[:3], self.ori_action])
		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]


	def update_memory(self, obs_logits):
		perhole_memory = []
		for i, memory in enumerate(self.obs_memory):
			if len(memory) == 0 and i != self.hole_idx:
				perhole_memory.append(torch.log(torch.ones_like(obs_logits) / obs_logits.size(1)))
				continue

			if i != self.hole_idx:
				perhole_memory.append(F.log_softmax(torch.cat(memory, dim=0).sum(0), dim = 0).unsqueeze(0))
			else:
				perhole_memory.append(F.log_softmax(torch.cat(memory + [F.log_softmax(obs_logits, dim = 1)], dim=0).sum(0), dim = 0).unsqueeze(0))

		logprob_matrix = torch.cat(perhole_memory, dim = 0).detach().cpu().numpy()

		prob_matrix = np.zeros_like(logprob_matrix)

		# assuming there is one of each hole type in the set of hole options
		perms = set(permutations(range(logprob_matrix.shape[0])))
		for perm in perms:
			curr_logprob = np.sum(logprob_matrix[np.array(range(logprob_matrix.shape[0])),perm])
			curr_prob = np.exp(curr_logprob)
			for row, col in enumerate(perm):
				prob_matrix[row, col] += curr_prob

		return prob_matrix

	def new_obs(self, observations):

		sample = concatenate(observations, self.toTorch(self.peg_vector))

		if sample['force_hi_freq'].size(1) > 10: # magic number / edge case when num sensor readings is too small
			logits = self.eval_model.process(sample)
			prob_matrix = self.update_memory(logits)

			# updating hole_memory and normalizing
			for i, memory in enumerate(self.hole_memory):
				memory.update_probs(prob_matrix[i], norm = True)

			self.obs_memory[self.hole_idx].append(F.log_softmax(logits, dim = 1))

	def print_hypothesis(self):
		block_length = 5 # magic number
		histogram_height = 13 # magic number
		fill = "#"
		line = "-"
		gap = " "

		for hole_idx, memory in enumerate(self.hole_memory):
			probs = memory.probs
			counts = np.round(probs * histogram_height) - 1

			print("Model Hypthesis:", self.hole_poses[np.argmax(probs)][1] , " , Ground Truth:", self.hole_poses[hole_idx][1])
			print("With the following histograms:")

			for line_idx in range(histogram_height, 0, -1):
				string = "   "

				for i in range(self.num_options):
					count = counts[i]
					if count <= line_idx:
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

		most_prob_config = 1

		for hole_idx, memory in enumerate(self.hole_memory):
			probs = memory.probs
			most_prob_config *= probs.max()

		print("Most probable event is", most_prob_config, " likely")

		return most_prob_config

		# a = input("Should I continue exploring?")

