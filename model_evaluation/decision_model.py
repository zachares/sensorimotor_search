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

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

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
	return np.where(belief != 0, -1.0 * belief * np.log(belief), 0).sum(-1)

def obs2Torch(numpy_dict, device): #, hole_type, macro_action):
	tensor_dict = {}
	for key in numpy_dict.keys():
		tensor_dict[key] = torch.from_numpy(np.concatenate(numpy_dict[key], axis = 0)).float().unsqueeze(0).to(device)
	return tensor_dict

class DecisionModel(object):
	def __init__(self, hole_poses, num_options, workspace_dim, num_samples, ori_action, plus_offset, sensor, confusion, confusion_mat, sensor_pred, insert_model, peg_idx = 0):
		self.hole_poses = hole_poses # list
		self.ori_action = ori_action
		self.plus_offset = plus_offset

		self.num_options = num_options

		self.workspace_dim = workspace_dim
		self.num_samples = num_samples

		self.sensor = sensor
		self.confusion = confusion
		self.confusion_mat = confusion_mat
		self.sensor_pred = sensor_pred
		self.insert_model = insert_model
		self.random = False
		self.mat_bool = False

		self.set_pegidx(peg_idx)

		self.reset_memory()

	def reset_memory(self):

		self.perms = set(permutations(range(self.num_options)))

		for i, perm in enumerate(self.perms):
			if perm[0] == 0 and perm[1] == 1 and perm[2] == 2:
				self.correct_idx = i

		self.hole_memory = np.ones(len(self.perms)) / len(self.perms)

	def set_pegidx(self, peg_idx):
		self.peg_idx = peg_idx
		self.peg_vector = np.zeros(self.num_options)
		self.peg_vector[self.peg_idx] = 1

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.sensor.device).float()

	def choose_hole(self):
		max_entropy = 0
		for i in range(self.num_options):
			prob = self.marginalize(i)
			entropy = calc_entropy(prob)
			if entropy > max_entropy:
				idx = i
				max_entropy = entropy

		self.hole_idx = idx
		self.hole_type = self.hole_poses[self.hole_idx][1]
		# self.hole_idx = self.peg_idx
		# self.hole_type = self.hole_poses[self.hole_idx][1]

	def choose_action(self):
		cand_actions = slidepoints(self.workspace_dim, self.num_samples)
		prior_logprobs = torch.log(self.toTorch(self.hole_memory))

		if self.random == True:
			max_idx = random.choice(list(range(cand_actions.shape[0])))
		else:
			hole_memory = self.update_memory(self.hole_idx, cand_actions)

			pred_entropy = calc_entropy(hole_memory)

			max_idx = pred_entropy.argmin()

		self.macro_action = np.expand_dims(cand_actions[max_idx], axis = 0)

		top_goal = self.hole_poses[self.hole_idx][0]

		init_point = np.concatenate([self.macro_action[0,:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([self.macro_action[0,3:] + top_goal[:3], self.ori_action])

		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

	def choose_both(self):
		cand_actions = slidepoints(self.workspace_dim, self.num_samples)

		action_logits = self.insert_model.process(self.toTorch(cand_actions), \
			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

		peg_probs = torch.sigmoid(action_logits[:,self.peg_idx])

		max_insert_idx = peg_probs.max(0)[1]
		max_prob = 0

		for i in range(self.num_options):
			prob = self.marginalize(i)[self.peg_idx]
			if prob > max_prob:
				max_prob = prob
				hole_insert_idx = i

		curr_entropy = calc_entropy(self.hole_memory)

		min_pred_entropy = np.inf

		for k in range(len(self.hole_poses)):
			self.hole_idx = copy.deepcopy(k)
			hole_memory = self.update_memory(self.hole_idx, cand_actions)
			pred_entropy = calc_entropy(hole_memory)
			if pred_entropy.min() < min_pred_entropy:
				hole_info_idx = copy.deepcopy(k)
				max_info_idx = pred_entropy.argmin()
				min_pred_entropy = pred_entropy.min()

		print("Current entropy:", curr_entropy)
		print("Min Exp entropy: ", min_pred_entropy)

		print("##################")
		print("# Action Choice #\n")
		if min_pred_entropy < 0.25:
			print("CHOOSING TO INSERT")
			self.hole_idx = hole_insert_idx
			self.insert_bool = 1.0
			max_idx = max_insert_idx
		else:
			print("CHOOSING TO COLLECT INFORMATION")
			print("Current entropy is: " + str(curr_entropy)[:5] + "\nEntropy is predicted to decrease to: " + str(min_pred_entropy)[:5])
			self.hole_idx = hole_info_idx
			self.insert_bool = 0.0
			max_idx = max_info_idx

		print("\n#####################\n")
		self.macro_action = np.expand_dims(cand_actions[max_idx], axis = 0)

		top_goal = self.hole_poses[self.hole_idx][0]

		init_point = np.concatenate([self.macro_action[0,:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([self.macro_action[0,3:] + top_goal[:3], self.ori_action])

		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

	def marginalize(self, hole_idx):
		hole_margin = np.zeros(self.num_options)
		for i, perm in enumerate(self.perms):
			hole_margin[perm[hole_idx]] += self.hole_memory[i]

		return hole_margin

	def update_memory(self, hole_pos_idx, macro_actions_np, obs = None):
		macro_actions = self.toTorch(macro_actions_np)
		prior_logprobs = torch.log(self.toTorch(self.hole_memory)).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)
		hole_memory = torch.zeros_like(prior_logprobs)
		peg_type = self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

		# remove later
		top_goal = self.toTorch(self.hole_poses[hole_pos_idx][0]).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)
		macro_actions[:, 3:] = macro_actions[:, 3:] + top_goal[:,:3]

		for j, perm in enumerate(self.perms):
			hole_idx = perm[hole_pos_idx]
			hole_vector = np.zeros(self.num_options)
			hole_vector[hole_idx] = 1
			hole_type = self.toTorch(hole_vector).unsqueeze(0).repeat_interleave(macro_actions.size(0), dim = 0)

			if obs is not None:
				conf_logprobs = self.sensor.process(obs, peg_type)
			else:
				conf_logprobs = self.sensor_pred.logits(macro_actions, peg_type, hole_type)

			# class_probs = F.softmax(class_logits, dim = 1)

			# hole_est_idxs = class_probs.max(1)[1]

			# if self.mat_bool == True:
			# 	conf_logits = self.confusion_mat.logits(macro_actions, peg_type, hole_type)
			# else:
			# 	conf_logits = self.confusion.logits(macro_actions, peg_type, hole_type)

			# conf_logprobs = F.log_softmax(conf_logits, dim = 1)

			# obs_logprob = conf_logprobs[torch.arange(macro_actions.size(0)), hole_est_idxs]
			hole_memory[:,j] = conf_logprob + prior_logprobs[:,j]

		hole_memory = F.softmax(hole_memory, dim = 1)

		return hole_memory.detach().cpu().numpy()

	def new_obs(self, observations):

		observations['peg_type'] = [self.peg_vector]
		sample = obs2Torch(observations, self.sensor.device)

		if sample['force_hi_freq'].size(1) > 10: # magic number / edge case when num sensor readings is too small
			self.hole_memory = self.update_memory(self.hole_idx, self.macro_action, sample)[0]

	def print_hypothesis(self):
		block_length = 5 # magic number
		histogram_height = 10 # magic number
		fill = "#"
		line = "-"
		gap = " "

		for i in range(self.num_options):
			probs = self.marginalize(i)
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
			prob = self.marginalize(i)[self.peg_idx]
			if most_prob_config < prob:
				most_prob_config = prob
				c_idx = i

		print("Robot estimates that the correct position\n is the", self.hole_poses[c_idx][1], "hole with certainty", str(most_prob_config)[:5], "\n")

		# return most_prob_config

		# a = input("Should I continue exploring?")


# class DecisionModel(object):
# 	def __init__(self, hole_poses, num_options, workspace_dim, num_samples, ori_action, plus_offset, pred_model, eval_model, insert_model, peg_idx = 0):
# 		self.hole_poses = hole_poses # list
# 		self.ori_action = ori_action
# 		self.plus_offset = plus_offset

# 		self.num_options = num_options

# 		self.workspace_dim = workspace_dim
# 		self.num_samples = num_samples

# 		self.pred_model = pred_model
# 		self.eval_model = eval_model
# 		self.insert_model = insert_model

# 		self.set_pegidx(peg_idx)

# 		self.reset_memory()

# 	def reset_memory(self):

# 		self.hole_memory = []
# 		self.obs_memory = []
# 		prior = self.toTorch(np.ones(self.num_options) / self.num_options).unsqueeze(0)

# 		for i in range(self.num_options):
# 			self.hole_memory.append(Multinom(size = (self.num_options)))
# 			self.obs_memory.append([prior.clone()])

# 	def set_pegidx(self, peg_idx):
# 		self.peg_idx = peg_idx
# 		self.peg_vector = np.zeros(self.num_options)
# 		self.peg_vector[self.peg_idx] = 1

# 	def toTorch(self, array):
# 		return torch.from_numpy(array).to(self.pred_model.device).float()

# 	def choose_hole(self):
# 		max_entropy = 0
# 		for idx, memory in enumerate(self.hole_memory):
# 			if max_entropy < memory.entropy:
# 				max_entropy = memory.entropy
# 				max_idx = idx

# 		self.hole_idx = max_idx
# 		self.hole_type = self.hole_poses[self.hole_idx][1]

# 		# self.hole_idx = self.peg_idx
# 		# self.hole_type = self.hole_poses[self.hole_idx][1]

# 	def choose_action(self):

# 		# print("Peg IDX: ", self.peg_idx)
# 		# print("Hole IDX: ", self.hole_idx)

# 		logprobs = self.toTorch(self.hole_memory[self.hole_idx].logprobs).unsqueeze(0).repeat_interleave(self.num_samples, 0)

# 		cand_actions = slidepoints(self.workspace_dim, self.num_samples)

# 		logits = self.pred_model.process(self.toTorch(cand_actions), \
# 			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

# 		exp_entropy = np.zeros(self.num_samples)
# 		probs = self.hole_memory[self.hole_idx].probs

# 		for i in range(logits.size(0)):
# 			logit = logits[i]
# 			for j in range(logit.size(0)):
# 				logitt = logit[j].unsqueeze(0)
# 				prob_matrix = self.update_memory(logitt)
# 				entropy = calc_entropy(prob_matrix)
# 				exp_entropy[i] += probs[j] * entropy

# 		max_idx = exp_entropy.argmin()

# 		action = cand_actions[max_idx]

# 		top_goal = self.hole_poses[self.hole_idx][0]

# 		init_point = np.concatenate([action[:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
# 		final_point = np.concatenate([action[3:] + top_goal[:3], self.ori_action])
# 		top_plus = top_goal + self.plus_offset

# 		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

# 	def choose_both(self):
# 		cand_actions = slidepoints(self.workspace_dim, self.num_samples)

# 		action_logits = self.insert_model.process(self.toTorch(cand_actions), \
# 			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

# 		peg_probs = torch.sigmoid(action_logits[:,self.peg_idx])
# 		max_insert_idx = peg_probs.max(0)[1]

# 		curr_pm = np.zeros((self.num_options, self.num_options))
# 		max_prob = 0

# 		for i, memory in enumerate(self.hole_memory):
# 			curr_pm[i] = memory.probs

# 			if memory.probs[self.peg_idx] > max_prob:
# 				max_prob = memory.probs[self.peg_idx]
# 				hole_insert_idx = i

# 		curr_entropy = calc_entropy(curr_pm)

# 		logits = self.pred_model.process(self.toTorch(cand_actions), \
# 			self.toTorch(self.peg_vector).unsqueeze(0).repeat_interleave(self.num_samples, dim = 0))

# 		min_exp_entropy = np.inf

# 		for k in range(len(self.hole_poses)):

# 			self.hole_idx = k

# 			logprobs = self.toTorch(self.hole_memory[self.hole_idx].logprobs).unsqueeze(0).repeat_interleave(self.num_samples, 0)
# 			exp_entropy = np.zeros(self.num_samples)
# 			probs = self.hole_memory[self.hole_idx].probs

# 			for i in range(logits.size(0)):
# 				logit = logits[i]
# 				for j in range(logit.size(0)):
# 					logitt = logit[j].unsqueeze(0)
# 					prob_matrix = self.update_memory(logitt)
# 					entropy = calc_entropy(prob_matrix)
# 					exp_entropy[i] += probs[j] * entropy

# 			if exp_entropy.min() < min_exp_entropy:
# 				hole_info_idx = k
# 				max_info_idx = exp_entropy.argmin()
# 				min_exp_entropy = exp_entropy.min()

# 		print("##################")
# 		print("# Action Choice #\n")
# 		if curr_entropy - min_exp_entropy < 0.2:
# 			print("CHOOSING TO INSERT")
# 			self.hole_idx = hole_insert_idx
# 			max_idx = max_insert_idx
# 		else:
# 			print("CHOOSING TO COLLECT INFORMATION")
# 			print("Expected Entropy diff / Information Gain: ", str(curr_entropy - min_exp_entropy)[:5])
# 			self.hole_idx = hole_info_idx
# 			max_idx = max_info_idx

# 		print("\n#####################\n")
# 		action = cand_actions[max_idx]

# 		top_goal = self.hole_poses[self.hole_idx][0]

# 		init_point = np.concatenate([action[:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
# 		final_point = np.concatenate([action[3:] + top_goal[:3], self.ori_action])
# 		top_plus = top_goal + self.plus_offset

# 		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]


# 	def update_memory(self, obs_logits):
# 		perhole_memory = []
# 		for i, memory in enumerate(self.obs_memory):
# 			if len(memory) == 0 and i != self.hole_idx:
# 				perhole_memory.append(torch.log(torch.ones_like(obs_logits) / obs_logits.size(1)))
# 				continue

# 			if i != self.hole_idx:
# 				perhole_memory.append(F.log_softmax(torch.cat(memory, dim=0).sum(0), dim = 0).unsqueeze(0))
# 			else:
# 				perhole_memory.append(F.log_softmax(torch.cat(memory + [F.log_softmax(obs_logits, dim = 1)], dim=0).sum(0), dim = 0).unsqueeze(0))

# 		logprob_matrix = torch.cat(perhole_memory, dim = 0).detach().cpu().numpy()

# 		prob_matrix = np.zeros_like(logprob_matrix)

# 		# assuming there is one of each hole type in the set of hole options
# 		perms = set(permutations(range(logprob_matrix.shape[0])))
# 		prob_sum = 0
# 		for perm in perms:
# 			curr_logprob = np.sum(logprob_matrix[np.array(range(logprob_matrix.shape[0])),perm])
# 			curr_prob = np.exp(curr_logprob)
# 			prob_sum += curr_prob
# 			for row, col in enumerate(perm):
# 				prob_matrix[row, col] += curr_prob

# 		prob_matrix = prob_matrix / prob_sum

# 		# for i in range(prob_matrix.shape[0]):
# 		# 	prob_matrix[:,i] = prob_matrix[:,i] / prob_matrix[:,i].sum()

# 		return prob_matrix

# 	def new_obs(self, observations):

# 		sample = concatenate(observations, self.toTorch(self.peg_vector))

# 		if sample['force_hi_freq'].size(1) > 10: # magic number / edge case when num sensor readings is too small
# 			logits = self.eval_model.process(sample)
# 			prob_matrix = self.update_memory(logits)

# 			# updating hole_memory and normalizing
# 			for i, memory in enumerate(self.hole_memory):
# 				memory.update_probs(prob_matrix[i], norm = True)

# 			self.obs_memory[self.hole_idx].append(F.log_softmax(logits, dim = 1))

# 	def print_hypothesis(self):
# 		block_length = 5 # magic number
# 		histogram_height = 10 # magic number
# 		fill = "#"
# 		line = "-"
# 		gap = " "

# 		for hole_idx, memory in enumerate(self.hole_memory):
# 			probs = memory.probs
# 			counts = np.round(probs * histogram_height)

# 			print("Model Hypthesis:", self.hole_poses[np.argmax(probs)][1] , " , Ground Truth:", self.hole_poses[hole_idx][1])
# 			print("With the following histograms:")

# 			for line_idx in range(histogram_height, 0, -1):
# 				string = "   "

# 				for i in range(self.num_options):
# 					count = counts[i]
# 					if count < line_idx:
# 						string += (block_length * gap)
# 					else:
# 						string += (block_length * fill)

# 					string += "   "

# 				print(string)

# 			string = "   "

# 			for hole_pose in self.hole_poses:
# 				hole_type = hole_pose[1]
# 				remainder = block_length - len(hole_type)

# 				if remainder % 2 == 0:
# 					offset = int(remainder / 2)
# 					string += ( offset * line + hole_type + offset * line)
# 				else:
# 					offset = int((remainder - 1) / 2)
# 					string += ( (offset + 1) * line + hole_type + offset * line)

# 				string += "   "

# 			print(string)

# 			string = "   "

# 			for i in range(self.num_options):
# 				string += (block_length * line)
# 				string += "   "

# 			print(string)
# 			print("\n")

# 		most_prob_config = 0

# 		for i, memory in enumerate(self.hole_memory):
# 			prob = memory.probs[self.peg_idx]
# 			if most_prob_config < prob:
# 				most_prob_config = prob
# 				c_idx = i

# 		print("Robot estimates that the correc position\n is the", self.hole_poses[c_idx][1], "hole with certainty", str(most_prob_config)[:5], "\n")

# 		# return most_prob_config

# 		# a = input("Should I continue exploring?")