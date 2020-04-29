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

def multinomial_KL(logits_q, logits_p):
	return -(F.softmax(logits_p, dim =-1) * (F.log_softmax(logits_q, dim =-1) - F.log_softmax(logits_p, dim =-1))).sum(-1)

def DJS(logits):
	# logits of size batch_size x num_options x num_options
	div_js = torch.zeros(logits.size(0)).to(logits.device).float()

	for idx0 in range(logits.size(1) - 1):
		for idx1 in range(idx0 + 1, logits.size(1)):
			if idx0 == idx1:
				continue

			dis0 = logits[:,idx0]
			dis1 = logits[:,idx1]
			djs_distrs = 0.5 * (multinomial_KL(dis0, dis1) + multinomial_KL(dis1, dis0))
			div_js[:] += djs_distrs 
			div_js[:] += djs_distrs

	return div_js

def calc_entropy(belief):
	return torch.where(belief != 0, -1.0 * belief * torch.log(belief), torch.zeros_like(belief)).sum(-1)

def obs2Torch(numpy_dict, device): #, hole_type, macro_action):
	tensor_dict = {}
	for key in numpy_dict.keys():
		tensor_dict[key] = torch.from_numpy(np.concatenate(numpy_dict[key], axis = 0)).float().unsqueeze(0).to(device)
	return tensor_dict

def gen_state_dict(pos_info, option_names, constraint = True):
	state_dict = {}

	if constraint:
		state_dict["states"] = tuple(itertools.permutations(range(len(option_names)), len(pos_info.keys())))
	else:
		state_dict["states"] = tuple(itertools.product(range(len(option_names)), repeat = len(pos_info.keys())))

	for k, state in enumerate(state_dict["states"]):
		correct = True
		for i, j in enumerate(state):
			if pos_info[i]["name"] != option_names[j]:
				correct = False

		if correct:
			state_dict["correct_state"] = state
			state_dict["correct_idx"] = k

	state_dict["option_names"] = option_names
	state_dict["num_options"] = len(option_names)
	state_dict["num_pos"] = len(pos_info.keys())
	state_dict["num_states"] = len(state_dict["states"])

	return state_dict

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

class Action_PegInsertion(object):
	def __init__(self, hole_info, workspace_dim, ori_action, plus_offset, num_actions):
		self.ori_action = ori_action
		self.plus_offset = plus_offset
		self.workspace_dim = workspace_dim
		self.hole_info = hole_info
		self.num_actions = num_actions

	def generate_actions(self):
		self.actions = slidepoints(self.workspace_dim, self.num_actions)

	def get_action(self, act_idx):
		return self.actions[act_idx]

	def transform_action(self, pos_idx, act_idx):
		action = self.actions[act_idx]
		top_goal = self.hole_info[pos_idx]["pos"]
		init_point = np.concatenate([action[:3] + top_goal[:3], self.ori_action]) # ori action to compensate for only doing position control
		final_point = np.concatenate([action[6:] + top_goal[:3], self.ori_action])

		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

class Probability_PegInsertion(object):
	def __init__(self, sensor, confusion_model, num_options):
		self.sensor = sensor
		self.conf = confusion_model
		self.num_options = num_options
		self.device = self.sensor.device

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.device).float()

	def expand(self, idx, size):
		vector = np.zeros(self.num_options)
		vector[idx] = 1
		return self.toTorch(vector).unsqueeze(0).repeat_interleave(size, dim = 0)

	def set_tool_idx(self, tool_idx):
		self.tool_idx = tool_idx

	def record_test(self, obs):
		if len(obs['force_hi_freq']) < 10: # magic number / edge case when num sensor readings is too small
			return False
		else:
			return True

	def transform_action(self, obs, actions_np):
		proprio = obs["proprio"][0]
		return np.expand_dims(np.concatenate([proprio[0,:6], actions_np[6:]]), axis = 0)

	def sensor_obs(self, obs, actions_np):
		action = self.toTorch(actions_np)
		sample = obs2Torch(obs, self.device)
		sample['peg_type'] = self.expand(self.tool_idx, 1)
		
		return self.sensor.probs(sample, action).max(1)[1]

	def conf_logits(self, actions_np):
		actions = self.toTorch(actions_np)
		batch_size = actions.size(0)
		peg_type = self.expand(self.tool_idx, batch_size)
		
		return self.conf.logits(peg_type, actions)

	def conf_logprob(self, options_idx, actions_np, obs_idx):
		actions = self.toTorch(actions_np)
		batch_size = actions.size(0)
		tool_type = self.expand(self.tool_idx, batch_size)
		options_type = self.expand(options_idx, batch_size)

		return self.conf.conf_logprobs(tool_type, options_type, actions, obs_idx)

class Action_Ideal(object):
	def __init__(self, num_actions):
		self.num_actions = num_actions

	def generate_actions(self):
		if self.num_actions == 0:
			return np.array([0])
		else:
			self.actions = np.array(range(self.num_actions))

	def get_action(self, act_idx):
		return self.actions[act_idx]

	def transform_action(self, pos_idx, act_idx):
		return (pos_idx, act_idx)

class Probability_Ideal(object):
	def __init__(self, num_options, num_actions, uncertainty_range, device):
		self.num_options = num_options
		self.num_actions = num_actions
		self.u_r = uncertainty_range
		self.device = device

		self.gen_cm()

	def gen_cm(self):
		self.conf = gen_conf_mat(self.num_options, self.num_actions, self.u_r)

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.device).float()

	def record_test(self, obs):
		return True

	def transform_action(self, options_idx, act_idx):
		return np.array([act_idx])

	def sensor_obs(self, options_idx, act_idx):
		probs = self.conf[act_idx[0], options_idx]
		return np.random.multinomial(1, probs, size = 1).argmax()

	def conf_logits(self, actions):
		return torch.log(self.toTorch(self.conf))

	def conf_logprob(self, options_idx, act_idx, obs_idx):
		return torch.log(self.toTorch(self.conf[act_idx, [options_idx]*len(act_idx), [obs_idx]*len(act_idx)]))

class Decision_Model(object):
	def __init__(self, state_dict, prob_model, act_model):
		self.state_dict = state_dict
		self.prob_model = prob_model
		self.act_model = act_model
		self.device = self.prob_model.device

		self.reset_dis()

	def reset_dis(self):
		self.state_dis = torch.ones((1,self.state_dict["num_states"])) / self.state_dict["num_states"]
		self.state_dis = self.state_dis.to(self.device).float()
		self.max_entropy = calc_entropy(self.state_dis).item()
		self.curr_entropy = calc_entropy(self.state_dis).item()

		self.step_count = 0

	def max_ent_pos(self):
		max_entropy = 0
		for i in range(self.state_dict["num_pos"]):
			prob = self.marginalize(self.state_dis, pos_idx = i).squeeze()
			entropy = calc_entropy(prob)
			if entropy > max_entropy:
				idx = i
				max_entropy = entropy

		return idx

	def max_ent_action(self, actions):
		logits = self.prob_model.conf_logits(actions)
		div_js = DJS(logits)

		return div_js.max(0)[1]

	def choose_action(self, pol_num):
		self.act_model.generate_actions()
		print("Current entropy is: " + str(self.curr_entropy)[:5])

		if pol_num == 0:
			pos_idx = np.random.choice(range(self.state_dict["num_pos"]))
			act_idx = np.random.choice(range(self.act_model.num_actions))

		elif pol_num == 1:
			pos_idx = self.max_ent_pos()
			act_idx = self.max_ent_action(self.act_model.actions)

		elif pol_num == 2:
			min_entropy = copy.deepcopy(self.max_entropy)

			for k in range(self.state_dict["num_pos"]):
				exp_entropy = self.expected_entropy(k, self.act_model.actions)

				if exp_entropy.min(0)[0] < min_entropy:
					pos_idx = copy.deepcopy(k)
					act_idx = exp_entropy.min(0)[1]
					min_entropy = exp_entropy.min(0)[0]
			
			print("Entropy is expected to decrease to: " + str(min_entropy)[:5])

		self.pos_idx = pos_idx
		self.act_idx = act_idx

		print("\n#####################\n")

		return self.act_model.transform_action(self.pos_idx, self.act_idx)

	def marginalize(self, state_dis, pos_idx = None, options_idx = None):
		if pos_idx is not None:
			margin = torch.zeros((self.state_dict["num_options"], state_dis.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				margin[state[pos_idx]] += state_dis.transpose(0,1)[i]
		else:
			margin = torch.zeros((self.state_dict["num_pos"], state_dis.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				if options_idx in state:
					idx = state.index(options_idx)
					margin[idx] += state_dis.transpose(0,1)[i]

		return margin.transpose(0,1)

	def expected_entropy(self, pos_idx, actions):
		batch_size = actions.shape[0]
		prior_logprobs = torch.log(self.state_dis.repeat_interleave(batch_size, dim = 0))
		expected_entropy = torch.zeros(batch_size).float().to(self.device)

		for obs_idx in range(self.state_dict["num_options"]):
			p_o_given_a = torch.zeros(batch_size).float().to(self.device)

			for j, state in enumerate(self.state_dict["states"]):
				options_idx = state[pos_idx]

				conf_logprobs = self.prob_model.conf_logprob(options_idx, actions, obs_idx)

				# print(conf_logprobs.size())

				p_o_given_a[:] += torch.exp(prior_logprobs[:,j] + conf_logprobs)

			expected_entropy[:] += p_o_given_a[:] * calc_entropy(self.update_dis(pos_idx, actions, obs_idx))

		return expected_entropy

	def update_dis(self, pos_idx, actions, obs_idx):
		batch_size = actions.shape[0]
		prior_logprobs = torch.log(self.state_dis.repeat_interleave(batch_size, dim = 0))
		state_dis = torch.zeros_like(prior_logprobs)

		for j, state in enumerate(self.state_dict["states"]):
			options_idx = state[pos_idx]

			conf_logprobs =self.prob_model.conf_logprob(options_idx, actions, obs_idx)

			# print(conf_logprobs.size())

			state_dis[:,j] = conf_logprobs + prior_logprobs[:,j]

			# print(conf_logprobs)
			# print(state_dis)

		state_dis = F.softmax(state_dis, dim = 1)

		return state_dis

	def new_obs(self, obs):
		if self.prob_model.record_test(obs):
			action = self.act_model.get_action(self.act_idx)
			# print(action)
			action = self.prob_model.transform_action(obs, action)
			# print(action)
			obs_idx = self.prob_model.sensor_obs(obs, action)
			# print(obs_idx)
			# print(self.state_dis)

			self.state_dis = self.update_dis(self.pos_idx, action, obs_idx)
			# print(self.state_dis)

			self.curr_entropy = calc_entropy(self.state_dis).item()
			self.step_count += 1

	def print_hypothesis(self):
		block_length = 10 # magic number
		histogram_height = 10 # magic number
		fill = "#"
		line = "-"
		gap = " "

		for i in range(self.state_dict["num_pos"]):
			probs = self.marginalize(self.state_dis, pos_idx = i)

			counts = torch.round(probs * histogram_height).squeeze()

			print("Model Hypthesis:", self.state_dict["option_names"][probs.max(1)[1]] ,\
			 " , Ground Truth:", self.state_dict["option_names"][self.state_dict["correct_state"][i]])
			
			print("Probabilities", probs.detach().cpu().numpy())
			for line_idx in range(histogram_height, 0, -1):
				string = "   "

				for i in range(self.state_dict["num_options"]):
					count = counts[i]
					if count < line_idx:
						string += (block_length * gap)
					else:
						string += (block_length * fill)

					string += "   "

				print(string)

			string = "   "

			for option_name in self.state_dict["option_names"]:
				remainder = block_length - len(option_name)

				if remainder % 2 == 0:
					offset = int(remainder / 2)
					string += ( offset * line + option_name + offset * line)
				else:
					offset = int((remainder - 1) / 2)
					string += ( (offset + 1) * line + option_name + offset * line)

				string += "   "

			print(string)

			string = "   "

			for i in range(self.state_dict["num_options"]):
				string += (block_length * line)
				string += "   "

			print(string)
			print("\n")

		print("##############################################\n")


def main():

	option_names = ["Apple", "Orange", "Banana", "Bread"]
	pos_info = {}
	pos_info[0] = {"name": "Orange"}
	pos_info[1] = {"name": "Bread"}

	num_options = len(option_names)
	num_actions = 20

	uncertainty_range = [0.5, 0.5]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	act_model = Action_Ideal(num_actions)

	prob_model = Probability_Ideal(num_options, num_actions, uncertainty_range, device)

	state_dict = gen_state_dict(pos_info, option_names)

	decision_model = Decision_Model(state_dict, prob_model, act_model)


	num_trials = 1000
	pol_type = [0, 1, 2]
	step_counts = np.zeros((len(pol_type), num_trials))

	pol_idx = 0
	trial_idx = 0

	while trial_idx < num_trials:
		pos_idx, act_idx = decision_model.choose_action(pol_idx)
		options_idx = decision_model.state_dict["correct_state"][pos_idx]
		decision_model.new_obs(options_idx)
		decision_model.print_hypothesis()
		a = input("Continue?")

		if decision_model.curr_entropy < 0.3:
			step_counts[pol_idx, trial_idx] =  decision_model.step_count
			pol_idx += 1

			if pol_idx == len(pol_type):
				decision_model.prob_model.gen_cm()
				pol_idx = pol_idx % len(pol_type)
				trial_idx += 1

			decision_model.reset_dis()

			if (trial_idx + 1) % 50 == 0 and pol_idx == (len(pol_type) - 1):
				print(trial_idx + 1, " trials completed of ", num_trials)

	print("Averages steps: ", np.mean(step_counts, axis = 1))

if __name__ == "__main__":
	main()

		#### Insertion calculation #####
		# insert_probs = self.calc_insert_probs(macro_actions).squeeze()

		# curr_peg_memory = self.marginalize(self.hole_memory, shape_idx = self.peg_idx)[0]

		# insert_prob = insert_probs.max(0)[0] * curr_peg_memory.max()

		# max_insert_idx = insert_probs.max(0)[1]
		# hole_insert_idx = curr_peg_memory.argmax()

		# print("##################")
		# print("# Action Choice #\n")
		# if self.marginalize(self.hole_memory, shape_idx = self.peg_idx)[0].max() > 0.9:
		# 	print("CHOOSING TO INSERT")
		# 	self.hole_idx = hole_insert_idx
		# 	self.insert_bool = 1.0
		# 	max_idx = max_insert_idx
		# else:

	# def calc_insert_probs(self, macro_actions):
	# 	peg_type = self.expand(self.peg_idx, macro_actions.size(0))
	# 	return torch.sigmoid(self.insert_model.process(macro_actions, peg_type, peg_type))
# def mat_conf_logprobs(peg_type, hole_type, probs):
# 	conf_logprobs = torch.zeros_like(peg_type[:,0])
# 	certainty = 180
# 	uncertainty = 20

# 	conf_matrix = np.array([\
# 		[[certainty, uncertainty, uncertainty],\
# 		[uncertainty, certainty, uncertainty],\
# 		[uncertainty, uncertainty, certainty]],\
# 		[[certainty, uncertainty, uncertainty],\
# 		[uncertainty, certainty, uncertainty],\
# 		[uncertainty, uncertainty, certainty]],\
# 		[[certainty, uncertainty, uncertainty],\
# 		[uncertainty, certainty, uncertainty],\
# 		[uncertainty, uncertainty, certainty]],\
# 		])

# 	conf_matrix = conf_matrix / np.tile(np.expand_dims(np.sum(conf_matrix, axis = 2), axis = 2), (1,1,peg_type.shape[1]))

# 	conf_matrix = torch.log(torch.from_numpy(conf_matrix).float().to(peg_type.device))

# 	for i in range(peg_type.size(0)):
# 		p_idx = peg_type[i].max(0)[1]
# 		h_idx = hole_type[i].max(0)[1]
# 		c_idx = probs[i].max(0)[1]

# 		conf_logprobs[i] = conf_matrix[p_idx, h_idx, c_idx]

# 	return conf_logprobs


# def calc_entropy(belief):
# 	return np.where(belief != 0, -1.0 * belief * np.log(belief), 0).sum(-1)