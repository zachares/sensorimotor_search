import copy
import os
import sys
import random
import itertools

import torch
import torch.nn.functional as F

sys.path.insert(0, "../") 
sys.path.insert(0, "../supervised_learning/") 

from multinomial import *
from project_utils import *

class Probability_PegInsertion(object):
	def __init__(self, sensor, likelihood_model):
		self.sensor = sensor
		self.likelihood = likelihood_model
		self.device = self.sensor.device

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.device).float()

	def sensor_obs(self, obs)
		sample = obs2Torch(obs, self.device)
		return self.sensor.probs(sample)

	def likelihood_logprobs(self, tool_idx, substates_idx, obs_idx, actions_partial):
		actions = self.toTorch(actions_np)
		batch_size = actions.size(0)

		tool_type = self.expand(self.tool_idx, batch_size, self.num_tools)

		substates_type = self.expand(substates_idx, batch_size, self.num_substates)

		macro_actions = torch.cat([actions, self.step_tensor.repeat_interleave(batch_size, dim = 0) ], axis = 1)

		return self.likelihood.logprobs(tool_type, substates_type, macro_actions, obs_idx)

	# def expand(self, idx, size0, size1):
	# 	vector = np.zeros(size1)
	# 	vector[idx] = 1
	# 	return self.toTorch(vector).unsqueeze(0).repeat_interleave(size0, dim = 0)

	# def conf_logits(self, actions_np):
	# 	actions = self.toTorch(actions_np)
	# 	batch_size = actions.size(0)

	# 	peg_type = self.expand(self.tool_idx, batch_size, self.num_tools)

	# 	macro_actions = torch.cat([actions, self.step_tensor.repeat_interleave(batch_size, dim = 0) ], axis = 1)

	# 	return self.conf.logits(peg_type, macro_actions)

# class Action_Ideal(object):
# 	def __init__(self, num_actions):
# 		self.num_actions = num_actions

# 	def generate_actions(self):
# 		if self.num_actions == 0:
# 			return np.array([0])
# 		else:
# 			self.actions = np.array(range(self.num_actions))

# 	def get_action(self, act_idx):
# 		return self.actions[act_idx]

# 	def transform_action(self, pos_idx, act_idx):
# 		return (pos_idx, act_idx)

# class Probability_Ideal(object):
# 	def __init__(self, num_options, num_actions, uncertainty_range, device):
# 		self.num_options = num_options
# 		self.num_actions = num_actions
# 		self.u_r = uncertainty_range
# 		self.device = device

# 		self.gen_cm()

# 	def gen_cm(self):
# 		self.conf = gen_conf_mat(self.num_options, self.num_actions, self.u_r)

# 	def toTorch(self, array):
# 		return torch.from_numpy(array).to(self.device).float()

# 	def record_test(self, obs):
# 		return True

# 	def transform_action(self, options_idx, act_idx):
# 		return np.array([act_idx])

# 	def sensor_obs(self, options_idx, act_idx):
# 		probs = self.conf[act_idx[0], options_idx]
# 		return np.random.multinomial(1, probs, size = 1).argmax()

# 	def conf_logits(self, actions):
# 		return torch.log(self.toTorch(self.conf[actions]))

# 	def conf_logprob(self, options_idx, act_idx, obs_idx):
# 		return torch.log(self.toTorch(self.conf[act_idx, [options_idx]*len(act_idx), [obs_idx]*len(act_idx)]))