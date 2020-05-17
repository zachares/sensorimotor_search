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

class Action_PegInsertion(object):
	def __init__(self, hole_info, workspace_dim, tol, plus_offset, num_actions):
		self.plus_offset = plus_offset
		self.workspace_dim = workspace_dim
		self.hole_info = hole_info
		self.num_actions = num_actions
		self.tol = tol

	def generate_actions(self):
		self.actions = slidepoints(self.workspace_dim, self.tol, self.num_actions)

	def get_action(self, act_idx):
		return self.actions[act_idx]

	def transform_action(self, pos_idx, act_idx):
		action = self.actions[act_idx]
		top_goal = self.hole_info[pos_idx]["pos"]
		init_point = action[:3] + top_goal
		final_point = action[6:] + top_goal

		top_plus = top_goal + self.plus_offset

		return [(top_plus, 0, "top_plus"), (init_point, 0, "init_point"), (final_point, 1, "final_point"), (top_plus, 0, "top_plus")]

class Probability_PegInsertion(object):
	def __init__(self, sensor, confusion_model, num_tools, num_options):
		self.sensor = sensor
		self.conf = confusion_model
		self.num_tools = num_tools
		self.num_options = num_options
		self.device = self.sensor.device
		self.step_tensor = self.toTorch(np.array([75])).unsqueeze(0)

	def toTorch(self, array):
		return torch.from_numpy(array).to(self.device).float()

	def expand(self, idx, size0, size1):
		vector = np.zeros(size1)
		vector[idx] = 1
		return self.toTorch(vector).unsqueeze(0).repeat_interleave(size0, dim = 0)

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

		sample['peg_type'] = self.expand(self.tool_idx, 1, self.num_tools)
		# sample['macro_action'] = torch.cat([action, self.step_tensor], axis = 1)

		return self.sensor.probs(sample).max(1)[1]

	def conf_logits(self, actions_np):
		actions = self.toTorch(actions_np)
		batch_size = actions.size(0)

		peg_type = self.expand(self.tool_idx, batch_size, self.num_tools)

		macro_actions = torch.cat([actions, self.step_tensor.repeat_interleave(batch_size, dim = 0) ], axis = 1)

		return self.conf.logits(peg_type, macro_actions)

	def conf_logprob(self, options_idx, actions_np, obs_idx):
		actions = self.toTorch(actions_np)
		batch_size = actions.size(0)

		tool_type = self.expand(self.tool_idx, batch_size, self.num_tools)

		options_type = self.expand(options_idx, batch_size, self.num_options)

		macro_actions = torch.cat([actions, self.step_tensor.repeat_interleave(batch_size, dim = 0) ], axis = 1)

		return self.conf.conf_logprobs(tool_type, options_type, macro_actions, obs_idx)

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
		return torch.log(self.toTorch(self.conf[actions]))

	def conf_logprob(self, options_idx, act_idx, obs_idx):
		return torch.log(self.toTorch(self.conf[act_idx, [options_idx]*len(act_idx), [obs_idx]*len(act_idx)]))