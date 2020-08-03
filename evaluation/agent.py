import numpy as np
import time
import sys
import copy
import os
import random
import itertools

import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from collections import OrderedDict

import multinomial as multinomial
import project_utils as pu

class Outer_Loop(object):
	def __init__(self, env, task_dict, mode = 0, success_rate = None, k = 0, device = None):

		self.env = env
		self.task_dict = task_dict

		self.success_rate = success_rate

		self.device = device
		self.k = k
		self.mode = mode
		self.sample = True
		
		self.print_info = False
		self.reset()

	def reset(self):
		self.env.reset()
		self.gt_dict = self.env.get_gt_dict()
		self.reset_probs()
		self.step_count = 1
		self.action_counts = torch.ones(self.task_dict['num_actions']).float().to(self.device)

		if self.print_info and self.use_state_est:
			self.print_hypothesis()
		
	def reset_probs(self):
		self.state_probs = torch.ones((self.task_dict["num_states"])) / self.task_dict["num_states"]
		self.state_probs = self.state_probs.float().to(self.device)

		# self.max_action_entropy = multinomial.inputs2ent(multinomial.probs2inputs(torch.ones((self.task_dict["num_actions"])) / self.task_dict["num_substates"]))
		# self.max_state_entropy = multinomial.inputs2ent(multinomial.probs2inputs(self.state_probs)).item()		
		# self.curr_state_entropy = multinomial.inputs2ent(multinomial.probs2inputs(self.state_probs)).item()
		
	def choose_action(self):
		if self.mode == 0: # random
			action_idx = random.choice(range(self.task_dict['num_actions']))

		elif self.mode == 1: # iterator
			value_estimate = torch.sqrt(np.log(self.step_count) / self.action_counts)

			weights = (value_estimate / value_estimate.sum()).cpu().numpy()

			action_idx = np.argmax(weights)

			# if sum(weights[:-1]) > 1.0:
			# 	action_idx = np.argmax(weights)
			# else:
			# 	action_idx = np.argmax(np.random.multinomial(1, weights, size = 1))

		elif self.mode == 2: # greedy
			value_estimate = self.pomdp_value_iteration(self.state_probs, k = self.k)

			weights = (value_estimate / value_estimate.sum()).cpu().numpy()

			action_idx = np.argmax(weights)

			# if sum(weights[:-1]) > 1.0:
			# 	action_idx = np.argmax(weights)
			# else:
			# 	action_idx = np.argmax(np.random.multinomial(1, weights, size = 1))

		else:
			raise Exception('Mode ', self.mode, ' current unsupported')

		if self.print_info:
			print("Index of hole choice: ", action_idx)

		return action_idx  

	def pomdp_value_iteration(self, state_probs, k = 0): # depth of value iteration, calculating expected value
		alpha_vectors = pu.toTorch(self.task_dict['alpha_vectors'][self.env.get_goal()], self.device)
		state_prior = state_probs.unsqueeze(0).repeat_interleave(self.task_dict['num_actions'],0)

		expected_reward = (alpha_vectors * state_prior).sum(1)

		if k == 0:
			return expected_reward * self.success_rate
		else:
			state_prior =state_prior.unsqueeze(0).repeat_interleave(self.task_dict['num_obs'], 0)
			
			loglikelihood_matrix = pu.toTorch(self.task_dict['loglikelihood_matrix'], self.device)

			state_posterior = F.softmax(torch.log(state_prior) + loglikelihood_matrix, dim = 2)
		
			p_o_given_a = torch.where(state_prior != 0,\
			 torch.exp(torch.log(state_prior) + loglikelihood_matrix), torch.zeros_like(state_prior)).sum(-1)

			lookahead_value = torch.zeros_like(p_o_given_a)

			for obs_idx in range(self.task_dict['num_obs']):
				for act_idx in range(self.task_dict['num_actions']):
					lookahead_value[obs_idx, act_idx] = self.pomdp_value_iteration(state_posterior[obs_idx, act_idx], k = k-1).max(0)[0]

			expected_lookahead_value = (p_o_given_a * lookahead_value).sum(0)

			return self.success_rate * expected_reward + (1 - self.success_rate) * expected_lookahead_value

	def update_state_probs(self, action_idx, obs_idxs):
		state_prior = self.state_probs.clone()
		obs_idx = self.task_dict['obs2idx'][obs_idxs]
		ll_logprobs = pu.toTorch(self.task_dict['loglikelihood_matrix'][obs_idx,action_idx], self.device)
		# print(ll_logprobs)
		state_posterior_logits = torch.log(state_prior) + ll_logprobs
		# print(F.softmax(state_posterior_logits, dim = 0))
		self.state_probs = multinomial.logits2probs(state_posterior_logits)
		self.curr_state_entropy = multinomial.inputs2ent(multinomial.probs2inputs(self.state_probs)).item()
		# print(self.state_probs)
		# a = input("Continue?")

	def new_obs(self, action_idx, obs_idxs):
		# obs_idxs is a tuple of indicies to the likelihood matrix element to look up
		self.action_counts[action_idx] += 1
		self.step_count += 1

		if obs_idxs != None:
			# print(self.state_probs)
			self.update_state_probs(action_idx, obs_idxs)
			# print(self.state_probs)

			### ignoring observation of insertion or not insertion
			if self.print_info:
				for i, obs_idx in enumerate(obs_idxs):
					obs = self.task_dict['obs_names'][i][obs_idx]

					print("Observation_" + str(i) + " " + obs)

				gt_props = "Ground Truth Properties"
				for info in self.gt_dict[action_idx]:
					if type(info) == str:
						gt_props += " - " + info 

				print(gt_props)
				self.print_hypothesis()
		else:
			if self.print_info:
				print("Bad Observation")

	def print_hypothesis(self):
		state_probs = self.state_probs.unsqueeze(0).unsqueeze(0).repeat_interleave(self.task_dict['num_actions'], 0)\
		.repeat_interleave(self.task_dict['num_substates'], 1)

		beta_vectors = pu.toTorch(self.task_dict['beta_vectors'], self.device)

		substate_probs = (beta_vectors * state_probs).sum(-1)

		for act_idx in range(self.task_dict["num_actions"]):
			probs = substate_probs[act_idx]

			print("Model Hypthesis:", self.task_dict["substate_names"][probs.max(0)[1]] ,\
			 " , Ground Truth:", self.gt_dict[act_idx][0])

			print("Probabilities", probs.detach().cpu().numpy())

			pu.print_histogram(probs, self.task_dict["substate_names"])
		print("##############################################\n")	

	# def marginalize(self, state_probs, action_idx = None, substate_idx = None):
	# 	#### marginalize out other actions / hole choices
	# 	if action_idx is not None:
	# 		margin = torch.zeros((self.task_dict["num_substates"])).to(self.device)
	# 		for i, state in enumerate(self.task_dict["states"]):
	# 			margin[state[action_idx]] += state_probs[i]

	# 	#### marginalize out other state types
	# 	else:
	# 		margin = torch.zeros((self.task_dict["num_actions"])).to(self.device)
	# 		for i, state in enumerate(self.task_dict["states"]):
	# 			if substate_idx in state:
	# 				idx = state.index(substate_idx)
	# 				margin[idx] += state_probs[i]

	# 	return margin

def main():
	## success rate if correct fruit is chosen
	device = torch.device("cuda")
	# Toy problem
	fruit_properties = [
	['apple', 'sweet', 'citric_acid'],
	['orange', 'sour', 'citric_acid'],
	['lemon', 'sweet', 'citric_acid'],
	['banana', 'sweet', 'potassium']
	]

	substate_names = []

	for fp in fruit_properties:
		substate_names.append(fp[0])

	observation_names = [['sweet', 'sour'], ['potassium', 'citric_acid']]

	goal = 'orange'
	goal_idx = substate_names.index(goal)

	### uncertainty addition to observation model
	uncertainty = 0.1

	num_each_fruit = [1,1,1,1]
	corr_idxs = []

	cand_fruits = OrderedDict()
	cand_num = 0

	for i, num_fruit in enumerate(num_each_fruit):
		fp =  fruit_properties[i]
		for j in range(num_fruit):
			cand_fruits[cand_num] = fp

			if goal in fp:
				corr_idxs.append(cand_num)

			cand_num += 1

	num_fruits = len(cand_fruits.keys())

	### generating observation model
	likelihood_model = torch.zeros((len(substate_names)))

	for obs_names in observation_names:
		likelihood_model = likelihood_model.unsqueeze(-1).repeat_interleave(len(obs_names), -1)

	for i, fp in enumerate(fruit_properties):
		obs_idxs = []

		for obs_names in observation_names:
			for j, obs in enumerate(obs_names):
				if obs in fp:
					obs_idxs.append(j)

		obs_idxs = tuple([i] + obs_idxs)
		likelihood_model[obs_idxs] = 1.0

	### adding uncertainty and normalizing
	noise = Uniform(0, uncertainty)
	likelihood_model += noise.rsample(sample_shape=likelihood_model.size())
	likelihood_model = likelihood_model / likelihood_model.sum()

	# print("Success Rate: ", success_rate)

	class Fruit_Env(object):
		def __init__(self, goal_idx, cand_fruits, likelihood_model):
			self.goal_idx = goal_idx
			self.cand_fruits = cand_fruits
			self.likelihood_model = likelihood_model

		def reset(self):
			pass

		def get_goal(self):
			return self.goal_idx

		def get_gt_dict(self):
			return self.cand_fruits

		def get_obs(self, action_idx):
			substate_idx = substate_names.index(self.cand_fruits[action_idx][0])

			logits = self.likelihood_model[substate_idx]
			probs = logits / logits.sum()
			probs_shape = probs.size()

			probs_flattened = torch.flatten(probs)

			obs_idx_flat = np.argmax(np.random.multinomial(1, probs_flattened.numpy(), size = 1))

			sample_flattened = torch.zeros((probs_flattened.size()))
			sample_flattened[obs_idx_flat] = 1.0

			obs_idx = np.unravel_index(np.argmax(sample_flattened.numpy()), probs_shape)

			# print(self.cand_fruits[action_idx])
			# print("Logits: ", logits)
			# print("Probs: ", probs)
			# print("Probs Flattened: ", probs_flattened)
			# print("Obs_idx_flat: ", obs_idx_flat)
			# print("Sample Flattened: ", sample_flattened)
			# print("Obs idx: ", obs_idx)
			# a = input("Continue?")
			return obs_idx

	fruit_env = Fruit_Env(goal_idx, cand_fruits, likelihood_model)

	success_rate =  0.7

	task_dict = pu.gen_task_dict(num_fruits, substate_names, observation_names,\
	 likelihood_model = likelihood_model.cpu().numpy(), constraint_type = 1)

	k = 1

	print("Num steps: ", k)
	decision_model = Outer_Loop(fruit_env, task_dict, success_rate = success_rate, k=k, device = device)

	'''
	Questions
	1. How can you automatically choose the discount factor
	2. How can you automattically choose the number of steps for lookahead
	'''

	decision_model.mode = 2

	num_trials = 10000

	trial_idx = 0
	step_counts = []
	# c_range = np.arange(0, 5, 0.5)

	# for i in range(c_range.size):
	# 	decision_model.c = c_range[i]
	# 	trial_idx = 0

	while trial_idx < num_trials:
		# print(trial_idx)
		# print('\n #################### \n New Task \n ###################### \n')
		decision_model.reset()
		done_bool = False
		num_steps = 0
		while not done_bool:
			act_idx = decision_model.choose_action()
			obs_idxs = decision_model.env.get_obs(act_idx)
			num_steps += 1

			# print("Comparison", act_idx, corr_idxs)
			if act_idx in corr_idxs:
				if np.random.binomial(1, success_rate, 1) == 1:
					done_bool = True
		
			decision_model.new_obs(act_idx, obs_idxs)

			# a = input("Continue?")

		step_counts.append(num_steps)

		trial_idx += 1

	print("Averages Number of Steps: ", np.mean(step_counts))
	print("Standard Deviation of Steps: ", np.std(step_counts))
		# print("Averages Number of Steps: ", np.mean(step_counts), ' c parameter', c_range[i])

if __name__ == "__main__":
	main()


		# elif self.mode == 3: # greedy with uncertainty
		# 	state_probs = self.state_probs.unsqueeze(0).unsqueeze(0).repeat_interleave(self.task_dict['num_actions'], 0)\
		# 	.repeat_interleave(self.task_dict['num_substates'], 1)

		# 	beta_vectors = pu.toTorch(self.task_dict['beta_vectors'], self.device)

		# 	action_entropy = multinomial.inputs2ent(multinomial.probs2inputs((beta_vectors * state_probs).sum(-1)))

		# 	certainty = (1 - action_entropy / self.max_action_entropy)

		# 	if (torch.isnan(certainty) * 1.0).sum() > 0:
		# 		weighted_value_estimate = self.pomdp_value_iteration(self.state_probs, k = self.k)
		# 	else:
		# 		value_estimate = self.pomdp_value_iteration(self.state_probs, k = self.k)
		# 		weighted_value_estimate = certainty * value_estimate

		# 	weights = (weighted_value_estimate / weighted_value_estimate.sum()).cpu().numpy()

		# 	action_idx = np.argmax(weights)

			# if sum(weights[:-1]) > 1.0:
			# 	action_idx = np.argmax(weights)
			# else:
			# 	action_idx = np.argmax(np.random.multinomial(1, weights, size = 1))

			# info_value_est = self.info_value_est(self.state_probs)
			# action_idx = (value_estimate + info_value_est).max(0)[1].item()

# def gen_likelihood_mat(num_obs, num_actions, uncertainty_range):
# 	if num_actions == 0:
# 		return np.expand_dims(create_confusion(num_obs, uncertainty_range), axis = 0)
# 	else:
# 		conf_mat = []
# 		for i in range(num_actions):
# 			conf_mat.append(np.expand_dims(gen_likelihood(num_obs, uncertainty_range), axis = 0))

# 		return np.concatenate(conf_mat, axis = 0)

# def gen_likelihood(num_obs, uncertainty_range):
# 	uncertainty = uncertainty_range[0] + (uncertainty_range[1] - uncertainty_range[0]) * np.random.random_sample()
# 	confusion_matrix = uncertainty * np.random.random_sample((num_options, num_options))

# 	confusion_matrix[range(num_options), range(num_options)] = 1.0

# 	confusion_matrix = confusion_matrix / np.tile(np.expand_dims(np.sum(confusion_matrix, axis = 1), axis = 1), (1, num_options))

# 	return confusion_matrix

			# print("Action", action)
			# print("Partial Action", action_partial)
	# def robust_action(self):
	# 	max_entropy = 0
	# 	for i in range(self.state_dict["num_cand"]):
	# 		probs = self.marginalize(self.state_probs, cand_idx = i)
	# 		entropy = inputs2ent(probs2inputs(probs)).item()
	# 		# print(entropy)

	# 		if entropy > max_entropy:
	# 			idx = i
	# 			max_entropy = entropy

	# 	return idx

	# def robust_action(self, actions):
	# 	logits = self.prob_model.conf_logits(actions)
	# 	div_js = pairwise_divJS(logits)

	# 	return div_js.max(0)[1]

	# def expected_entropy(self, state_probs, cand_idx, actions):
	# 	batch_size = actions.shape[0]

	# 	state_prior =state_probs.repeat_interleave(batch_size, dim = 0)

	# 	expected_entropy = torch.zeros(batch_size).float().to(self.device)

	# 	for obs_idx in range(self.state_dict["num_options"]):
	# 		p_o_given_a = torch.zeros(batch_size).float().to(self.device)
	# 		state_posterior_logprobs = torch.zeros_like(state_prior)

	# 		for j, state in enumerate(self.state_dict["states"]):
	# 			substate_idx = state[cand_idx]
	# 			conf_logprobs = self.prob_model.conf_logprob(substate_idx, actions, obs_idx)

	# 			p_o_given_a += torch.where(state_prior[:,j] != 0, torch.exp(torch.log(state_prior[:,j]) + conf_logprobs),\
	# 			 torch.zeros_like(state_prior[:,j]))
				
	# 			state_posterior_logprobs[:,j] = torch.log(state_prior[:,j]) + conf_logprobs 

	# 		expected_entropy += p_o_given_a * inputs2ent(logits2inputs(state_posterior_logprobs))

	# 	return expected_entropy

# def main():

# 	options_names = ["Apple", "Orange", "Banana", "Bread"]
# 	cand_info = {}
# 	cand_info[0] = {"name": "Orange"}
# 	cand_info[1] = {"name": "Bread"}

# 	num_options = len(options_names)
# 	num_actions = 20

# 	uncertainty_range = [0.5, 0.5]

# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	act_model = Action_Ideal(num_actions)

# 	prob_model = Probability_Ideal(num_options, num_actions, uncertainty_range, device)

# 	state_dict = gen_state_dict(cand_info, options_names)

# 	decision_model = Decision_Model(state_dict, prob_model, act_model)


# 	num_trials = 1000
# 	pol_type = [0, 1, 2]
# 	step_counts = np.zeros((len(pol_type), num_trials))

# 	pol_idx = 0
# 	trial_idx = 0

# 	while trial_idx < num_trials:
# 		cand_idx, act_idx = decision_model.choose_action(pol_idx)
# 		options_idx = decision_model.state_dict["correct_state"][cand_idx]
# 		decision_model.new_obs(options_idx)
# 		decision_model.print_hypothesis()
# 		a = input("Continue?")

# 		if decision_model.curr_entropy < 0.3:
# 			step_counts[pol_idx, trial_idx] =  decision_model.step_count
# 			pol_idx += 1

# 			if pol_idx == len(pol_type):
# 				decision_model.prob_model.gen_cm()
# 				pol_idx = pol_idx % len(pol_type)
# 				trial_idx += 1

# 			decision_model.reset_probs()

# 			if (trial_idx + 1) % 50 == 0 and pol_idx == (len(pol_type) - 1):
# 				print(trial_idx + 1, " trials completed of ", num_trials)

# 	print("Averages steps: ", np.mean(step_counts, axis = 1))

# if __name__ == "__main__":
# 	main()

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

		# if pol_idxs is None:
		# 	policy_idxs = self.policy_idxs
		# else:
		# 	policy_idxs = pol_idxs

		# self.act_model.generate_actions()
		# print("Current entropy is: " + str(self.curr_state_entropy)[:5])

		# cand_pol_idx = policy_idxs[0]
		# action_pol_idx =policy_idxs[1]

		# if cand_pol_idx == 0:
		# 	print("Candidate Chosen Randomly")
		# elif cand_pol_idx == 1:
		# 	print("Candidate Chosen using Robust Criteria")
		# elif cand_pol_idx == 2:
		# 	print("Candidate Chosen to Minimize Entropy")
		# else:
		# 	raise Exception(str(cand_pol_idx) + " candidate policy number is not a valid number")

		# if action_pol_idx == 0:
		# 	print("Action Chosen Randomly")
		# elif action_pol_idx == 1:
		# 	print("Action Chosen using Robust Criteria")
		# elif action_pol_idx == 2:
		# 	print("Action Chosen to Minimize Entropy")
		# else:
		# 	raise Exception(str(action_pol_idx) + " action policy number is not a valid number")


		# if cand_pol_idx == 0:
		# 	cand_idx = np.random.choice(range(self.state_dict["num_cand"]))

		# elif cand_pol_idx == 1:
		# 	cand_idx = self.robust_cand()

		# if action_pol_idx == 0:
		# 	act_idx = np.random.choice(range(self.act_model.num_actions))

		# elif action_pol_idx == 1:
		# 	act_idx = self.robust_action(self.act_model.actions)


		# if cand_pol_idx == 2 and action_pol_idx != 2:
		# 	# cand_idx1 = self.robust_cand()
		# 	min_entropy = copy.deepcopy(self.max_entropy)
		# 	action = np.expand_dims(self.act_model.actions[act_idx], axis = 0)

		# 	for k in range(self.state_dict["num_cand"]):
		# 		exp_entropy = self.expected_entropy(k, action)

		# 		if exp_entropy.min(0)[0] < min_entropy:
		# 			cand_idx = copy.deepcopy(k)
		# 			min_entropy = exp_entropy.min(0)[0]

		# elif cand_pol_idx != 2 and action_pol_idx == 2:
		# 	exp_entropy = self.expected_entropy(cand_idx, self.act_model.actions)
		# 	act_idx = exp_entropy.min(0)[1]

		# elif cand_pol_idx == 2 and action_pol_idx == 2:
		# 	min_entropy = copy.deepcopy(self.max_entropy)

		# 	for k in range(self.state_dict["num_cand"]):
		# 		exp_entropy = self.expected_entropy(k, self.act_model.actions)

		# 		if exp_entropy.min(0)[0] < min_entropy:
		# 			cand_idx = copy.deepcopy(k)
		# 			act_idx = exp_entropy.min(0)[1]
		# 			min_entropy = exp_entropy.min(0)[0]

		# self.cand_idx = cand_idx
		# self.act_idx = act_idx

		# print("\n#####################\n")

		# return self.act_model.transform_action(self.cand_idx, self.act_idx)