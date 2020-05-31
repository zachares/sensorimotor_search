import numpy as np
import time
import sys
import copy
import os
import random
import itertools

import torch
import torch.nn.functional as F

sys.path.insert(0, "../") 
sys.path.insert(0, "../supervised_learning/") 

from multinomial import *
from project_utils import *
from task_models import *

class Decision_Model(object):
	def __init__(self, state_dict, prob_model, act_model, policy_idxs):
		self.state_dict = state_dict
		self.prob_model = prob_model
		self.act_model = act_model
		self.device = self.prob_model.device
		self.policy_idxs = policy_idxs

	def _record_correct_state(self, correct_substates):
		for k, state in enumerate(self.state_dict["states"]):
			correct = True
			for i, j in enumerate(state):
				if correct_substates[i] != self.state_dict["substate_names"][j]:
					correct = False

			if correct:
				self.state_dict["correct_state"] = state
				self.state_dict["correct_idx"] = k


		print("Correct State: ", self.state_dict["correct_state"])

	def _record_correct_options(self, correct_options):
		correct_idxs = []

		for option in correct_options:
			correct_idxs.append(self.state_dict["option_names"].index(option))

		self.state_dict["correct_options"] = correct_idxs

		print("Correct State: ", self.state_dict["correct_options"])

	def reset_probs(self, correct_substates, correct_options):
		self.state_probs = torch.ones((1,self.state_dict["num_states"])) / self.state_dict["num_states"]
		self.state_probs = self.state_probs.to(self.device).float()
		self.max_entropy = inputs2ent(probs2inputs(self.state_probs)).item()
		self.curr_state_entropy = inputs2ent(probs2inputs(self.state_probs)).item()

		self.step_count = 0
		self.num_misclassifcations = 0

		self._record_correct_state(correct_substates)
		self._record_correct_options(correct_options)

	def robust_cand(self):
		max_entropy = 0
		for i in range(self.state_dict["num_cand"]):
			probs = self.marginalize(self.state_probs, cand_idx = i)
			entropy = inputs2ent(probs2inputs(probs)).item()
			# print(entropy)

			if entropy > max_entropy:
				idx = i
				max_entropy = entropy

		return idx

	def robust_action(self, actions):
		logits = self.prob_model.conf_logits(actions)
		div_js = pairwise_divJS(logits)

		return div_js.max(0)[1]

	def choose_action(self, pol_idxs = None):
		if pol_idxs is None:
			policy_idxs = self.policy_idxs
		else:
			policy_idxs = pol_idxs

		self.act_model.generate_actions()
		print("Current entropy is: " + str(self.curr_state_entropy)[:5])

		cand_pol_idx = policy_idxs[0]
		action_pol_idx =policy_idxs[1]

		if cand_pol_idx == 0:
			print("Candidate Chosen Randomly")
		elif cand_pol_idx == 1:
			print("Candidate Chosen using Robust Criteria")
		elif cand_pol_idx == 2:
			print("Candidate Chosen to Minimize Entropy")
		else:
			raise Exception(str(cand_pol_idx) + " candidate policy number is not a valid number")

		if action_pol_idx == 0:
			print("Action Chosen Randomly")
		elif action_pol_idx == 1:
			print("Action Chosen using Robust Criteria")
		elif action_pol_idx == 2:
			print("Action Chosen to Minimize Entropy")
		else:
			raise Exception(str(action_pol_idx) + " action policy number is not a valid number")


		if cand_pol_idx == 0:
			cand_idx = np.random.choice(range(self.state_dict["num_cand"]))

		elif cand_pol_idx == 1:
			cand_idx = self.robust_cand()

		if action_pol_idx == 0:
			act_idx = np.random.choice(range(self.act_model.num_actions))

		elif action_pol_idx == 1:
			act_idx = self.robust_action(self.act_model.actions)


		if cand_pol_idx == 2 and action_pol_idx != 2:
			# cand_idx1 = self.robust_cand()
			min_entropy = copy.deepcopy(self.max_entropy)
			action = np.expand_dims(self.act_model.actions[act_idx], axis = 0)

			for k in range(self.state_dict["num_cand"]):
				exp_entropy = self.expected_entropy(k, action)

				if exp_entropy.min(0)[0] < min_entropy:
					cand_idx = copy.deepcopy(k)
					min_entropy = exp_entropy.min(0)[0]

		elif cand_pol_idx != 2 and action_pol_idx == 2:
			exp_entropy = self.expected_entropy(cand_idx, self.act_model.actions)
			act_idx = exp_entropy.min(0)[1]

		elif cand_pol_idx == 2 and action_pol_idx == 2:
			min_entropy = copy.deepcopy(self.max_entropy)

			for k in range(self.state_dict["num_cand"]):
				exp_entropy = self.expected_entropy(k, self.act_model.actions)

				if exp_entropy.min(0)[0] < min_entropy:
					cand_idx = copy.deepcopy(k)
					act_idx = exp_entropy.min(0)[1]
					min_entropy = exp_entropy.min(0)[0]

		self.cand_idx = cand_idx
		self.act_idx = act_idx

		print("\n#####################\n")

		return self.act_model.transform_action(self.cand_idx, self.act_idx)

	def marginalize(self, state_probs, cand_idx = None, substate_idx = None):
		if cand_idx is not None:
			margin = torch.zeros((self.state_dict["num_substates"], state_probs.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				margin[state[cand_idx]] += state_probs.transpose(0,1)[i]
		else:
			margin = torch.zeros((self.state_dict["num_cand"], state_probs.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				if substate_idx in state:
					idx = state.index(substate_idx)
					margin[idx] += state_probs.transpose(0,1)[i]

		return margin.transpose(0,1)

	def expected_entropy(self, cand_idx, actions):
		batch_size = actions.shape[0]

		state_prior =self.state_probs.repeat_interleave(batch_size, dim = 0)
		# sp = torch.zeros((1,self.state_dict["num_states"])) / self.state_dict["num_states"]
		# sp[:,self.state_dict["correct_idx"]] = 1.0
		# state_prior = sp.float().to(self.device).repeat_interleave(batch_size, dim =0)

		expected_entropy = torch.zeros(batch_size).float().to(self.device)


		for obs_idx in range(self.state_dict["num_options"]):
			p_o_given_a = torch.zeros(batch_size).float().to(self.device)
			state_posterior_logprobs = torch.zeros_like(state_prior)

			for j, state in enumerate(self.state_dict["states"]):
				substate_idx = state[cand_idx]
				conf_logprobs = self.prob_model.conf_logprob(substate_idx, actions, obs_idx)

				p_o_given_a += torch.where(state_prior[:,j] != 0, torch.exp(torch.log(state_prior[:,j]) + conf_logprobs), torch.zeros_like(state_prior[:,j]))
				
				state_posterior_logprobs[:,j] += torch.log(state_prior[:,j]) + conf_logprobs 

			state_posterior = logits2probs(state_posterior_logprobs)

			# print(state_probs)
			# print(p_o_given_a)
			expected_entropy += p_o_given_a * inputs2ent(probs2inputs(state_posterior))

			# idx = expected_entropy.min(0)[1]
			# print(state_probs[idx])
			# print(p_o_given_a[idx])
			# self.marginalize(state_probs, cand_idx = cand_idx)
			# self.marginalize(state_probs, substate_idx = self.tool_idx))

		return expected_entropy

	def update_probs(self, cand_idx, obs_idx, actions):
		batch_size = actions.shape[0]
		state_prior = self.state_probs.repeat_interleave(batch_size, dim = 0)
		state_posterior_logits = torch.zeros_like(state_prior)

		for j, state in enumerate(self.state_dict["states"]):
			substate_idx = state[cand_idx]
			conf_logprobs =self.prob_model.conf_logprob(substate_idx, actions, obs_idx)
			state_posterior_logits[:,j] = torch.log(state_prior[:,j]) + conf_logprobs

		state_posterior = logits2probs(state_posterior_logits)

		return state_posterior

	def new_obs(self, obs):
		if self.prob_model.record_test(obs):
			action = self.act_model.get_action(self.act_idx)
			# print("Original Action: ", action)
			action = self.prob_model.transform_action(obs, action)
			# print("Transformed Action: ", action)
			obs_idx = self.prob_model.sensor_obs(obs, action)
			# print("Action Index: ", self.act_idx)
			# print("Cand Index: ", self.cand_idx)
			# print("Observation Index: ", obs_idx.item())
			# print("State Distribution: ", self.state_probs)

			self.state_probs = self.update_probs(self.cand_idx, obs_idx, action)
			# print("Updated State Distribution: ", self.state_probs)



			self.step_count += 1

			corr_idx = self.state_dict['correct_options'][self.cand_idx]

			self.calc_metrics()

			print("Obs idx: ", obs_idx.item(), " Corr idx: ", corr_idx)
			if corr_idx != obs_idx.item():
				self.num_misclassifcations += 1

	def calc_metrics(self):
		margin_probs = self.marginalize(self.state_probs, substate_idx = self.tool_idx)

		self.curr_state_prob = self.state_probs[0,self.state_dict['correct_idx']].item()
		self.curr_state_entropy = inputs2ent(probs2inputs(self.state_probs)).item()
		self.curr_fit_entropy = inputs2ent(probs2inputs(margin_probs)).item()
		self.curr_fit_prob = margin_probs[0,self.tool_idx].item()

	def print_hypothesis(self):
		for i in range(self.state_dict["num_cand"]):
			probs = self.marginalize(self.state_probs, cand_idx = i)

			print("Model Hypthesis:", self.state_dict["substate_names"][probs.max(1)[1]] ,\
			 " , Ground Truth:", self.state_dict["substate_names"][self.state_dict["correct_state"][i]])

			print("Probabilities", probs.detach().cpu().numpy())

			print_histogram(probs, self.state_dict["substate_names"])
		print("##############################################\n")	

def main():

	options_names = ["Apple", "Orange", "Banana", "Bread"]
	cand_info = {}
	cand_info[0] = {"name": "Orange"}
	cand_info[1] = {"name": "Bread"}

	num_options = len(options_names)
	num_actions = 20

	uncertainty_range = [0.5, 0.5]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	act_model = Action_Ideal(num_actions)

	prob_model = Probability_Ideal(num_options, num_actions, uncertainty_range, device)

	state_dict = gen_state_dict(cand_info, options_names)

	decision_model = Decision_Model(state_dict, prob_model, act_model)


	num_trials = 1000
	pol_type = [0, 1, 2]
	step_counts = np.zeros((len(pol_type), num_trials))

	pol_idx = 0
	trial_idx = 0

	while trial_idx < num_trials:
		cand_idx, act_idx = decision_model.choose_action(pol_idx)
		options_idx = decision_model.state_dict["correct_state"][cand_idx]
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

			decision_model.reset_probs()

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
