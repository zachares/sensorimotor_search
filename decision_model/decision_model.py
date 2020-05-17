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
	def __init__(self, state_dict, prob_model, act_model, policy_number):
		self.state_dict = state_dict
		self.prob_model = prob_model
		self.act_model = act_model
		self.device = self.prob_model.device
		self.policy_number = policy_number

	def _record_correct_state(self, correct_options):
		for k, state in enumerate(self.state_dict["states"]):
			correct = True
			for i, j in enumerate(state):
				if correct_options[i] != self.state_dict["options_names"][j]:
					correct = False

			if correct:
				self.state_dict["correct_state"] = state
				self.state_dict["correct_idx"] = k

		print("Correct State: ", self.state_dict["correct_state"])

	def reset_probs(self, correct_options):
		self.state_probs = torch.ones((1,self.state_dict["num_states"])) / self.state_dict["num_states"]
		self.state_probs = self.state_probs.to(self.device).float()
		self.max_entropy = probs2ent(self.state_probs).item()
		self.curr_entropy = probs2ent(self.state_probs).item()

		self.step_count = 0
		self.num_misclassifcations = 0

		self._record_correct_state(correct_options)

	def robust_cand(self):
		max_entropy = 0
		for i in range(self.state_dict["num_cand"]):
			probs = self.marginalize(self.state_probs, cand_idx = i).squeeze()
			entropy = probs2ent(probs)
			if entropy > max_entropy:
				idx = i
				max_entropy = entropy

		return idx

	def robust_action(self, actions):
		logits = self.prob_model.conf_logits(actions)
		div_js = pairwise_divJS(logits)

		return div_js.max(0)[1]

	def choose_action(self, pol_num = None):
		if pol_num is None:
			policy_number = self.policy_number
		else:
			policy_number = pol_num

		self.act_model.generate_actions()
		print("Current entropy is: " + str(self.curr_entropy)[:5])

		if policy_number == 0:
			cand_idx = np.random.choice(range(self.state_dict["num_cand"]))
			act_idx = np.random.choice(range(self.act_model.num_actions))

		elif policy_number == 1:
			cand_idx = self.robust_cand()
			act_idx = self.robust_action(self.act_model.actions)

		elif policy_number == 2:
			min_entropy = copy.deepcopy(self.max_entropy)

			for k in range(self.state_dict["num_cand"]):
				exp_entropy = self.expected_entropy(k, self.act_model.actions)

				if exp_entropy.min(0)[0] < min_entropy:
					cand_idx = copy.deepcopy(k)
					act_idx = exp_entropy.min(0)[1]
					min_entropy = exp_entropy.min(0)[0]
			
			print("Entropy is expected to decrease to: " + str(min_entropy.item())[:5])
		else:
			raise Exception(str(policy_number) + " policy number is not a valid number")

		self.cand_idx = cand_idx
		self.act_idx = act_idx

		print("\n#####################\n")

		return self.act_model.transform_action(self.cand_idx, self.act_idx)

	def marginalize(self, state_probs, cand_idx = None, options_idx = None):
		if cand_idx is not None:
			margin = torch.zeros((self.state_dict["num_options"], state_probs.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				margin[state[cand_idx]] += state_probs.transpose(0,1)[i]
		else:
			margin = torch.zeros((self.state_dict["num_cand"], state_probs.size(0))).to(self.device)
			for i, state in enumerate(self.state_dict["states"]):
				if options_idx in state:
					idx = state.index(options_idx)
					margin[idx] += state_probs.transpose(0,1)[i]

		return margin.transpose(0,1)

	def expected_entropy(self, cand_idx, actions):
		batch_size = actions.shape[0]
		prior_logprobs = torch.log(self.state_probs.repeat_interleave(batch_size, dim = 0))
		expected_entropy = torch.zeros(batch_size).float().to(self.device)

		for obs_idx in range(self.state_dict["num_options"]):
			p_o_given_a = torch.zeros(batch_size).float().to(self.device)

			for j, state in enumerate(self.state_dict["states"]):
				options_idx = state[cand_idx]

				conf_logprobs = self.prob_model.conf_logprob(options_idx, actions, obs_idx)

				# print(conf_logprobs.size())

				p_o_given_a += torch.exp(prior_logprobs[:,j] + conf_logprobs)

			expected_entropy += p_o_given_a * probs2ent(self.update_probs(cand_idx, obs_idx, actions))

		return expected_entropy

	def update_probs(self, cand_idx, obs_idx, actions):
		batch_size = actions.shape[0]
		prior_logprobs = torch.log(self.state_probs.repeat_interleave(batch_size, dim = 0))
		state_probs = torch.zeros_like(prior_logprobs)

		for j, state in enumerate(self.state_dict["states"]):
			# print("State: ", state)
			options_idx = state[cand_idx]
			# print("Options idx: ", options_idx)

			conf_logprobs =self.prob_model.conf_logprob(options_idx, actions, obs_idx)

			# print(conf_logprobs.size())

			state_probs[:,j] = conf_logprobs + prior_logprobs[:,j]

			# print(conf_logprobs)
			# print(state_probs)

		state_probs = F.softmax(state_probs, dim = 1)

		return state_probs

	def new_obs(self, obs):
		if self.prob_model.record_test(obs):
			action = self.act_model.get_action(self.act_idx)
			print("Original Action: ", action)
			action = self.prob_model.transform_action(obs, action)
			print("Transformed Action: ", action)
			obs_idx = self.prob_model.sensor_obs(obs, action)
			# print("Action Index: ", self.act_idx)
			# print("Cand Index: ", self.cand_idx)
			# print("Observation Index: ", obs_idx.item())
			# print("State Distribution: ", self.state_probs)

			self.state_probs = self.update_probs(self.cand_idx, obs_idx, action)
			print("Updated State Distribution: ", self.state_probs)

			self.curr_entropy = probs2ent(self.state_probs).item()
			self.step_count += 1

			corr_idx = self.state_dict['correct_state'][self.cand_idx]

			print("Obs idx: ", obs_idx.item(), " Corr idx: ", corr_idx)
			if corr_idx != obs_idx.item():
				self.num_misclassifcations += 1

	def print_hypothesis(self):
		for i in range(self.state_dict["num_cand"]):
			probs = self.marginalize(self.state_probs, cand_idx = i)

			print("Model Hypthesis:", self.state_dict["options_names"][probs.max(1)[1]] ,\
			 " , Ground Truth:", self.state_dict["options_names"][self.state_dict["correct_state"][i]])

			print("Probabilities", probs.detach().cpu().numpy())

			print_histogram(probs, self.state_dict["options_names"])
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
