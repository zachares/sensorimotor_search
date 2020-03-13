import io
import numpy as np
import sys
import copy
import yaml
from gym.envs.toy_text import discrete
from gridworld import GridworldEnv

import itertools

sys.path.insert(0, "../learning/datalogging/") 

from logger import Logger

def KL_div(p, q):
	p_idx = np.argwhere(p.flatten() > 0)

	p_dis = p.flatten()[p_idx]
	q_dis = q.flatten()[p_idx]

	q_idx = np.argwhere(q_dis.flatten() > 0)

	p_dis0 = p_dis[q_idx]
	q_dis0 = q_dis[q_idx]

	return - (p_dis0 * np.log(q_dis0 / p_dis0)).sum()

def calc_entropy(belief):
	probs = belief[belief.nonzero()]
	return -1.0 * (probs * np.log(probs)).sum()

def reset_envs(init_s, num_envs, envs):
	for env in envs:
		env.s = init_s	

	belief = np.ones(num_envs)
	belief = belief / sum(belief)

	return belief, False, 0, 0 #belief, done_bool, returns, iter_count

class RLagent(object):
	def __init__(self, state_size, dyn, gamma, eps = 1e-3, num_bins = 6, bam_bool = True, save_path = "", load_path = "", save_bool = True, nn_num = 3):

		self.dyn = dyn #list of dynamics models
		self.nD = len(self.dyn)
		self.num_bins = num_bins
		self.eps = eps
		self.state_size = state_size
		self.nn_num = nn_num
		self.bam_bool = bam_bool
		self.save_bool = save_bool

		self.return_state_list()
		self.actions = [0,1,2,3] #list(itertools.product(*[[0,1,2,3], list(range(self.nD))]))
		self.ref_dict = {}

		self.nA = 4
		self.T = 40

		self.nS = 100

		self.nIter = 40

		self.gamma = gamma

		self.save_path = save_path
		self.load_path = load_path

		if load_path != "": # and self.save_bool == False:
			self.load_model()
		# elif load_path != "" and self.save_bool:
		# 	self.load_model()
		# 	self.local_value_iteration()
		else:
			self.local_value_iteration()

	def yx2s(self, yx):
		return yx[1] + yx[0] * self.state_size[1]

	def s2yx(self, s):
		return (np.floor(s / self.state_size[1]), s % self.state_size[1]) 

	def calc_entropy_reward(self, belief, next_belief):
		probs0 = belief[belief.nonzero()]
		probs1 = next_belief[next_belief.nonzero()]
		ent0 = -1.0 * (probs0 * np.log(probs0)).sum()
		ent1 = -1.0 * (probs1 * np.log(probs1)).sum()

		return ent0 - ent1 ### trying to maximize the negative change in entropy

	def return_state_list(self):
		state_iter = [list(range(self.state_size[0])), list(range(self.state_size[1]))]

		for idx in range(self.nD):
			state_iter.append(list(np.linspace(0.0, 1.0, num=self.num_bins)))

		state_list = list(itertools.product(*state_iter))
		self.state_list = []
		ref_list = []

		for idx in range(len(state_list)):
			belief = list(state_list[idx])[2:]
			belief_sum = sum(belief)

			if belief_sum == 1:
				self.state_list.append(state_list[idx])
				if state_list[idx][0] == 0 and state_list[idx][1] == 0:
					ref_list.append(np.expand_dims(np.array(belief), axis = 0))

		self.ref_array = np.concatenate(ref_list, axis = 0)

		self.values = np.zeros(len(self.state_list))

		self.idx_dict = {}

		for idx in range(len(self.state_list)):
			self.idx_dict[self.state_list[idx]] = idx

		print("The discretized state space has ", len(self.state_list), " states")

	def lookup_values(self, point):
		return self.values[self.idx_dict[point]]

	def interpolate(self, state):
		if state in self.ref_dict.keys():
			betas, points, point = self.ref_dict[state]
			if point is not None:
				return self.lookup_values(point)

			values = np.zeros_like(betas)

			for idx, point in enumerate(points):
				values[idx] = self.lookup_values(point)						
		else:
			belief = np.expand_dims(np.array(list(state)[2:]), axis = 0)
			distances = np.linalg.norm(self.ref_array - belief, axis = 1)
			idxs = np.argsort(distances)
			beliefs = self.ref_array[idxs[:(self.nn_num + 1)],:]
			distances.sort()

			if distances[0] == 0:
				point = tuple(list(state)[:2] + list(beliefs[0]))
				self.ref_dict[state] = (None, None, point)
				return self.lookup_values(point)

			betas = distances[:(self.nn_num + 1)] / sum(distances[:(self.nn_num + 1)])

			values = np.zeros_like(betas)
			points = []

			for idx in range(values.size):
				point = tuple(list(state)[:2] + list(beliefs[idx]))
				points.append(point)
				values[idx] = self.lookup_values(point)

			self.ref_dict[state] = (betas, points, None)

		return sum(betas * values)

	def get_rewardandbelief(self, state, action, next_state, belief):
		rewards = -np.zeros_like(belief)
		probs = np.zeros_like(belief)

		for idx, dyn in enumerate(self.dyn):
			for dyn_tuple in dyn[state][action]:
				prob, next_state_t, reward, is_done = dyn_tuple

				if next_state_t == next_state:
					rewards[idx] = reward 
					probs[idx] = prob

		if sum(belief * probs) == 0:
			return rewards, np.ones_like(belief) / sum(np.ones_like(belief))
		else:
			return rewards, (belief * probs) / np.sum((belief * probs))

	def parallel_function(self, action, pose, s, belief, idx_dyn):
		value_est = 0

		for dyn_tuple in self.dyn[idx_dyn][s][action]:
			prob, next_state, reward, is_done = dyn_tuple
			rewards, next_belief = self.get_rewardandbelief(s, action, next_state, belief)
			full_next_state = tuple(list(self.s2yx(next_state)) + list(next_belief))
			next_value = self.interpolate(full_next_state)

			if self.bam_bool:
				reward = (rewards * belief).sum()
			else:
				reward = self.calc_entropy_reward(belief, next_belief)

			value_est += prob * (reward + self.gamma * next_value)

		return value_est

	def max_value(self, state):
		pose = np.array(list(state)[:2])
		s = self.yx2s(pose)
		belief = np.array(list(state)[2:])
		idx_dyn = belief.argmax()
		value_est = np.zeros(len(self.actions))

		value_est = np.array([self.parallel_function(action, pose, s, belief, idx_dyn) for action in self.actions])

		return value_est.max(), abs(value_est.max() - self.interpolate(state)), self.actions[value_est.argmax()]

	def local_value_iteration(self):
		max_error = np.inf
		iter_count = 0
		if self.save_bool:
			self.save_model(iter_count)

		while max_error > self.eps:
			print(iter_count)
			if (iter_count + 1) % 10 == 0:
				print(iter_count + 1, " iterations of value iteration have been performed with max error ", max_error)
				if self.save_bool:
					self.save_model(iter_count)
			max_error = 0
			iter_count += 1

			values_temp = np.zeros(len(self.state_list))

			for idx, state in enumerate(self.state_list):
				if (idx + 1) % 10000 == 0:
					print(idx + 1, " states have been evaluated during this iteration")
				values_temp[idx], error, act = self.max_value(state)

				if error > max_error:
					max_error = error
					# print("Types of values: ", values)
					# print(values.max())
					# print(error)

			self.values = copy.copy(values_temp)

		if self.save_bool:
			self.save_model(iter_count)

		print("Finished local value iteration")

	def policy(self, s, belief):
		pose = list(self.s2yx(s))
		state = tuple(pose + list(belief))
		value, error, action = self.max_value(state)
		return action, value

	def save_model(self, iter_count):
		print("Model saved to: ",self.save_path + "_iteration_" + str(iter_count))
		np.save(self.save_path + "_iteration_" + str(iter_count), self.values)

	def load_model(self):	
		print("Model Loaded from: ", self.load_path)	
		self.values = np.load(self.load_path)
		print(self.values.sum())

class Goal_Reacher(object):

	def __init__(self, dyn_array, goal):

		self.dyn = dyn_array
		self.goal = goal
		self.policy =  {}
		self.policy[goal] = 0

		self.dijkstra(self.goal, first_time = True)

	def get_neighbors(self, state):
		trans_probs = self.dyn[state[0], state[1]].sum(0)
		neighbors = []

		if trans_probs[0] != 0 and state[0] - 1 >= 0:
			neighbors.append((state[0] - 1, state[1]))

		if trans_probs[0] != 1 and state[1] + 1  < self.dyn.shape[1] :
			neighbors.append((state[0], state[1] + 1))

		if trans_probs[0] != 2 and state[0] + 1 < self.dyn.shape[0] :
			neighbors.append((state[0] + 1, state[1]))

		if trans_probs[0] != 3 and state[1] - 1 >= 0:
			neighbors.append((state[0], state[1] - 1))

		return neighbors

	def max_prob_action(self, state, next_state):
		vert = next_state[0] - state[0]
		hor = next_state[1] - state[1]

		if vert == 1:
			idx = 0
		elif vert == -1:
			idx = 2
		elif hor == 1:
			idx = 3
		elif hor == -1:
			idx = 1
		else:
			raise Exception("One of the cases above should be true")

		trans_probs = self.dyn[next_state[0], next_state[1], :, idx]

		action = np.argmax(trans_probs)
		prob = np.max(trans_probs)

		return prob, action

	def dijkstra(self, goal, first_time = False):

		if goal != self.goal or first_time:
			self.goal = goal
			open_set = [goal]
			prob_dict = {}
			prob_dict[goal] = 1

			while len(open_set) != 0:
				state = open_set[0]
				state_prob = prob_dict[state]
				open_set.pop(0)

				neighbors = self.get_neighbors(state)

				for neighbor in neighbors:


					prob, action = self.max_prob_action(state, neighbor)

					if neighbor in prob_dict.keys():
						if prob_dict[neighbor] < prob * state_prob:
							prob_dict[neighbor] = prob * state_prob
							self.policy[neighbor] = action
							open_set.append(neighbor)
					else:
						prob_dict[neighbor] = prob * state_prob
						self.policy[neighbor] = action
						open_set.append(neighbor)

	def optimal_action(self, state, goal):
		self.dijkstra(goal)
		return self.policy[state]

if __name__ == '__main__':
	with open("framework_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	debugging_val = cfg['debugging_params']['debugging_val']
	save_model_val = cfg['debugging_params']['save_model_val']

	num_envs = cfg['evaluation_params']['num_envs']
	nn_num = cfg['evaluation_params']['nn_num']
	seed = cfg['evaluation_params']['seed']
	num_trials = cfg['evaluation_params']['num_trials']
	gamma = cfg['evaluation_params']['gamma']
	shape =cfg['evaluation_params']['shape']
	num_bins = cfg['evaluation_params']['num_bins']
	eps = cfg['evaluation_params']['eps']
	max_radius = cfg['evaluation_params']['max_radius']
	large_value = cfg['evaluation_params']['large_value']
	prob_goal = cfg['evaluation_params']['prob_goal']
	prob_not_goal = cfg['evaluation_params']['prob_not_goal']
	load_bamdp_path = cfg['evaluation_params']['load_bamdp_path']
	load_ent_path = cfg['evaluation_params']['load_ent_path']

    ##################################################################################
    ### Setting Debugging Flag and Save Model Flag
    ##################################################################################
	if debugging_val == 1.0:
		debugging_flag = True
		var = input("Debugging flag activated. No Results will be saved. Continue with debugging [y,n]: ")
		if var != "y":
			debugging_flag = False
	else:
		debugging_flag = False

	if debugging_flag:
		print("Currently Debugging")
	else:
		print("Training with debugged code")

	if save_model_val == 1.0:
		save_model_flag = True
	else:
		var = input("Save Model flag deactivated. No models will be saved. Are you sure[y,n]: ")
		if var == "y":
			save_model_flag = False
		else:
			save_model_flag = True
    ###############################################################################################
    ### Setting Up Logging
    ###############################################################################################
	logger = Logger(cfg, debugging_flag, save_model_flag)
	save_path = logger.models_folder
		
	logging_dict = {}
	logging_dict['scalar'] = {}
	logging_dict['image'] = {}
	################################################################################################
	### Setting random seed and initializing environments
	################################################################################################
	np.random.seed(seed)

	envs = []
	envs_P = []

	full_goal_list = [(int(0.5 * shape[0]),int(0.8 * shape[1])),(int(0.5 * shape[0]),int(0.2 * shape[1])),\
	(int(0.7 * shape[0]),int(0.4 * shape[1])), (int(0.3 * shape[0]),int(0.6 * shape[1])),\
	(int(0.8 * shape[0]),int(0.8 * shape[1])), (int(0.75 * shape[0]),int(0.4 * shape[1])),\
	(int(0.35 * shape[0]),int(0.9 * shape[1])), (int(0.21 * shape[0]),int(0.71 * shape[1])),\
	(int(0.71 * shape[0]),int(0.21 * shape[1])), (int(0.6 * shape[0]),int(0.6 * shape[1]))]

	goal_list = full_goal_list[:num_envs]

	for idx in range(num_envs):
		envs.append(GridworldEnv(shape=shape, goal_list = goal_list, goal_idx = idx, max_radius = max_radius,\
		 large_value = large_value, prob_goal = prob_goal, prob_not_goal = prob_not_goal))
		envs_P.append(envs[-1].P)


	kl_div = np.zeros(num_envs)


	for idx in range(num_envs):
		kl_div[idx] = KL_div(envs[0].P_vals, envs[idx].P_vals)
		logging_dict['scalar']['kl'] = kl_div[idx] / envs[0].nS
		logger.save_scalars(logging_dict, idx, 'eval/')

	print("\nKL Divergence between chosen dynamics and other dynamics: ", kl_div / envs[0].nS, "\n")
	###########################################################################################################
	### Training models or loading pretrained models
	##################################################################################################
	print("Calculating Local Value Function for BAMDP")
	bamdp = RLagent(shape, envs_P, gamma, eps = eps, num_bins = num_bins, bam_bool = True, save_path = save_path + "bamdp" + "_numenvs_" + str(num_envs)\
	 + "_large_value_" + str(large_value) + "_shape_" + str(shape[0]) + "_" + str(shape[1]) + "_nnnum_" + str(nn_num) + "_numbins_" + str(num_bins),\
	  load_path = load_bamdp_path, save_bool = save_model_flag, nn_num = nn_num)
	print("Calculating Local Value Function for Entropy Minimization")
	entmdp = RLagent(shape, envs_P, gamma, eps = eps, num_bins = num_bins, bam_bool = False, save_path = save_path + "entmdp" + "_numenvs_" + str(num_envs)\
	 + "_large_value_" + str(large_value) + "_shape_" + str(shape[0]) + "_" + str(shape[1]) + "_nnnum_" + str(nn_num) + "_numbins_" + str(num_bins),\
	  load_path = load_ent_path, save_bool = save_model_flag, nn_num = nn_num)
	greedy_policies = []
	for idx in range(num_envs):
		greedy_policies.append(Goal_Reacher(envs[idx].P_vals, envs[idx].goal))
	###############################################################################
	### Evaluating Trained Models
	##############################################################################
	bamdp_vis_list = []
	entmin_vis_list = []
	greedy_vis_list = []

	envs_idx = np.random.choice(num_envs)
	correct_env = envs[envs_idx]
	init_s = correct_env.s

	random_trial = 10 #np.random.choice(num_trials) + 1
	correct_bamdp = []
	iter_bamdp = []

	correct_entmin = []
	iter_entmin = []

	correct_greedy = []
	iter_greedy = []	

	action_match = []

	for idx_trial in range(1, num_trials + 1):
		if idx_trial % 10 == 0:
			print("Completed: ", idx_trial, " trials of ", num_trials)

		envs_idx = np.random.choice(num_envs)

		correct_env = envs[envs_idx]
		goal = envs[envs_idx].goal
		init_s = envs[envs_idx].yx2s((np.random.choice(shape[0]), np.random.choice(shape[1])))

		belief, done_bool, returns, iter_count = reset_envs(init_s, num_envs, envs)
		belief_count = 0
		action_count = 0

		while not done_bool:
			# if (iter_count + 1) % 10 == 0:
			# 	print("Number of iterations of action: ", iter_count + 1)
			# 	print("Results of Estimation:  ", np.round(100 * belief))
			idx_dyn_g = belief.argmax()
			goal_g = envs[idx_dyn_g].goal

			state_g = correct_env.s

			action_g = greedy_policies[idx_dyn_g].optimal_action(correct_env.s2yx(state_g), goal_g)

			state = correct_env.s
			bamdp_vis_list.append(correct_env.s2yx(state))

			action, value = bamdp.policy(state, belief)

			if action == action_g:
				action_count += 1

			next_state, r, d, p = correct_env.step(action)

			rew_temp, belief = bamdp.get_rewardandbelief(state, action, next_state, belief)
			
			returns += r
			
			if d == True: done_bool = True

			iter_count += 1

			if belief.argmax() == envs_idx:
				belief_count += 1

		if envs_idx == belief.argmax():
			correct_test = 1
		else:
			correct_test = 0

		# print(envs_idx)
		# print(returns)
		# print(belief_count / iter_count)

		correct_bamdp.append(belief_count / iter_count)
		iter_bamdp.append(iter_count)
		action_match.append(action_count / iter_count)
		
		logging_dict['scalar']['bamdp/iter_count'] = iter_count
		logging_dict['scalar']['bamdp/correct_fraction'] = belief_count / iter_count

		# print("\n#################################")
		# print("BAMDP Results")
		# print("Results of Estimation:  ", np.round(100 * belief))
		# print("Correct Dynamics are: ", envs_idx)
		# print("Estimated Dynamics are: ", belief.argmax())
		# print("Number of Iterations till Convergence: ", iter_count)
		# print("Returns for run: ", returns)
		# print("##################################")

		belief, done_bool, returns, iter_count = reset_envs(init_s, num_envs, envs)
		belief_count = 0

		while(belief.max() < 0.99) and not done_bool:
			# print(iter_count)
			# if (iter_count + 1) % 10 == 0:
				# print("Number of iterations of action: ", iter_count + 1)
				# print("Results of Estimation:  ", np.round(100 * belief))

			state = correct_env.s
			entmin_vis_list.append(correct_env.s2yx(state))

			action, value = entmdp.policy(state, belief)

			# print("State: ", entmdp.s2yx(state))
			# print("Action: ", action)

			next_state, r, d, p = correct_env.step(action)

			# print("Next State: ", entmdp.s2yx(next_state))

			rew_temp, belief = entmdp.get_rewardandbelief(state, action, next_state, belief)
			
			# print("Belief: ", belief)
			returns += r
			
			if d == True: done_bool = True

			iter_count += 1

			if belief.argmax() == envs_idx:
				belief_count += 1

		while not done_bool and returns > -1000000:
			idx_dyn = belief.argmax()
			goal = envs[idx_dyn].goal

			state = correct_env.s
			entmin_vis_list.append(correct_env.s2yx(state))

			action = greedy_policies[idx_dyn].optimal_action(correct_env.s2yx(state), goal)

			next_state, r, d, p = correct_env.step(action)

			rew_temp, belief = entmdp.get_rewardandbelief(state, action, next_state, belief)

			returns += r

			if d == True: done_bool = True

			iter_count += 1

			if belief.argmax() == envs_idx:
				belief_count += 1


		correct_entmin.append(belief_count / iter_count)
		iter_entmin.append(iter_count)

		logging_dict['scalar']['ent_min/iter_count'] = iter_count
		logging_dict['scalar']['ent_min/correct_fraction'] = belief_count / iter_count

		# print("\n#######################################")
		# print("Entropy Minimization Results")
		# print("Results of Estimation:  ", np.round(100 * belief))
		# print("Correct Dynamics are: ", envs_idx)
		# print("Estimated Dynamics are: ", belief.argmax())
		# print("Number of Iterations till Convergence: ", iter_count)
		# print("Returns for run: ", returns)
		# print("#########################################")

		belief, done_bool, returns, iter_count = reset_envs(init_s, num_envs, envs)
		belief_count = 0
		action_count = 0

		while not done_bool and returns > -1000000:
			idx_dyn = belief.argmax()
			goal = envs[idx_dyn].goal

			state = correct_env.s
			greedy_vis_list.append(correct_env.s2yx(state))

			action = greedy_policies[idx_dyn].optimal_action(correct_env.s2yx(state), goal)
			# action_bamdp, value_bamdp = bamdp.policy(state, belief)


			next_state, r, d, p = correct_env.step(action)

			rew_temp, belief = bamdp.get_rewardandbelief(state, action, next_state, belief)

			returns += r

			if d == True: done_bool = True

			iter_count += 1

			if belief.argmax() == envs_idx:
				belief_count += 1

		correct_greedy.append(belief_count / iter_count)
		iter_greedy.append(iter_count)


		logging_dict['scalar']['greedy/iter_count'] = iter_count
		logging_dict['scalar']['greedy/correct_fraction'] = belief_count / iter_count

		# print("\n#######################################")
		# print("Greedy Approach Results")
		# print("Results of Estimation:  ", np.round(100 * belief))
		# print("Correct Dynamics are: ", envs_idx)
		# print("Estimated Dynamics are: ", belief.argmax())
		# print("Number of Iterations till Convergence: ", iter_count)
		# print("Returns for run: ", returns)
		# print("#########################################")


		logger.save_scalars(logging_dict, idx_trial, 'eval/')

	bamdp_visitation = np.zeros(shape)
	entmin_visitation = np.zeros(shape)
	greedy_visitation = np.zeros(shape)

	for visit in bamdp_vis_list:
		bamdp_visitation[int(visit[0]), int(visit[1])] += 1

	for visit in entmin_vis_list:
		entmin_visitation[int(visit[0]), int(visit[1])] += 1

	for visit in greedy_vis_list:
		greedy_visitation[int(visit[0]), int(visit[1])] += 1

	bamdp_visitation = bamdp_visitation / bamdp_visitation.max()
	entmin_visitation = entmin_visitation / entmin_visitation.max()
	greedy_visitation = greedy_visitation / greedy_visitation.max()

	for goal in goal_list:
		bamdp_visitation[int(goal[0]), int(goal[1])] = 1
		entmin_visitation[int(goal[0]), int(goal[1])] = 1
		greedy_visitation[int(goal[0]), int(goal[1])] = 1


	logging_dict['image']['bamdp'] = [bamdp_visitation, np.where(bamdp_visitation != 0, np.ones_like(bamdp_visitation), np.zeros_like(bamdp_visitation))]
	logging_dict['image']['entmin'] = [entmin_visitation, np.where(entmin_visitation != 0, np.ones_like(entmin_visitation), np.zeros_like(entmin_visitation))]
	logging_dict['image']['greedy'] = [greedy_visitation, np.where(greedy_visitation != 0, np.ones_like(greedy_visitation), np.zeros_like(greedy_visitation))]
	logging_dict['image']['greedy0'] = [greedy_visitation]

	logger.save_npimages2D(logging_dict, 0, 'eval/')

	print("bamdp - Average Number of Iterations to Goal: ", sum(iter_bamdp) / len(iter_bamdp))
	print("bamdp - Average Correct Fraction: ", sum(correct_bamdp) / len(correct_bamdp))
	print("entmin - Average Number of Iterations to Goal: ", sum(iter_entmin) / len(iter_entmin))
	print("entmin - Average Correct Fraction: ", sum(correct_entmin) / len(correct_entmin))
	print("greedy - Average Number of Iterations to Goal: ", sum(iter_greedy) / len(iter_greedy))
	print("greedy - Average Correct Fraction: ", sum(correct_greedy) / len(correct_greedy))
	print("action - Average Correct Fraction: ", sum(action_match) / len(action_match))