def calc_entropy(beliefs, rews, gamma):
	probs = beliefs[:, -1]
	probs = probs[probs.nonzero()]
	return -1.0 * (probs * np.log(probs)).sum()

def calc_bamdp_exp_returns(beliefs, rews, gamma):
	gamma_series = np.power(gamma * np.ones_like(beliefs), np.tile(np.expand_dims(np.arange(beliefs.shape[1]), axis = 0), (beliefs.shape[0], 1)))
	return (beliefs * rews * gamma_series).sum()

class Planner(object):

	def __init__(self, state_size, dyn, objective_function, gamma = 1, sample_method = 'max', eps = 1e-3, num_bins = 6):

		self.dyn = dyn #list of dynamics models
		self.nD = len(self.dyn)
		self.num_bins = num_bins
		self.eps = eps
		self.state_size = state_size

		self.state_list = self.return_state_list()
		self.actions = [0,1,2,3]
		self.ref_dict = {}

		self.nA = 4
		self.T = 40
		self.obj_funct = objective_function

		self.nS = 100

		self.nIter = 40

		self.sample_method = sample_method
		self.gamma = gamma

	def minimize(self, state, trajectory):

		self.min_score = np.inf

		#### calculating sequence values from past experience
		if len(trajectory) != 0:
			probs_old, rews_old = self.calc_from_past(trajectory)

		for idx_trial in range(self.nIter):
			action_sequences = self.generate_action_sequence()
			scores = np.ones(self.nS)

			#### finding the objective function value for each sample
			for idx_sample in range(self.nS):
				action_sequence = action_sequences[idx_sample]

				if len(trajectory) == 0:
					probs, rews = self.calc_from_samples(state, action_sequence, mode = self.sample_method)

				else: 
					probs, rews = self.calc_from_samples(state, action_sequence, mode = self.sample_method)
					probs = np.concatenate([probs_old, probs], axis = 1)
					rews = np.concatenate([rews_old, rews], axis = 1)

				beliefs = self.calc_beliefs(probs)

				scores[idx_sample] = self.obj_funct(beliefs, rews, self.gamma)

				if scores[idx_sample] < self.min_score:
					self.min_score = scores[idx_sample]#
					best_belief = exp_belief
					best_action_sequence = action_sequence

		print("Min Score: ",self.min_score)
		# returning first action in best action sequence
		return best_action_sequence[0]

	def generate_action_sequence(self):

		return np.random.choice(self.actions, size=(self.nS, self.T))

	def calc_from_samples(self, state, action_sequence, mode = 'max'):

		probs = np.zeros((self.nD, action_sequence.shape[0]))
		rews = np.zeros((self.nD, action_sequence.shape[0]))

		for idx_dyn in range(self.nD):
			temp_state = copy.copy(state)

			for idx_act in range(action_sequence.shape[0]):
				temp_state, probs[idx_dyn, idx_act], rews[idx_dyn, idx_act] = self.next_state(temp_state, action_sequence[idx_act], self.dyn[idx_dyn], mode = mode) 

		return probs, rews

	def calc_from_past(self, trajectory):

		probs = np.zeros((self.nD, len(trajectory))) # probability of each transition
		rews = np.zeros((self.nD, len(trajectory))) # reward from each transition

		for idx_dyn in range(self.nD):
			for idx_traj in range(len(trajectory)):
				probs[idx_dyn, idx_traj], rews[idx_dyn, idx_traj] = self.trans_prob(trajectory[idx_traj], self.dyn[idx_dyn])

		return (probs, rews)

	def calc_belief_from_past(self, trajectory):
		probs, rews = self.calc_from_past(trajectory)
		beliefs = self.calc_beliefs(probs)
		return beliefs[:, -1]

	def next_state(self, state, action, dyn, mode = 'max'):

		if mode == 'max':
			max_prob = 0
			max_next_state = None
			max_reward = None

			for dyn_tuple in dyn[state][action]:

				prob, next_state, reward, is_done = dyn_tuple

				if prob > max_prob:
					max_next_state = next_state
					max_reward = reward 
					max_prob = prob

			return max_next_state, max_reward, max_prob

		elif mode == 'sample':
			prob_list = []
			values_list = []

			for dyn_tuple in dyn[state][action]:

				prob, next_state, reward, is_done = dyn_tuple

				prob_list.append(prob)
				values_list.append((next_state, reward, prob))

			return np.random.choice(values_list, p = prob_list)

	def trans_prob(self, step, dyn):

		next_state, action, state = step

		for dyn_tuple in dyn[state][action]:
			prob, next_state_pred, reward, is_done = dyn_tuple

			if next_state_pred == next_state:
				return (prob, reward)

		return (0, 0)

	# calculates the probability of a dynamics model given a trajectory
	def calc_beliefs(self, probs): 

		#### prob tau is the probability of each state transition for each dynamics model
		# row 0 = (prob s1 | s0, a0 - prob s2 | s1, a1 - ... - prob sN | sN-1, aN-1) for dynamic model 0
		# row 1 = (prob s1 | s0, a0 - prob s2 | s1, a1 - ... - prob sN | sN-1, aN-1) for dynamic model 0
		cum_log_probs = np.zeros_like(probs)
		beliefs = np.zeros_like(probs)

		# code to handle case where the probability of a transition in the trajectory is zero for at least one of the dynamics models
		options = np.ones(probs.shape[0])

		for idx in range(probs.shape[0]):
			if np.where(new_probs[idx] > 0, np.ones_like(new_probs[idx]), np.zeros_like(new_probs[idx])).prod() == 0:
				options[idx] = 0
			else:
				cum_log_probs[idx] = np.cumsum(np.log(probs[idx]))

		if options.sum() == 0:
			return None

		logbelief_prior = np.log(1 / options.sum())

		for idx in range(probs.shape[0]):
			if options[idx] == 1:
				beliefs[idx] = np.exp(logbelief_prior + cum_log_probs[idx])

		beliefs = beliefs / beliefs.sum(0) #normalizing beliefs

		return  beliefs