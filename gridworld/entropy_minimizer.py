import io
import numpy as np
import sys
import copy
from gym.envs.toy_text import discrete
from scipy.special import softmax
from gridworld import GridworldEnv

def KL_div(p, q):

	p_idx = np.argwhere(p.flatten() > 0)

	p_dis = p.flatten()[p_idx]
	q_dis = q.flatten()[p_idx]

	q_idx = np.argwhere(q_dis.flatten() > 0)

	p_dis0 = p_dis[q_idx]
	q_dis0 = q_dis[q_idx]

	return - (p_dis0 * np.log(q_dis0 / p_dis0)).sum()

class Entropy_Minimizer(object):

	def __init__(self, dyn):

		self.dyn = dyn
		self.nD = len(self.dyn)
		self.nA = 4
		self.T = 40

		self.nS = 100

		self.nIter = 40

	def optimize(self, state, trajectory):

		self.min_entropy = np.inf

		if len(trajectory) != 0:
			prob_tau_old = self.calc_prob_tau_from_trajectory(trajectory)

		for idx_trial in range(self.nIter):
			action_sequences = self.generate_action_sequence()
			entropy_array = np.ones(self.nS)

			for idx_sample in range(self.nS):

				action_sequence = action_sequences[idx_sample]

				if len(trajectory) == 0:
					prob_tau = self.calc_prob_tau(state, action_sequence)

				else: 
					prob_tau_new = self.calc_prob_tau(state, action_sequence)
					prob_tau = np.concatenate((prob_tau_old, prob_tau_new), axis = 1)

				prob_dyn_tau = self.calc_prob_dyn(prob_tau)

				entropy_array[idx_sample] = self.calc_entropy(prob_dyn_tau)

				if entropy_array[idx_sample] < self.min_entropy:
					self.min_entropy = entropy_array[idx_sample]# / prob_dyn_tau[dyn_est]
					best_probs = prob_dyn_tau
					best_action_sequence = action_sequence

		return best_action_sequence

	def generate_action_sequence(self):

		return np.random.choice([0,1,2,3], size=(self.nS, self.T))

	def calc_prob_tau(self, state, action_sequence):

		prob_tau = np.zeros((self.nD, action_sequence.shape[0]))

		for idx_dyn in range(self.nD):
			temp_state = copy.copy(state)

			for idx_act in range(action_sequence.shape[0]):
				temp_state, prob_tau[idx_dyn, idx_act] = self.next_state(temp_state, action_sequence[idx_act], self.dyn[idx_dyn]) 

		return prob_tau

	def calc_entropy(self, probs):

		return -1.0 * (probs * np.log(probs)).sum()

	def calc_prob_dyn(self, prob_tau): # calculates the probability of a dynamics model given a trajectory

		#### prob tau is the probability of each state transition for each dynamics model
		# row 0 = (prob s1 | s0, a0 - prob s2 | s1, a1 - ... - prob sN | sN-1, aN-1) for dynamic model 0
		# row 1 = (prob s1 | s0, a0 - prob s2 | s1, a1 - ... - prob sN | sN-1, aN-1) for dynamic model 0

		# print(prob_tau)

		log_prob_tau_dyn = np.zeros(prob_tau.shape[0])
		prob_dyn = np.array([1/self.nD])

		for idx in range(prob_tau.shape[0]):

			######################
			# Add code incase transition probability is zero for a specific action
			######################
			log_prob_tau_dyn[idx] =  np.log(prob_tau[idx]).sum()

		log_prob_dyn_tau = log_prob_tau_dyn - np.log(np.exp(log_prob_tau_dyn).sum())


		return np.exp(log_prob_dyn_tau)

	def next_state(self, state, action, dyn):
		max_prob = 0
		max_next_state = -1

		for dyn_tuple in dyn[state][action]:

			prob, next_state, reward, is_done = dyn_tuple

			if prob > max_prob:
				max_next_state = next_state 
				max_prob = prob

		return next_state, max_prob

	def trans_prob(self, step, dyn):

		next_state, action, state = step


		for dyn_tuple in dyn[state][action]:

			prob, next_state_pred, reward, is_done = dyn_tuple

			if next_state_pred == next_state:
				return prob
		return 0

	def calc_prob_tau_from_trajectory(self, trajectory):

		prob_tau = np.zeros((self.nD, len(trajectory)))

		for idx_dyn in range(self.nD):
			for idx_traj in range(len(trajectory)):
				prob_tau[idx_dyn, idx_traj] = self.trans_prob(trajectory[idx_traj], self.dyn[idx_dyn])

		return prob_tau

	def calc_prod_dyn_from_trajectory(self, trajectory):

		prob_tau = self.calc_prob_tau_from_trajectory(trajectory)

		return self.calc_prob_dyn(prob_tau)

if __name__ == '__main__':

	num_envs = 3
	envs = []
	envs_P = []

	for idx in range(num_envs):
		envs.append(GridworldEnv())
		envs_P.append(envs[-1].P)

	envs_idx = 2
	
	optimizer = Entropy_Minimizer(envs_P)
	
	prob_dyn = np.zeros(num_envs)
	kl_div = np.zeros(num_envs)

	for idx in range(num_envs):
		kl_div[idx] = KL_div(envs[envs_idx].P_vals, envs[idx].P_vals)

	iter_count = 0
	
	trajectory = []

	while(prob_dyn.max() < 0.95):

		if (iter_count + 1) % 10 == 0:
			print("Number of iterations of action: ", iter_count + 1)
			print("Results of Estimation:  ", np.round(100 * prob_dyn))

		state = envs[envs_idx].s

		action_sequence = optimizer.optimize(state, trajectory)

		action = action_sequence[0]

		next_state, r, d, p = envs[envs_idx].step(action)

		trajectory.append((next_state, action, state))

		prob_dyn = optimizer.calc_prod_dyn_from_trajectory(trajectory)

		iter_count += 1

	print("KL Divergence between chosen dynamics and other dynamics: ", kl_div)
	print("Results of Estimation:  ", np.round(100 * prob_dyn))
	print("Correct Dynamics are: ", envs_idx)
	print("Estimated Dynamics are: ", prob_dyn.argmax())
	print("Number of Iterations till Convergence: ", iter_count)


