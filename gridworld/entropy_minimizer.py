import io
import numpy as np
import sys
import copy
from gym.envs.toy_text import discrete
from scipy.special import softmax
from gridworld import GridworldEnv

class Entropy_Minimizer(object):

	def __init__(self, dyn_1, dyn_2, dyn_3):

		self.dyn = []
		self.dyn.append(dyn_1)
		self.dyn.append(dyn_2)
		self.dyn.append(dyn_3)

		self.nD = len(self.dyn)
		self.nA = 4
		self.T = 40

		self.nS = 100

		self.nIter = 40

	def optimize(self, state):

		self.min_entropy = np.inf
		# print("Initial Minimum Entropy Estimate: ", self.min_entropy)

		for idx_trial in range(self.nIter):
			action_sequences = self.generate_action_sequence()
			entropy_array = self.min_entropy * np.ones(self.nS)

			for idx_sample in range(self.nS):

				action_sequence = action_sequences[idx_sample]

				prob_tau = self.calc_prob_tau(state, action_sequence)

				prob_dyn_tau = self.calc_prob_dyn(prob_tau)

				entropy_array[idx_sample] = self.calc_entropy(prob_dyn_tau)

				if entropy_array[idx_sample] < self.min_entropy:
					self.min_entropy = entropy_array[idx_sample]# / prob_dyn_tau[dyn_est]
					best_probs = prob_dyn_tau
					best_action_sequence = action_sequence

		# print("Final Minimum Entropy Estimate: ", self.min_entropy)

		# print("Estimated Probabilities of Action Sequence: ", best_probs)

		return best_action_sequence


	def generate_action_sequence(self):

		action_sequences = np.zeros((self.nS, self.T))

		for idx in range(self.nS):
			for idx_traj in range(self.T):
				action_sequences[idx, idx_traj] = np.random.choice([0,1,2,3])

		return action_sequences

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

		# a = input("")
		log_prob_tau_dyn = np.zeros(prob_tau.shape[0])
		prob_dyn = np.array([1/self.nD])

		for idx in range(prob_tau.shape[0]):

			######################
			# Add code incase transition probability is zero for a specific action
			######################
			# if np.count_nonzero(prob_tau[idx]):
			# 	print("This happened")
			# 	log_prob_tau_dyn[idx] = -np.inf



			log_prob_tau_dyn[idx] =  np.log(prob_tau[idx]).sum()

		# print(np.exp(log_prob_tau_dyn))
		# print(np.exp(log_prob_tau_dyn).sum())
		# a = input("")

		log_prob_dyn_tau = log_prob_tau_dyn + np.log(1/self.nD) - np.log(np.exp(log_prob_tau_dyn).sum())

		# print(np.exp(log_prob_dyn_tau))
		# a = input("")

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

	def calc_prod_dyn_from_trajectory(self, trajectory):

		prob_tau = np.zeros((self.nD, len(trajectory)))

		for idx_dyn in range(self.nD):
			for idx_traj in range(len(trajectory)):
				prob_tau[idx_dyn, idx_traj] = self.trans_prob(trajectory[idx_traj], self.dyn[idx_dyn])

		return self.calc_prob_dyn(prob_tau)

if __name__ == '__main__':

    env_1 = GridworldEnv()
    env_2 = GridworldEnv()
    env_3 = GridworldEnv()
    env_4 = GridworldEnv()
    env_5 = GridworldEnv()

    dyn_count = np.zeros(3)

    for idx in range(30):
	    optimizer = Entropy_Minimizer(env_4.P, env_2.P, env_5.P)

	    action_sequence = optimizer.optimize(env_1.s)

	    trajectory = []

	    for idx_act in range(action_sequence.shape[0]):

	    	state = env_2.s
	    	action = action_sequence[idx_act]
	    	next_state, r, d, p = env_2.step(action)
	    	trajectory.append((next_state, action, state))

	    prob_dyn = optimizer.calc_prod_dyn_from_trajectory(trajectory)

	    dyn_count[prob_dyn.argmax()] += 1


    print("Estimate of Correct Dynamics Model: ")
    print(dyn_count.argmax())
    print("Probabilities of Dynamics Model: ")
    print(np.round(100 *dyn_count/dyn_count.sum()))
    print("")