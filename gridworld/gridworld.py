"""
This code was copied from the reposity of dennybritz/reinforcement-learning and
modified for use and experiments in this project
"""
import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) #number of states
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P_vals = np.zeros((shape[0], shape[1], nA, nA + 1))

        for idx_row in range(shape[0]):
            for idx_col in range(shape[1]):
                for idx_hrow in range(nA):

                    # UP = 0 up one row
                    # RIGHT = 1 up one col
                    # DOWN = 2 down one row
                    # LEFT = 3 down one col
                    P_vals[idx_row, idx_col, idx_hrow] = np.random.uniform(0.0, 1.0, nA + 1)

                    large_value = 18

                    if idx_col == 0 and idx_row == 0:
                        P_vals[idx_row, idx_col, idx_hrow, 0] = 0
                        P_vals[idx_row, idx_col, idx_hrow, 3] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value / 2
  

                    elif idx_col == 0 and idx_row == MAX_Y:
                        P_vals[idx_row, idx_col, idx_hrow, 2] = 0
                        P_vals[idx_row, idx_col, idx_hrow, 3] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value / 2

                    elif idx_col == MAX_X and idx_row == 0:
                        P_vals[idx_row, idx_col, idx_hrow, 0] = 0
                        P_vals[idx_row, idx_col, idx_hrow, 1] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value / 2

                    elif idx_col == MAX_X and idx_row == MAX_Y:
                        P_vals[idx_row, idx_col, idx_hrow, 1] = 0
                        P_vals[idx_row, idx_col, idx_hrow, 2] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value / 2

                    elif idx_col == 0:
                        P_vals[idx_row, idx_col, idx_hrow, 3] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value * 1.5 / 2

                    elif idx_col == MAX_X:
                        P_vals[idx_row, idx_col, idx_hrow, 1] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value * 1.5 / 2

                    elif idx_row == 0:
                        P_vals[idx_row, idx_col, idx_hrow, 0] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value * 1.5 / 2

                    elif idx_row == MAX_Y:
                        P_vals[idx_row, idx_col, idx_hrow, 2] = 0                                               
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value * 1.5 / 2

                    else:
                        P_vals[idx_row, idx_col, idx_hrow, idx_hrow] = large_value
                    
                    P_vals[idx_row, idx_col, idx_hrow] =   P_vals[idx_row, idx_col, idx_hrow] / P_vals[idx_row, idx_col, idx_hrow].sum()                 

        P = {}
        self.P_vals = P_vals
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            ns_up = s if y == 0 else s - MAX_X
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1

            s_trans = [ns_up, ns_right, ns_down, ns_left, s]

            for idx_act in range(nA):
                for idx_dyn in range(nA + 1):
                    if P_vals[y, x, idx_act, idx_dyn] != 0:
                        P[s][idx_act].append((P_vals[y, x, idx_act, idx_dyn], s_trans[idx_dyn], reward, is_done(s_trans[idx_dyn])))
                # P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                # P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                # P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                # P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " o "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

if __name__ == '__main__':

    env = GridworldEnv()

    action = 0

    print("##################")
    print("Action Instructions")
    print(" ")
    print("UP      =  0")
    print("RIGHT   =  1")
    print("DOWN    =  2")
    print("LEFT    =  3")
    print("##################")

    env._render()

    while action != -1:

        action = int(input("Please input an action from the list [0,1,2,3]: "))

        print(env.P[env.s][action])
        s, r, d, p = env.step(action)
        env._render()