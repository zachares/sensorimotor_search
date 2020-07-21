import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import copy
import os
import matplotlib.pyplot as plt
import random
import itertools

sys.path.insert(0, "../robosuite/")

from datacollection_util import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

	env = robosuite.make("PandaPegInsertion", 
	has_renderer= True, ignore_done=True,\
	use_camera_obs= False, 
	has_offscreen_renderer=False, 
	gripper_visualization=True, 
	control_freq=10,\
	gripper_type ="SquarePegwForce", 
	controller='position', 

	 )

	env.viewer.set_camera(camera_id=2)
	env.reset()

	file_name = "data/insertion0_noise0.4_0002.h5"

	file = h5py.File(file_name, 'r')

	actions = file['action'][:]

	for i in range(actions.shape[0]):
		env.step(actions[i])
		env.render()


