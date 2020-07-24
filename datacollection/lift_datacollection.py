import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools

from project_utils import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import PandaLiftWrapper

def sample_waypoints(env):
	# waypoints_list = [\
	# np.array([0,0,0.1,0.02]),
	# np.array([0,0,0,0.02]),
	# np.array([0,0,0,-0.002]),
	# np.array([0,0,0.1,-0.002]),
	# np.array([0 ,0.1,0.1,-0.002])\
	# ]
	cube_size = env.robo_env.cube_size
	gripper_open = -1
	gripper_closed = 1

	num_random_points = 10

	waypoints_list = [np.array([0,0,0.04, gripper_open])]

	def random_point(cube_size, gripper_open, gripper_closed):
		eef_noise = np.random.uniform(low=-0.04, high =0.04, size = 3)
		cube_bool = random.choice([0,1])
		eef_noise[2] = abs(eef_noise[2]) + cube_bool * cube_size

		gripper_noise = np.random.uniform(low = gripper_open, high = gripper_closed, size =1)

		return np.concatenate([eef_noise, gripper_noise])

	### exploration point
	for i in range(num_random_points):
		waypoints_list.append(random_point(cube_size, gripper_open, gripper_closed))

	### pregrasp point
	eef_noise = np.random.uniform(low=-0.04, high =0.04, size = 3)
	eef_noise[2] = abs(eef_noise[2]) + cube_size

	gripper_noise = np.random.uniform(low =gripper_open, high = gripper_open / 2, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### grasp point - open
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = eef_noise[2]

	gripper_noise = np.random.uniform(low =gripper_open, high = gripper_open / 2, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### grasp point - closed
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = eef_noise[2]

	gripper_noise = np.random.uniform(low =gripper_closed/2, high = gripper_closed, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### lifting point
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = abs(eef_noise[2])

	gripper_noise = np.random.uniform(low =gripper_closed/2, high = gripper_closed, size =1)

	### exploration point
	for i in range(num_random_points):
		waypoints_list.append(random_point(cube_size, gripper_open, gripper_closed))

	### pregrasp point
	eef_noise = np.random.uniform(low=-0.04, high =0.04, size = 3)
	eef_noise[2] = abs(eef_noise[2]) + cube_size

	gripper_noise = np.random.uniform(low =gripper_open, high = gripper_open / 2, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### grasp point - open
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = eef_noise[2]

	gripper_noise = np.random.uniform(low =gripper_open, high = gripper_open / 2, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### grasp point - closed
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = eef_noise[2]

	gripper_noise = np.random.uniform(low =gripper_closed/2, high = gripper_closed, size =1)

	waypoints_list.append(np.concatenate([eef_noise, gripper_noise]))

	### lifting point
	eef_noise = np.random.uniform(low=-0.01, high =0.01, size = 3)
	eef_noise[2] = abs(eef_noise[2])

	gripper_noise = np.random.uniform(low =gripper_closed/2, high = gripper_closed, size =1)

	return waypoints_list
			
if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("liftdatacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	seed = cfg['control_params']['seed']
	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["control_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']

	logging_folder = cfg["logging_params"]["logging_folder"]
	name = "datacollection_params"

	print("Saving ", name, " to: ", logging_folder + name + ".yml")

	with open(logging_folder + name + ".yml", 'w') as ymlfile2:
		yaml.dump(cfg, ymlfile2)

	##########################################################
	### Setting up hardware for loading models
	###########################################################
	random.seed(seed)
	np.random.seed(seed)
	
	##############################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	robo_env = robosuite.make("PandaLift",\
     has_renderer= display_bool,\
     ignore_done=True,
     use_camera_obs= not display_bool and collect_vision_bool,\
      has_offscreen_renderer = not display_bool and collect_vision_bool,\
      gripper_visualization=False, control_freq=ctrl_freq,\
      controller='position', camera_name=camera_name, camera_depth=collect_depth,\
       camera_width=image_size, camera_height=image_size, horizon = horizon)

	env = PandaLiftWrapper(robo_env, cfg)
	############################################################
	### Starting tests
	###########################################################
	while env.file_num < 150:
		waypoints_list = sample_waypoints(env)
		env.reset()
		env.waypoint_following(waypoints_list)


