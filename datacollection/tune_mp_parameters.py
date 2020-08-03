import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools
import pickle
from collections import OrderedDict
import datetime
import matplotlib.pyplot as plt

import project_utils as pu

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("tunempparameters_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["task_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']
	seed = cfg['task_params']['seed']
	ctrl_freq = cfg['control_params']['control_freq']
	noise_scale  = 0.5
	cfg['task_params']['noise_scale'] = noise_scale
	logging_folder = cfg["logging_params"]["logging_folder"]
	name = tunempparameters_params.yml

	print("Saving ", name, " to: ", logging_folder + name)

	with open(logging_folder + name, 'w') as ymlfile2:
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
	robo_env = robosuite.make("PandaPegInsertion",\
		has_renderer= display_bool,\
		ignore_done=True,\
		use_camera_obs= not display_bool and collect_vision_bool,\
		has_offscreen_renderer = not display_bool and collect_vision_bool,\
		gripper_visualization=False,\
		control_freq=ctrl_freq,\
		gripper_type ="CrossPegwForce",\
		controller='position',\
		camera_name=camera_name,\
		camera_depth=collect_depth,\
		camera_width=image_size,\
		camera_height=image_size,\
		horizon = horizon)

	env = SensSearchWrapper(robo_env, cfg, mode_vect = cfg['task_params']['mode_vect'])
	spiral_mp = pu.Spiral2D_Motion_Primitive()

	results = OrderedDict()
	results['radius_travelled'] = np.arange(0.01, 0.03 + 0.002, 0.002)
	results['number of rotations'] = np.arange(1, 5 + 0.4, 0.4)
	results['pressure'] = np.arange(0.001,0.004 + 0.001, 0.001)
	results['test_parameters'] = list(itertools.product(results['radius_travelled'], results['number of rotations'], results['pressure']))
	results['num_samples'] = 20
	results['num_insertions'] = []

	date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M')

	results_name = date + "_motion_primitive_tuning_results"

	print("Saving ", results_name, " to: ", logging_folder + results_name + ".pkl")
	with open(logging_folder + results_name + '.pkl', 'wb') as f:
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	num_tests = len(results['test_parameters'])

	prev_time = time.time()

	for epoch in range(num_tests):
		print("\n", epoch + 1, " parameter combinations of ", num_tests, " tested")
		print("\nEpoch took ", time.time() - prev_time, '\n')
		prev_time = time.time()

		env.epoch = np.array([epoch])
		radius_travelled = results['test_parameters'][epoch][0]
		num_rotations = results['test_parameters'][epoch][1]
		pressure = results['test_parameters'][epoch][2]

		spiral_mp.rt = radius_travelled
		spiral_mp.nr = num_rotations
		spiral_mp.pressure = pressure
		env.policy = spiral_mp

		for trial_num in range(results['num_samples']):
			env.reset(initialize = False)
			env.big_step(env.robo_env.cand_idx)

		results['num_insertions'].append(env.irate)

	print("Saving ", results_name, " to: ", logging_folder + results_name + ".pkl")
	with open(logging_folder + results_name + '.pkl', 'wb') as f:
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	plt.scatter(range(len(results['num_insertions'])), results['num_insertions'])
	plt.show()

	with open(logging_folder + results_name + '.pkl', 'rb') as f:
		results = pickle.load(f)

	results['num_samples'] = 300
	results['best_parameters'] = [[],[],[],[]]

	for i, num_insert in enumerate(results['num_insertions']):
		if num_insert == 10 and len(results['best_parameters'][0]) == 0:
			results['best_parameters'][0] = results['test_parameters'][i]

		elif num_insert == 12 and len(results['best_parameters'][1]) == 0:
			results['best_parameters'][1] = results['test_parameters'][i]

		elif num_insert == 14 and len(results['best_parameters'][2]) == 0:
			results['best_parameters'][2] = results['test_parameters'][i]

		elif num_insert == 16 and len(results['best_parameters'][3]) == 0:
			results['best_parameters'][3] = results['test_parameters'][i]

	num_tests = len(results['best_parameters'])
	results['observation_model'] = []
	prev_time =  time.time()	

	for epoch in range(num_tests):
		print("\n", epoch + 1, " parameter combinations of ", num_tests, " tested")
		print("\nEpoch took ", time.time() - prev_time, '\n')
		insertion_tracking = np.zeros((len(env.robo_env.hole_names), 2))
		prev_time = time.time()

		env.epoch = np.array([epoch])
		radius_travelled = results['best_parameters'][epoch][0]
		num_rotations = results['best_parameters'][epoch][1]
		pressure = results['best_parameters'][epoch][2]

		spiral_mp.rt = radius_travelled
		spiral_mp.nr = num_rotations
		spiral_mp.pressure = pressure
		env.policy = spiral_mp

		for trial_num in range(results['num_samples']):
			env.reset(initialize = False)
			env.big_step(env.robo_env.cand_idx)

			hole_idx = env.robo_env.hole_sites[env.robo_env.cand_idx][2]
			success_index = int(env.cfg['control_params']['done_bool'] * 1.0)
			insertion_tracking[hole_idx, success_index] += 1
		
		print("Primitive Parameters: ", results['best_parameters'][epoch])
		print("Insertion Results: ", insertion_tracking)

		results['observation_model'].append((results['best_parameters'][epoch], insertion_tracking, results['num_samples']))

	date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M')

	results_name = date + "_motion_primitive_tuning_results"

	print("Saving ", results_name, " to: ", logging_folder + results_name + ".pkl")
	with open(logging_folder + results_name + '.pkl', 'wb') as f:
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	# num_tests = len(results['test_parameters'])

	# prev_time = time.time()

	# for epoch in range(num_tests):
	# 	print("\n", epoch + 1, " parameter combinations of ", num_tests, " tested")
	# 	print("\nEpoch took ", time.time() - prev_time, '\n')
	# 	prev_time = time.time()

	# 	env.epoch = np.array([epoch])
	# 	radius_travelled = results['test_parameters'][epoch][0]
	# 	num_rotations = results['test_parameters'][epoch][1]
	# 	pressure = results['test_parameters'][epoch][2]

	# 	spiral_mp.rt = radius_travelled
	# 	spiral_mp.nr = num_rotations
	# 	spiral_mp.pressure = pressure

	# 	for trial_num in range(results['num_samples']):
	# 		env.reset()
	# 		env.big_step(0, goal = spiral_mp.trajectory)

	# 	results['num_insertions'].append(env.irate)

	# print("Saving ", results_name, " to: ", logging_folder + results_name + ".pkl")
	# with open(logging_folder + results_name + '.pkl', 'wb') as f:
	# 	pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


	# with open("/scr2/muj_motion_primitive_20200722/" + "20200729_motion_primitive_tuning_results.pkl", 'rb') as f:
	# 	results = pickle.load(f)

	# print(results['observation_model'][-1])
	# print(results['num_samples'])

	# results['num_samples'] = 30

	# print("Saving ", results_name, " to: ", logging_folder + results_name + ".pkl")
	# with open(logging_folder + results_name + '.pkl', 'wb') as f:
	# 	pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	# with open(logging_folder + results_name + '.pkl', 'rb') as f:
	# 	results = pickle.load(f)

	# print(results['num_samples'])

	# results['num_samples'] = 30

	# # print(test_parameters)
