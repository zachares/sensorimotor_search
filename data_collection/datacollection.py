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
sys.path.insert(0, "../decision_model")
sys.path.insert(0, "../")
sys.path.insert(0, "../../supervised_learning")

from project_utils import *
from task_models import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

	with open("datacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	logging_folder = cfg['datacollection_params']['logging_folder']
	logging_data_bool = cfg['datacollection_params']['logging_data_bool']
	display_bool = cfg['datacollection_params']['display_bool']
	num_samples = cfg['datacollection_params']['num_samples']
	plus_offset = np.array(cfg['datacollection_params']['plus_offset'])

	workspace_dim = cfg['control_params']['workspace_dim']
	kp = np.array(cfg['control_params']['kp'])
	ctrl_freq = np.array(cfg['control_params']['control_freq'])
	step_threshold = cfg['control_params']['step_threshold']
	tol = cfg['control_params']['tol']
	seed = cfg['control_params']['seed']

	dataset_keys = cfg['dataset_keys']

	peg_names = cfg['peg_names']
	hole_names = cfg['hole_names']
	fit_names = cfg['fit_names']

	random.seed(seed)
	np.random.seed(seed)

	if os.path.isdir(logging_folder) == False and logging_data_bool == 1:
		os.mkdir(logging_folder )

		with open(logging_folder + "datacollection_params.yml", 'w') as ymlfile2:
			yaml.dump(cfg, ymlfile2)

	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=False, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	env.viewer.set_camera(camera_id=2)

	hole_info = {}
	for i, hole_name in enumerate(hole_names):
		top_site = hole_name + "Peg_top_site"
		top = env._get_sitepos(top_site)
		top_height = top[2]
		hole_info[i] = {}
		hole_info[i]["pos"] = top
		hole_info[i]["name"] = hole_name

	act_model = Action_PegInsertion(hole_info, workspace_dim, tol, plus_offset, num_samples)
	act_model.generate_actions()

	peg_hole_options = list(itertools.product(*[peg_names, hole_names]))
	file_num = 0
	obs = {}

	fixed_params = (0, kp, tol, step_threshold, display_bool,top_height, dataset_keys)

	for peg_hole_option in peg_hole_options:
		option_file_num = 0

		peg_type = peg_hole_option[0]
		hole_type = peg_hole_option[1]

		if peg_type == hole_type:
			fit_type = "Fit"
		else:
			fit_type = "Not_fit"

		peg_idx = peg_names.index(peg_type)
		hole_idx = hole_names.index(hole_type)
		fit_idx = fit_names.index(fit_type)

		gripper_type = peg_type + "PegwForce" 

		env.reset(gripper_type)
		env.viewer.set_camera(camera_id=2)

		top_goal = hole_info[hole_idx]['pos']

		for i in range(num_samples):
			macro_action = act_model.get_action(i)
			points_list = act_model.transform_action(hole_idx, i)
			point_idx = 0

			point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs)
			point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list,  point_idx, obs)

			# print("moved to initial position")

			obs_dict = {}

			while point_idx < len(points_list):
				point_idx, done_bool, obs, obs_dict = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs, obs_dict)

			if logging_data_bool == 1:
				file_num += 1
				option_file_num += 1
				print("On ", file_num, " of ", len(macro_actions) * len(peg_hole_options), " trajectories")
				file_name = logging_folder + peg_type + "_" + hole_type + "_" + str(option_file_num).zfill(4) + ".h5"

				dataset = h5py.File(file_name, 'w')

				for key in obs_dict.keys():
					key_list = obs_dict[key]

					if key == dataset_keys[0]:
						print("Number of points: ", len(key_list))

					key_array = np.concatenate(key_list, axis = 0)
					chunk_size = (1,) + key_array[0].shape
					dataset.create_dataset(key, data= key_array, chunks = chunk_size)

				hole_vector = np.zeros(len(hole_names))
				peg_vector = np.zeros(len(peg_names))
				fit_vector = np.zeros(len(fit_names))

				hole_vector[hole_idx] = 1
				peg_vector[peg_idx] = 1
				fit_vector[fit_idx] = 1

				dataset.create_dataset("hole_type", data = hole_vector)
				dataset.create_dataset("peg_type", data = peg_vector)
				dataset.create_dataset("fit_type", data = fit_vector)
				dataset.create_dataset("macro_action", data = macro_action)
				dataset.close()

				print("Saving to file: ", file_name)

	print("Finished Data Collection")
