import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import copy
import os
import datacollection_util as dc_T
import matplotlib.pyplot as plt
import random
import itertools

sys.path.insert(0, "../robosuite/") 

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

	peg_types = ["Cross", "Rect", "Square"]
	obs_keys = [ "force_hi_freq", "proprio", "action", "contact", "joint_pos", "joint_vel", 'rgbd' ]
	peg_dict = {}

	with open("datacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	logging_folder = cfg['datacollection_params']['logging_folder']
	logging_data_bool = cfg['datacollection_params']['logging_data_bool']
	display_bool = cfg['datacollection_params']['display_bool']
	num_trajectories = cfg['datacollection_params']['num_trajectories']
	kp = np.array(cfg['datacollection_params']['kp'])
	noise_parameter = np.array(cfg['datacollection_params']['noise_parameter'])
	ctrl_freq = np.array(cfg['datacollection_params']['control_freq'])

	workspace_dim = cfg['datacollection_params']['workspace_dim']
	seed = cfg['datacollection_params']['seed']

	random.seed(seed + 10)
	np.random.seed(seed + 10)

	if os.path.isdir(logging_folder) == False and logging_data_bool == 1:
		os.mkdir(logging_folder )

	print("Robot operating with control frequency: ", ctrl_freq)
	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
	  gripper_type ="CrossPegwForce", controller='position', camera_depth=True)

	obs = env.reset()
	env.viewer.set_camera(camera_id=2)

	tol = 0.01
	tol_ang = 100 #this is so high since we are only doing position control

	for idx, peg_type in enumerate(peg_types):
		peg_bottom_site = peg_type + "Peg_bottom_site"
		peg_top_site = peg_type + "Peg_top_site"

		top = np.concatenate([env._get_sitepos(peg_top_site) - np.array([0, 0, 0.01]), np.array([np.pi, 0, np.pi])])
		bottom = np.concatenate([env._get_sitepos(peg_bottom_site) + np.array([0, 0, 0.05]), np.array([np.pi, 0, np.pi])])

		peg_vector = np.zeros(len(peg_types))
		peg_vector[idx] = 1.0

		peg_dict[peg_type] = [top, bottom, peg_vector]

	peg_hole_options = list(itertools.product(*[peg_types, peg_types]))

	fp_array = dc_T.slidepoints(workspace_dim, num_trajectories)

	file_num = 0

	for peg_hole_option in peg_hole_options:
		peg_type = peg_hole_option[0]
		hole_type = peg_hole_option[1]

		option_file_num = 0

		# if peg_type == hole_type:
		# 	continue

		env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
		 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=ctrl_freq,\
		  gripper_type = peg_type + "PegwForce", controller='position', camera_depth=True)

		obs = env.reset()
		env.viewer.set_camera(camera_id=2)

		top_goal = peg_dict[hole_type][0]
		bottom_goal = peg_dict[hole_type][1]

		hole_vector = peg_dict[hole_type][2]
		peg_vector = peg_dict[peg_type][2]

		if peg_type == hole_type:
			fit_bool = np.array([1.0])
		else:
			fit_bool = np.array([0.0])

		# moving to first initial position

		num_files = (fp_array.shape[0] // 2) * len(peg_hole_options)

		points_list = []
		point_idx = 0
		point_count = 0

		for idx in range(int(fp_array.shape[0] / 2)):
			idx0 = 2 * idx
			idx1 = 2 * idx + 1
			points_list.append((fp_array[idx0] + top_goal, 0, "sliding"))
			points_list.append((fp_array[idx1]+ top_goal, 1, "sliding_recording"))
			points_list.append((top_goal + np.array([0, 0, 0.02, 0,0,0]), 2, "top plus"))
			# point_count += 1
			# point_count %= 2
			# points_list.append((top_goal, 0, "top goal"))
			# points_list.append((bottom_goal, 1, "insertion"))

			# points_list.append((top_goal, 0, "top_goal"))

		goal = points_list[point_idx][0]

		counter = 0

		while env._check_poserr(goal, tol, tol_ang) == False and counter < 100:
			action, action_euler = env._pose_err(goal)
			pos_err = kp * action_euler[:3]
			noise = np.random.normal(0.0, 0.1, pos_err.size)
			pos_err += noise_parameter * noise
			obs, reward, done, info = env.step(pos_err)
			obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal
			counter += 1

			if display_bool:
				env.render()

		print("moved to initial position")
		# dc_T.plot_image(obs['image'])
		# a = input("")
		goal = points_list[point_idx][0]
		point_type = -1
		point_idx += 1
		point_num_steps = 0

		obs_dict = {}

		glb_ts = 0


		while point_idx < len(points_list) :
			prev_point_type = copy.deepcopy(point_type)
			goal = points_list[point_idx][0]
			point_type = points_list[point_idx][1]
			print("Translation type is ", points_list[point_idx][2])
			# print("previous point type ", prev_point_type, " current point type ", point_type)

			if point_type == 1:
				final_point = copy.deepcopy(goal)

			while env._check_poserr(goal, tol, tol_ang) == False:
				action, action_euler = env._pose_err(goal)
				pos_err = kp * action_euler[:3]
				noise = np.random.normal(0.0, 0.1, pos_err.size)
				pos_err += noise_parameter * noise

				obs['action'] = env.controller.transform_action(pos_err)

				if logging_data_bool == 1 and point_type == 1:
					dc_T.save_obs(copy.deepcopy(obs), obs_keys, obs_dict)

				obs, reward, done, info = env.step(pos_err)
				obs['proprio'][:top_goal.size] = obs['proprio'][:top_goal.size] - top_goal


				# if display_bool:
				# 	plt.scatter(glb_ts, obs['force'][2])
				# 	# plt.scatter(glb_ts, obs['contact'])
				# 	plt.pause(0.001)
				# glb_ts += 1

				if display_bool:
					env.render()
	        	
				if logging_data_bool == 1 and point_type == 2 and prev_point_type == 1:
					file_num += 1
					option_file_num += 1
					print("On ", file_num, " of ", num_files, " trajectories")
					file_name = logging_folder + peg_type + "_" + hole_type + "_" + str(option_file_num).zfill(4) + ".h5"

					dataset = h5py.File(file_name, 'w')

					for key in obs_dict.keys():

						key_list = obs_dict[key]

						if key == obs_keys[0]:
							print("Number of points: ", len(key_list))

						key_array = np.concatenate(key_list, axis = 0)
						chunk_size = (1,) + key_array[0].shape
						dataset.create_dataset(key, data= key_array, chunks = chunk_size)

					dataset.create_dataset("hole_type", data = hole_vector)
					dataset.create_dataset("final_point", data = final_point)
					dataset.create_dataset("peg_type", data = peg_vector)
					dataset.create_dataset("fit", data = fit_bool)
					dataset.close()

					print("Saving to file: ", file_name)
					obs_dict = {}
					prev_point_type = -1

				point_num_steps += 1

				if point_num_steps >= 75:
					point_idx += 1
					if point_idx >= len(points_list):
						break
					goal = points_list[point_idx][0]
					prev_point_type = copy.deepcopy(point_type)
					point_type = points_list[point_idx][1]
					print("Number of Steps", point_num_steps)
					point_num_steps = 0
					# print("Skipped Point")

					print("Translation type is ", points_list[point_idx][2])

			print("Number of Steps", point_num_steps)	
			point_num_steps = 0
			point_idx += 1


	print("Finished Data Collection")
