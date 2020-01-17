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


sys.path.insert(0, "../robosuite/") 

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

	with open("datacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	logging_folder = cfg['datacollection_params']['logging_folder']
	collection_details = cfg['datacollection_params']['collection_details']
	logging_data_bool = cfg['datacollection_params']['logging_data_bool']
	peg_type = cfg['datacollection_params']['peg_type']
	display_bool = cfg['datacollection_params']['display_bool']
	workspace = np.array(cfg['datacollection_params']['workspace'])

	workspace_dim = cfg['datacollection_params']['workspace_dim']
	seed = cfg['datacollection_params']['seed']

	random.seed(seed)
	np.random.seed(seed)

	if os.path.isdir(logging_folder) == False and logging_data_bool == 1:
		os.mkdir(logging_folder )

	env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
	 use_camera_obs= not display_bool, gripper_visualization=True, control_freq=100,\
	  gripper_type = peg_type + "PegwForce", controller='position', camera_depth=True)

	obs = env.reset()
	env.viewer.set_camera(camera_id=2)
	if display_bool:
		env.render()

	tol = 0.002
	tol_ang = 100

	peg_bottom_site = peg_type + "Peg_bottom_site"
	peg_top_site = peg_type + "Peg_top_site"
	offset = np.array([0, 0, 0.005])
	top_goal = np.concatenate([env._get_sitepos(peg_top_site) + offset, np.array([np.pi, 0, np.pi])])
	bottom_goal = np.concatenate([env._get_sitepos(peg_bottom_site) + offset, np.array([np.pi, 0, np.pi])])
	fp_array = dc_T.gridpoints_b(workspace_dim, top_goal, 10)

	fp_idx = 0

	print("Top goal: ", top_goal)
	print("Bottom_goal: ", bottom_goal)

	obs_keys = [ "image", "force", "proprio", "action", "contact", "joint_pos", "joint_vel", "depth"]

	# moving to first initial position
	points_list = []
	point_idx = 0
	kp = 10

	for idx in range(fp_array.shape[0]):
		point = fp_array[idx]

		if idx != fp_array.shape[0] - 1:
			points_list.append((point, 1))

			if np.random.binomial(1, 0.1) == 1:
				points_list.append((top_goal, 3))
				points_list.append((bottom_goal, 2))
				points_list.append((top_goal, 3))

	goal = points_list[point_idx][0]

	counter = 0

	while env._check_poserr(goal, tol, tol_ang) == False and counter < 100:
		action, action_euler = env._pose_err(goal)
		pos_err = kp * action_euler[:3]
		obs, reward, done, info = env.step(pos_err)
		obs['action'] = env.controller.transform_action(pos_err)
		counter += 1

	# dc_T.plot_image(obs['image'])
	# a = input("")
		if display_bool:
			env.render()

	point_idx += 1
	initial_point_idx = copy.deepcopy(point_idx)
	point_num_steps = 0
	num_points = 0
	print("moved to initial position")

	obs_dict = {}
	file_num = 0
	prev_obs = obs
	glb_ts = 0

	while point_idx != len(points_list) - 1:
		goal = points_list[point_idx][0]
		point_type = points_list[point_idx][1]

		if point_type == 1:
			print("exploring")
		elif point_type == 2:
			print("insertion")
		else:
			print("ontop of hole")


		# if logging_data_bool == 1 and point_type == 0 and point_idx != initial_point_idx:
		# 	print("On ", point_idx + 1, " of ", len(points_list), " points")
		# 	file_name = logging_folder + collection_details + "_" + str(file_num + 1).zfill(4) + ".h5"

		# 	dataset = h5py.File(file_name, 'w')

		# 	for key in obs_dict.keys():
		# 		key_list = obs_dict[key]
		# 		key_array = np.concatenate(key_list, axis = 0)
		# 		chunk_size = (1,) + key_array[0].shape
		# 		dataset.create_dataset(key, data= key_array, chunks = chunk_size)

		# 	dataset.close()
		# 	print("Saving to file: ", file_name)
		# 	file_num += 1

		# 	obs_dict = {}
		# 	num_points = 0

		while env._check_poserr(goal, tol, tol_ang) == False:
			action, action_euler = env._pose_err(goal)
			# print(action)
			pos_err = kp * action_euler[:3]

			# adding random noise
			if point_type == 0:
				pos_err += 0.2 * np.random.normal(0.0, 1.0, pos_err.size)
			else:
				pos_err += 0.2 * np.random.normal(0.0, 1.0, pos_err.size)

			obs, reward, done, info = env.step(pos_err)
			obs['action'] = env.controller.transform_action(pos_err)

			# if obs['force'][2] > 200:
			# 	print(obs['force'][2])
			if display_bool:
				# plt.scatter(glb_ts, obs['force'][2])
				plt.scatter(glb_ts, obs['contact'])
				plt.pause(0.001)

			# if obs['contact'] == 1:
			# 	print("contact")
			# else:
			# 	print(obs['contact'])

			point_num_steps += 1

			if logging_data_bool == 1:
				num_points += 1
				dc_T.save_obs(copy.deepcopy(obs), copy.deepcopy(prev_obs), obs_keys, obs_dict)

			prev_obs = obs

			if display_bool:
				env.render()
        	
			if logging_data_bool == 1 and num_points >= 200:
				print("On ", point_idx + 1, " of ", len(points_list), " points")
				file_name = logging_folder + collection_details + "_" + str(file_num + 1).zfill(4) + ".h5"

				dataset = h5py.File(file_name, 'w')

				for key in obs_dict.keys():
					key_list = obs_dict[key]
					key_array = np.concatenate(key_list, axis = 0)
					chunk_size = (1,) + key_array[0].shape
					dataset.create_dataset(key, data= key_array, chunks = chunk_size)

				dataset.close()
				print("Saving to file: ", file_name)
				file_num += 1

				obs_dict = {}
				num_points = 0

			if point_num_steps >= 300:
				if point_type == 0:
					print("Freespace")
				elif point_type == 0 or point_type == 2:
					print("Insertion")
				point_idx += 1
				goal = points_list[point_idx][0]
				point_type = points_list[point_idx][1]
				point_num_steps = 0

				# if point_type == 0:
				# 	break

			if file_num > 5000:
				break

			glb_ts += 1

		if file_num > 5000:
			break

		point_idx += 1
		point_num_steps = 0
		print("Next Point")	

	print("Finished Data Collection")

	if display_bool == 1:
		print("Closed Simulation")
