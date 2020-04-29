import scipy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline as spline_funct
from scipy.misc import imresize as resize
import matplotlib.pyplot as plt
import random
import copy

def plot_image(image):
	image = np.rot90(image, k =2)
	imgplot = plt.imshow(image)
	plt.show()
    
def save_obs(obs_dict, keys, array_dict):
	for key in keys:
		if key == "rgbd":
			continue
			obs0 = np.rot90(obs_dict['image'], k = 2).astype(np.uint8)
			obs0 = resize(obs0, (128, 128))
			obs0 = np.transpose(obs0, (2, 1, 0))
			obs0 = np.expand_dims(obs0, axis = 0)

			obs1 = np.rot90(obs_dict['depth'], k = 2).astype(np.uint8)
			obs1 = resize(obs1, (128, 128))
			obs1 = np.expand_dims(np.expand_dims(obs1, axis = 0), axis = 0)

			obs = np.concatenate([obs0, obs1], axis = 1)

		else:
			obs = np.expand_dims(obs_dict[key], axis = 0)

		if key in array_dict.keys():
			array_dict[key].append(obs)
		else:
			array_dict[key] = [obs]

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4


def slidepoints(workspace_dim, num_trajectories = 10):
	zmin = - 0.01
	# print("Zs: ", zmin)

	theta_init = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
	theta_delta = np.random.uniform(low=3 * np.pi / 4, high=np.pi, size = num_trajectories)
	theta_sign = np.random.choice([-1, 1], size = num_trajectories)
	theta_final = T_angle(theta_init + theta_delta * theta_sign)

	c_point_list = []

	for idx in range(theta_init.size):
		x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
		theta0 = theta_init[idx]
		theta1 = theta_final[idx]
		x_init = workspace_dim * np.cos(theta0)
		y_init = workspace_dim * np.sin(theta0)
		x_final = workspace_dim * np.cos(theta1)
		y_final = workspace_dim * np.sin(theta1) 

		# print("Initial point: ", x_init, y_init)
		# print("Final point: ", x_final, y_final)
		c_point_list.append(np.expand_dims(np.array([x_init, y_init, zmin, 0, 0, 0, x_final, y_final, zmin]), axis = 0))

	return np.concatenate(c_point_list, axis = 0)


def movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs, obs_dict = None):
	kp, noise_parameters, tol, tol_ang, step_threshold, display_bool, top_height, obs_keys = fixed_params

	step_count = 0
	goal = points_list[point_idx][0]
	point_type = points_list[point_idx][1]
	done_bool = False

	while env._check_poserr(goal, tol, tol_ang) == False and step_count < step_threshold:
		action, action_euler = env._pose_err(goal)
		pos_err = kp * action_euler[:3]

		# noise = np.random.normal(0.0, 0.1, pos_err.size)
		# noise[2] = -1.0 * abs(noise[2])
		# pos_err += noise_parameters * noise

		obs['action'] = env.controller.transform_action(pos_err)

		if point_type == 1 and obs_dict is not None:
			save_obs(copy.deepcopy(obs), obs_keys, obs_dict)
			# curr_pose = env._get_eepos()

		obs, reward, done, info = env.step(pos_err)
		obs['proprio'][:3] = obs['proprio'][:3] - top_goal[:3]
		
		if display_bool:
			env.render()

		# print(obs['proprio'][2])

		if obs['proprio'][2] < - 0.0001 and point_type == 1:
			# print("Inserted")
			obs['insertion'] = np.array([1.0])
			done_bool = True
			# step_count = step_threshold
		else:
			obs['insertion'] = np.array([0.0])

		# if display_bool:
		# 	plt.scatter(glb_ts, obs['force'][2])
		# 	# plt.scatter(glb_ts, obs['contact'])
		# 	plt.pause(0.001)
		# glb_ts += 1

		step_count += 1

	point_idx += 1

	if obs_dict is not None:
		return point_idx, done_bool, obs, obs_dict
	else:
		return point_idx, done_bool, obs


