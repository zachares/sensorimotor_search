import scipy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline as spline_funct
from scipy.misc import imresize as resize
import matplotlib.pyplot as plt
import random

def plot_image(image):
	image = np.rot90(image, k =2)
	imgplot = plt.imshow(image)
	plt.show()

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4
def save_obs(obs_dict, keys, array_dict):
	for key in keys:
		if key == "rgbd":
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

def gridpoints(workspace_dim, peg_top_site, num_points = 10):
	xmin = peg_top_site[0] - workspace_dim
	xmax = peg_top_site[0] + workspace_dim

	ymin = peg_top_site[1] - workspace_dim
	ymax = peg_top_site[1] + workspace_dim

	zmin = peg_top_site[2] - 0.008
	zmax = peg_top_site[2] + 2 * workspace_dim

	print("Zs: ", zmin, " ",zmax)
	x = np.random.uniform(low=xmin, high=xmax, size = num_points)#np.linspace(xmin, xmax, num = num_points, endpoint = True)
	y = np.random.uniform(low=ymin, high=ymax, size=num_points)#np.linspace(ymin, ymax, num = num_points, endpoint = True)
	z = np.random.uniform(low=zmin, high=zmax, size=num_points)#np.linspace(zmin, zmax, num = num_points, endpoint = True)

	xfloor = np.random.uniform(low=xmin, high=xmax, size = int(0.5 *(num_points**(3))))
	yfloor = np.random.uniform(low=ymin, high=ymax, size= int(0.5 * (num_points**(3))))

	nc_point_list = []
	c_point_list = []


	for idx_x in range(xfloor.size):
		for idx_y in range(yfloor.size):
				x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				c_point_list.append(np.expand_dims(np.array([xfloor[idx_x], yfloor[idx_y], zmin, x_ang, y_ang, z_ang]), axis = 0))

	# for idx_x in range(num_points):
	# 	for idx_y in range(num_points):
	# 		for idx_z in range(num_points):
	# 			x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	# 			y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	# 			z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
	# 			nc_point_list.append(np.expand_dims(np.array([x[idx_x], y[idx_y], z[idx_z], x_ang, y_ang, z_ang]), axis = 0))

	point_list = c_point_list + nc_point_list

	return np.concatenate(point_list, axis = 0)

def slidepoints(workspace_dim, num_trajectories = 10):
	zmin = - 0.00
	# print("Zs: ", zmin)

	theta_init = np.random.uniform(low=0, high=2*np.pi, size = num_trajectories)
	theta_delta = np.random.uniform(low=np.pi / 2, high=np.pi, size = num_trajectories)
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
		c_point_list.append(np.expand_dims(np.array([x_init, y_init, zmin, x_ang, y_ang, z_ang]), axis = 0))
		c_point_list.append(np.expand_dims(np.array([x_final, y_final, zmin, x_ang, y_ang, z_ang]), axis = 0))

	return np.concatenate(c_point_list, axis = 0)
# def gridpoints(workspace, peg_top_site, num_points = 10):
# 	xmin = workspace[0,0]
# 	xmax = workspace[1,0]

# 	ymin = workspace[0,1]
# 	ymax = workspace[1,1]

# 	zmin = workspace[0,2]
# 	zmax = workspace[1,2]

# 	print("Zs: ", zmin, " ",zmax)
# 	x = np.linspace(xmin, xmax, num = num_points, endpoint = True)
# 	y = np.linspace(ymin, ymax, num = num_points, endpoint = True)
# 	z = np.linspace(zmin, zmax, num = num_points, endpoint = True)

# 	point_list = []

# 	for idx_x in range(num_points):
# 		for idx_y in range(num_points):
# 			for idx_z in range(num_points):
# 				init_pos = np.array([x[idx_x], y[idx_y], z[idx_z]])
# 				vect = peg_top_site[:3] - init_pos
# 				vect = vect / np.linalg.norm(vect)
# 				x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
# 				y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
# 				z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)

# 				point_list.append(np.expand_dims(np.array([x[idx_x], y[idx_y], z[idx_z], x_ang, y_ang, z_ang]), axis = 0))

	# return np.concatenate(random.shuffle(point_list), axis = 0)



