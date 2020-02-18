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

	xfloor = np.random.uniform(low=xmin, high=xmax, size = int(0.5 *(num_points**(3/2))))
	yfloor = np.random.uniform(low=ymin, high=ymax, size= int(0.5 * (num_points**(3/2))))

	nc_point_list = []
	c_point_list = []


	for idx_x in range(xfloor.size):
		for idx_y in range(yfloor.size):
				x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				c_point_list.append(np.expand_dims(np.array([xfloor[idx_x], yfloor[idx_y], zmin, x_ang, y_ang, z_ang]), axis = 0))

	for idx_x in range(num_points):
		for idx_y in range(num_points):
			for idx_z in range(num_points):
				x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				nc_point_list.append(np.expand_dims(np.array([x[idx_x], y[idx_y], z[idx_z], x_ang, y_ang, z_ang]), axis = 0))

	point_list = c_point_list + nc_point_list

	return np.concatenate(point_list, axis = 0)

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



