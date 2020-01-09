import scipy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline as spline_funct
from scipy.misc import imresize as resize
import matplotlib.pyplot as plt


def plot_image(image):
	image = np.rot90(image, k =2)
	imgplot = plt.imshow(image)
	plt.show()

def save_obs(obs_dict, prev_obs_dict, keys, array_dict):
	for key in keys:
		if key == "image":
			obs = np.rot90(prev_obs_dict[key], k = 2)
			obs = resize(obs, (128, 128))
			obs = np.transpose(obs, (2, 1, 0))
			obs = np.expand_dims(obs, axis = 0)

		else:
			obs = np.expand_dims(prev_obs_dict[key], axis = 0)

		if key in array_dict.keys():
			array_dict[key].append(obs)
		else:
			array_dict[key] = [obs]

		# if key == "image":
		# 	obs = np.rot90(obs_dict[key], k = 2)
		# 	obs = resize(obs, (128, 128))
		# 	obs = np.transpose(obs, (2, 1, 0))
		# 	obs = np.expand_dims(obs, axis = 0)
		# elif key == "action":
		# 	continue
		# else:
		# 	obs = np.expand_dims(obs_dict[key], axis = 0)

		# fut_key = key + "_fut"
	
		# if fut_key in array_dict.keys():
		# 	array_dict[fut_key].append(obs)
		# else:
		# 	array_dict[fut_key] = [obs]

def gridpoints(workspace, peg_top_site, num_points = 10):
	xmin = workspace[0,0]
	xmax = workspace[1,0]

	ymin = workspace[0,1]
	ymax = workspace[1,1]

	zmin = workspace[0,2]
	zmax = workspace[1,2]

	print("Zs: ", zmin, " ",zmax)
	x = np.linspace(xmin, xmax, num = num_points, endpoint = True)
	y = np.linspace(ymin, ymax, num = num_points, endpoint = True)
	z = np.linspace(zmin, zmax, num = num_points, endpoint = True)

	point_list = []

	for idx_x in range(num_points):
		for idx_y in range(num_points):
			for idx_z in range(num_points):
				init_pos = np.array([x[idx_x], y[idx_y], z[idx_z]])
				vect = peg_top_site[:3] - init_pos
				vect = vect / np.linalg.norm(vect)
				x_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				y_ang = 0 #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)
				z_ang = np.pi #+ 0.2 * 2 * (np.random.random_sample(1) - 0.5)

				point_list.append(np.expand_dims(np.array([x[idx_x], y[idx_y], z[idx_z], x_ang, y_ang, z_ang]), axis = 0))

	return np.concatenate(point_list, axis = 0)

def create_points_array(points_list):
	p_list = []

	for idx in range(len(points_list)):
		p_list.append(np.expand_dims(points_list[idx], axis =0))

	return np.concatenate(p_list, axis = 0)

