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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tensorboardX import SummaryWriter

sys.path.insert(0, "../robosuite/") 
sys.path.insert(0, "../models/") 
sys.path.insert(0, "../robosuite/") 
sys.path.insert(0, "../datalogging/") 
sys.path.insert(0, "../supervised_learning/")
sys.path.insert(0, "../data_collection/")

from models import *
from logger import Logger
from decision_model import *
from datacollection_util import *
from collections import defaultdict

import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy
from numpy.random import randn
from scipy import array, newaxis

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def h52Torch(h5_dict, device): #, hole_type, macro_action):
	tensor_dict = {}
	for key in h5_dict.keys():
		tensor_dict[key] = torch.from_numpy(np.array(h5_dict[key])).float().unsqueeze(0).to(device)

	return tensor_dict

def calc_angle(pose, init_angle = None):
	x = pose[0]
	y = pose[1]

	angle = T_angle(np.arctan2(y, x))

	if init_angle is None:
		return angle
	else:
		return T_angle(angle - init_angle)

def main():
	options_list = ["Cross", "Rect", "Square"]
	dataset_path = "/scr2/muj_senssearch_dataset10HZ_slide_nonoise_20200428/"
	final_path = "/scr2/muj_senssearch_dataset10HZ_slide_nonoise_20200428_wobs/"

	with open("perception_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)
	
	seed = cfg['perception_params']['seed']

	use_cuda = cfg['perception_params']['use_GPU'] and torch.cuda.is_available()

	debugging_val = cfg['debugging_params']['debugging_val']

	info_flow = cfg['info_flow']

	force_size =info_flow['dataset']['outputs']['force_hi_freq'] 
	action_dim =info_flow['dataset']['outputs']['action']
	proprio_size = info_flow['dataset']['outputs']['proprio']
	num_options = info_flow['dataset']['outputs']['num_options']

	pose_size = 3 ### magic number

	if debugging_val == 1.0:
		debugging_flag = True
		var = input("Debugging flag activated. No Results will be saved. Continue with debugging [y,n]: ")
		if var != "y":
			debugging_flag = False
	else:
		debugging_flag = False

	if debugging_flag:
		print("Currently Debugging")
	else:
		print("Training with debugged code")
	##########################################################
	### Setting up hardware for loading models
	###########################################################
	device = torch.device("cuda" if use_cuda else "cpu")
	random.seed(seed)
	np.random.seed(seed)

	if use_cuda:
	    torch.cuda.manual_seed(seed)
	else:
	    torch.manual_seed(seed)

	if use_cuda:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")

	filename_list = []
	file_list = []
	for file in os.listdir(dataset_path):
		if file.endswith(".h5"): # and len(filename_list) < 20:
			filename_list.append(dataset_path + file)
			file_list.append(file)

	acc_dict = {}

	for i in range(num_options):
		for j in range(num_options):
			acc_dict[(i,j)] = defaultdict(list)
    ##########################################################
    ### Initializing and loading model
    ##########################################################
	sensor = Options_Sensor("", "Options_Sensor", info_flow, force_size, proprio_size, action_dim, num_options, device = device).to(device)
	sensor.eval()

	seen_list = []

	for k, filename in enumerate(filename_list):

		# if k > 100:
		# 	continue

		dataset = h5py.File(filename, 'r', swmr=True, libver = 'latest')
		sample = h52Torch(dataset, device)

		# init_proprio = sample['proprio'][0,:6]
		# sample["macro_action"] = np.concatenate([init_proprio, sample['macro_action'][6:]])

		obs_probs = sensor.probs(sample)

		obs_tensor = torch.zeros(num_options).float().to(device)
		obs_tensor[obs_probs.max(1)[1]] = 1.0

		obs_np = obs_tensor.detach().cpu().numpy()
		peg_type = np.array(dataset["peg_type"])

		hole_type = np.array(dataset["hole_type"])
		macro_action = np.array(dataset["macro_action"])

		init_pose = macro_action[:3]
		final_pose = macro_action[6:]

		init_angle = calc_angle(init_pose)
		final_angle = calc_angle(final_pose, init_angle = init_angle)

		if np.sum(np.where(obs_np == hole_type, np.zeros(num_options), np.ones(num_options))) > 0:
			correct_bool = 0.0
		else:
			correct_bool = 1.0

		i = peg_type.argmax(0)
		j = hole_type.argmax(0)

		acc_dict[(i,j)]["X"].append(np.array([[init_angle, final_angle]]))
		acc_dict[(i,j)]["Y"].append(np.array([obs_probs.max(1)[1].item()]))

	n_neighbors = 3
	h = .02

	# Create color maps
	color_list = ['green', 'red', 'cornflowerblue']
	cmap_bold = {0: 'darkgreen', 1:'darkred', 2:'darkblue'} #ListedColormap(['darkgreen', 'darkred', 'darkblue'])

	weights = 'distance'

	for k, key in enumerate(acc_dict.keys()):
		# if k > 2:
		# 	continue
		i, j = key

		X = np.concatenate(acc_dict[key]["X"], axis = 0)
		y = np.concatenate(acc_dict[key]["Y"], axis = 0)

		# print(X.shape)
		# print(y.shape)

		# print(y)

		clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
		clf.fit(X, y)

		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = X[:, 0].min(), X[:, 0].max()
		y_min, y_max = X[:, 1].min(), X[:, 1].max()
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                     np.arange(y_min, y_max, h))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure()

		c_map = ListedColormap(color_list[np.unique(y).min():(np.unique(y).max() + 1)])

		plt.pcolormesh(xx, yy, Z, cmap=c_map)

		# Plot also the training points
		for g in np.unique(y):
			ix = np.where(y == g)
			plt.scatter(X[ix, 0], X[ix, 1], c=cmap_bold[g], edgecolor='k', s=20, label = options_list[g])

		# plt.legend(('Cross', 'Rect', 'Square'))

		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title( options_list[i] + " Peg + " + options_list[j] + " Hole")
		plt.legend()



		# heatmap, xedges, yedges = np.histogram2d(X, Y, bins=25)
		# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		# heatmap = np.clip(heatmap, 0, 2)

		# plt.clf()
		# plt.imshow(heatmap.T, extent=extent, origin='lower')
		# plt.colorbar()
		# plt.show()

		# a = input("")

		# plt.close()

		# surf = ax.plot_trisurf(acc_dict[key]["X"], acc_dict[key]["Y"], acc_dict[key]["Z"], cmap=cm.jet, linewidth=0)
		# fig.colorbar(surf)
		# fig.tight_layout()
		# plt.show()

	plt.show()
if __name__ == "__main__":
	main()