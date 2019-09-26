from __future__ import print_function
import os
import sys
import time
import datetime

import argparse
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from tensorboardX import SummaryWriter
from trainer import Trainer
from logger import Logger
from dataloader import *
import yaml

from models import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile

###### figure out why intersection doesnt work #########

def find_action(logits):

	# print("Logits")
	# print((logits * 10).astype(np.int16))

	action = np.zeros_like(logits)

	lp_horizontal = logits[:,:-1] * logits[:,1:]
	lp_vertical = logits[:-1,:] * logits[1:,:]

	ind_lp_hor = np.unravel_index(np.argmax(lp_horizontal, axis=None), lp_horizontal.shape)
	ind_lp_ver = np.unravel_index(np.argmax(lp_vertical, axis=None), lp_vertical.shape)

	if lp_horizontal[ind_lp_hor] > lp_vertical[ind_lp_ver]:
		action[ind_lp_hor] = 1
		action[ind_lp_hor[0], ind_lp_hor[1] + 1] = 1 
	else:
		action[ind_lp_ver] = 1
		action[ind_lp_ver[0] + 1, ind_lp_ver[1]] = 1 

	# print("Action")
	# print(action.astype(np.int16)

	return action

class Triangle_solver(object):

	def __init__(self, rows, cols):

		self.rows = rows
		self.cols = cols

	def solve(self, end_points):

		self.end_points = end_points

		if self.check_end_points():
			self.solution = self.find_solution()
		else:
			print("The points cannot be used to produce a triangle")

	def find_action_sequence(self):

		action_sequence = []
		closed_set = []

		for key in self.solution.keys():

			if len(self.solution[key]) == 0:
				continue

			else:

				key_value_bool = np.random.randint(2)

				if key_value_bool == 1:
					## use the key as the origin block
					point_2 = self.solution[key][0]
					point_1 = key

				else:
					## use the value as the origin block
					point_1 = self.solution[key][0]
					point_2 = key

				if point_1 in closed_set or point_2 in closed_set:
					continue

				closed_set.append(point_1)
				closed_set.append(point_2)

				if point_2[0] - point_1[0] == 1:
					action_sequence.append(np.array([point_1[1], point_1[0], 0]))

				elif point_2[0] - point_1[0] == -1:
					action_sequence.append(np.array([point_1[1], point_1[0], 2]))

				elif point_2[1] - point_1[1] == 1:
					action_sequence.append(np.array([point_1[1], point_1[0], 1]))

				else:
					action_sequence.append(np.array([point_1[1], point_1[0], 3]))

		return action_sequence

	def find_solution(self):
		# triangle_points - (bottom left, bottom right, top point)

		x_range = np.array([ self.end_points[0,:].min(), np.ptp(self.end_points[0,:]) + self.end_points[0,:].min()])
		y_range = np.array([ self.end_points[1,:].min(), np.ptp(self.end_points[1,:]) + self.end_points[1,:].min()])

		ip_list = [[self.end_points[1,0], self.end_points[0,0]],\
		[self.end_points[1,1], self.end_points[0,1]],\
		[self.end_points[1,2], self.end_points[0,2]]] 

		for idx_x in range(int(x_range[0]), int(x_range[1] + 1)):
			for idx_y in range(int(y_range[0]), int(y_range[1] + 1)):

				if any((item[0] == idx_y and item[1]  == idx_x) for item in ip_list):
					continue

				cand_point = [idx_y, idx_x]

				# cand_point = [11,6]
				# point_1 = [8,6]
				# point_2 = [13,6]
				# point_3 = [8,7]

				# print("Test 1: ")
				# print(self.intersection(cand_point, point_3, point_1, point_2))				
				# a = input("Pause")

				if self.intersection(cand_point, ip_list[0], ip_list[1], ip_list[2]) and\
				self.intersection(cand_point, ip_list[1], ip_list[0], ip_list[2]) and \
				self.intersection(cand_point, ip_list[2], ip_list[1], ip_list[0]):
					# continue
					ip_list.append(np.array([idx_y, idx_x]))

		total_points = len(ip_list)

		self.pred_goal = np.ones((self.rows, self.cols))

		for point in ip_list:
			self.pred_goal[int(point[0]), int(point[1])] = 0

		# self.pred_goal[int(ip_list[0][0]), int(ip_list[0][1])] = 2
		# self.pred_goal[int(ip_list[1][0]), int(ip_list[1][1])] = 2
		# self.pred_goal[int(ip_list[2][0]), int(ip_list[2][1])] = 2

		connected_points = 0

		neighbor_dict = {}

		self.ip_list = ip_list

		for point in ip_list:

			neighbor_dict[tuple(point)] = self.get_neighbor(point, ip_list)

		nodes_unconnected = True

		closed_set = []
		assignments = {}

		while nodes_unconnected:

			nodes_unconnected = False

			for key in neighbor_dict.keys():

				if len(neighbor_dict[key]) == 0:

					closed_set.append(key)

				if len(neighbor_dict[key]) == 1:

					if neighbor_dict[key][0] not in closed_set and key not in closed_set:

						closed_set.append(neighbor_dict[key][0])
						closed_set.append(key)
						assignments[key] = neighbor_dict[key][0]
						assignments[neighbor_dict[key][0]] = key
						connected_points += 2

					elif neighbor_dict[key][0] in closed_set and key not in closed_set:

						neighbor_dict[key] = []
						closed_set.append(key)

			for key in neighbor_dict.keys():

				if len(neighbor_dict[key]) > 1:
					
					nodes_unconnected = True

					if key in assignments.keys():
						neighbor_dict[key] = [assignments[key]]
						continue

					open_list = []

					for neigh in neighbor_dict[key]:

						if neigh not in closed_set:
							open_list.append(neigh)

					if len(open_list) == 0:
						neighbor_dict[key] = []
						closed_set.append(key)

					elif len(open_list) == 1:
						closed_set.append(open_list[0])
						closed_set.append(key)
						assignments[key] = open_list[0]
						assignments[open_list[0]] = key

						connected_points += 2

					else:
						cn_idx = np.random.choice(range(len(open_list))) # chosen neighbor

						neighbor_dict[key] = [open_list[cn_idx]]

						closed_set.append(open_list[cn_idx])
						closed_set.append(key)
						assignments[key] = open_list[cn_idx]
						assignments[open_list[cn_idx]] = key

						connected_points += 2

						break

		self.quality = connected_points / total_points
		self.num_points = total_points

		return neighbor_dict

	def get_neighbor(self, point, list):

		cand_list = []

		for cand in list:

			if np.sqrt((cand[0] - point[0]) ** 2  + (cand[1] - point[1]) ** 2) == 0:
				continue

			elif np.sqrt((cand[0] - point[0]) ** 2  + (cand[1] - point[1]) ** 2) <= 1:

				cand_list.append(tuple(cand))

		return cand_list

	def check_end_points(self):

		idx_x = [self.end_points[0, 0], self.end_points[0, 1], self.end_points[0, 2]]
		idx_y = [self.end_points[1, 0], self.end_points[1, 1], self.end_points[1, 2]]

		if (idx_x[0] == idx_x[1] and idx_x[0] == idx_x[2]) or (idx_y[0] == idx_y[1] and idx_y[0] == idx_y[2]):

			return False

		elif (idx_x[0] == idx_x[1] and idx_y[0] == idx_y[1]):

			return False

		elif (idx_x[0] == idx_x[2] and idx_y[0] == idx_y[2]):

			return False

		elif (idx_x[1] == idx_x[2] and idx_y[1] == idx_y[2]):

			return False

		return True

	def intersection(self, a_1, a_2, b_1, b_2):

		if int(a_2[0]) == int(a_1[0]):

			a_m = np.inf

		else:

			a_m = (a_2[1] - a_1[1]) / (a_2[0] - a_1[0])
			a_c = a_1[1] - a_m * a_1[0]

		if int(b_2[0]) == int(b_1[0]):


			b_m = np.inf

		else:
			b_m = (b_2[1] - b_1[1]) / (b_2[0] - b_1[0])
			b_c = b_1[1] - b_m * b_1[0]


		if (a_m == np.inf and b_m == np.inf) or a_m == b_m:

			# print("False: Parallel Lines")

			return False

		elif a_m == np.inf:

			int_x = a_1[0]
			int_y = b_m * int_x + b_c

		elif b_m == np.inf:

			int_x = b_1[0]
			int_y = a_m * int_x + a_c

		else:

			int_x = (b_c - a_c) / (a_m - b_m)
			int_y = a_m * int_x + a_c

		# print("Intersection Point")
		# print(int_x, int_y)

		dist_1 = np.sqrt((b_2[0] - b_1[0]) ** 2 + (b_2[1] - b_1[1]) ** 2 )
		dist_2 = np.sqrt((b_2[0] - int_x) ** 2 + (b_2[1] - int_y) ** 2 ) 
		dist_3 = np.sqrt((b_1[0] - int_x) ** 2 + (b_1[1] - int_y) ** 2 ) 

		dist_a = np.sqrt((a_2[0] - a_1[0]) ** 2 + (a_2[1] - a_1[1]) ** 2 ) 
		dist_int = np.sqrt((a_2[0] - int_x) ** 2 + (a_2[1] - int_y) ** 2 )

		# print("Distance 1: ", dist_1)
		# print("Distance 2: ", dist_2)
		# print("Distance 3: ", dist_3)


		if dist_1 >= (dist_2 - 1e-6) and dist_1 >= (dist_3 - 1e-6): # tolerance for numerical calculation errors

			if dist_a <= dist_int + 1e-7: # tolerance for numerical calculation errors

				# print("True: Candidate Point Within Triangle")

				return True

			else:

				# print("False: Candidate Point Past Intersection Point")

				return False

		else:

			# print("False: Intersection Point Outside of Line Segement")
			return False

if __name__ == '__main__':

	##################################################################################
	##### Loading required config files
	##################################################################################
	with open("../sim_env/game_params.yml", 'r') as ymlfile:
		cfg_0 = yaml.safe_load(ymlfile)

	cols = cfg_0['game_params']['cols']
	rows = cfg_0['game_params']['rows']
	pose_dim = cfg_0['game_params']['pose_dim']

	with open("../data_collection/datacollection_params.yml", 'r') as ymlfile:
		cfg_1 = yaml.safe_load(ymlfile)

	dataset_path = cfg_1['datacollection_params']['logging_folder']

	with open("representation_params.yml", 'r') as ymlfile:
		cfg_2 = yaml.safe_load(ymlfile)

	debugging_val = cfg_2['debugging_params']['debugging_val']
	save_model_val = cfg_2['debugging_params']['save_model_val']

	z_dim = cfg_2["model_params"]["model_1"]["z_dim"]

	use_cuda = cfg_2['training_params']['use_GPU'] and torch.cuda.is_available()

	seed = cfg_2['training_params']['seed']
	regularization_weight = cfg_2['training_params']['regularization_weight']
	learning_rate = cfg_2['training_params']['lrn_rate']
	beta_1 = cfg_2['training_params']['beta1']
	beta_2 = cfg_2['training_params']['beta2']
	max_epoch = cfg_2['training_params']['max_training_epochs']
	val_ratio = cfg_2['training_params']['val_ratio']
	logging_folder = cfg_2['logging_params']['logging_folder']
	run_description = cfg_2['logging_params']['run_description'] 

	num_workers = cfg_2['dataloading_params']['num_workers']

	if run_description == "testing":
		max_epoch = 1
		val_ratio = 0
		test_run = True
	else:
		test_run = False

	path = "/scr-ssd/gg_rl_overfit/dynamic_programming/models/201909090920_training_0__three_point_policy_observation_encoder.ckpt.6"


	##################################################################################
	# hardware and low level training details
	##################################################################################
	device = torch.device("cuda" if use_cuda else "cpu")
	random.seed(seed)
	np.random.seed(seed)

	if use_cuda:
		torch.cuda.manual_seed(seed)
	else:
		torch.manual_seed(seed)

	if use_cuda:
		print("Let's use", torch.cuda.device_count(), "GPUs!")

	##################################################################################
	#### Training tool to train and evaluate neural networks
	##################################################################################
	trainer = Trainer(cfg_0, cfg_1, cfg_2, device)

	trainer.load_model(path)


	##################################################################################
	### Initializing Triangle Solver to evaluate predicted triangle vertices
	##################################################################################
	triangle = Triangle_solver(rows, cols)

	filename_list = []
	for file in os.listdir(dataset_path):
		if file.endswith(".h5"):
			filename_list.append(dataset_path + file)


	for filename in filename_list:

		dataset = h5py.File(filename, 'r', swmr=True, libver = 'latest')

		for idx in range(100):

			trajectory = np.array(dataset["trajectory_" + str(idx)])

			total_steps = trajectory.shape[0]

			for step_idx in range(total_steps):

				if step_idx < 4:
					continue

				obs_t0_array = np.array(trajectory[step_idx,:-1,:])
				goal_array = np.array(trajectory[-1,:-1,:])

				# print("  ")
				# print("#############################################")

				obs_t0 = torch.tensor(obs_t0_array).float().to(device).unsqueeze(0).unsqueeze(0)

				trainer.model_dict['observation_encoder'].eval()

				point_1, point_2, point_3 = trainer.model_dict['observation_encoder'].pred_points(obs_t0)

				point_1 = point_1.detach().cpu().numpy()
				point_2 = point_2.detach().cpu().numpy()
				point_3 = point_3.detach().cpu().numpy()

				end_points = np.array([[point_1[1], point_2[1], point_3[1]], [point_1[0], point_2[0], point_3[0]]]).astype(np.int16)

				triangle.solve(end_points)

				print("Predicted Solution with current state")
				print(((np.ones_like(triangle.pred_goal) - triangle.pred_goal) + 5 * obs_t0_array).astype(np.int16))

				print("Predicted Solution with goal state")
				print(((np.ones_like(triangle.pred_goal) - triangle.pred_goal) + 5 * goal_array).astype(np.int16))

				print("Sum of nonintersecting blocks")
				print((triangle.pred_goal * obs_t0_array).sum())

				print("Intersection with goal")

				print(np.absolute((np.ones_like(triangle.pred_goal) - triangle.pred_goal) - goal_array).sum())

				print("Solution Number Internal Points")
				print(triangle.num_points)
				print("Solution Quality")
				print(triangle.quality)

				var = input("Continue?")

		dataset.close()






