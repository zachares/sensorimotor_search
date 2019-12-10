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
import yaml
import pickle

sys.path.insert(0, "../datalogging/") 

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 
from shutil import copyfile


class Logger(object):
	def __init__(self, cfg, debugging_flag, save_model_flag):
		self.debugging_flag = debugging_flag

		run_description = cfg['logging_params']['run_description'] 
		trial_description = cfg['logging_params']['run_notes']
		logging_folder = cfg['logging_params']['logging_folder']

		##### Code to keep track of runs during a day and create a unique path for logging each run
		with open("/scr-ssd/sens_search/learning/datalogging/run_tracking.yml", 'r+') as ymlfile1:
			load_cfg = yaml.safe_load(ymlfile1)

		t_now = time.time()

		date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d')
		time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M')

		# print("Run tracker Date: ", load_cfg['run_tracker']['date'] ,"  Actual Date: ", date)
		if load_cfg['run_tracker']['date'] == date:
			load_cfg['run_tracker'][run_description] +=1
		else:
			print("New day of training!")
			load_cfg['run_tracker']['date'] = date
			load_cfg['run_tracker']['debugging'] -= load_cfg['run_tracker']['debugging']
			load_cfg['run_tracker']['training'] -= load_cfg['run_tracker']['training']        
			load_cfg['run_tracker']['testing'] -= load_cfg['run_tracker']['testing']

		with open("/scr-ssd/sens_search/learning/datalogging/run_tracking.yml", 'r+') as ymlfile1:
			yaml.dump(load_cfg, ymlfile1)

		self.runs_folder = ""
		self.models_folder = ""

		if self.debugging_flag == False:

			run_tracking_num = load_cfg['run_tracker'][run_description]

			if os.path.isdir(logging_folder) == False:
				os.mkdir(logging_folder)

			if os.path.isdir(logging_folder + 'runs/') == False:
				os.mkdir(logging_folder + 'runs/')

			self.runs_folder = logging_folder + 'runs/' + time_str + "_"+ run_description + "_" +\
			str(run_tracking_num) + trial_description + "/"

			print("Runs folder: ", self.runs_folder)    

			os.mkdir(self.runs_folder)
			
			if save_model_flag:
				if os.path.isdir(logging_folder + 'models/') == False:
					os.mkdir(logging_folder + 'models/')

				self.models_folder = logging_folder + 'models/' + time_str + "_"+ run_description + "_" +\
				str(run_tracking_num) + trial_description + "/"

				os.mkdir(self.models_folder)
				print("Model Folder:  ", self.models_folder)
				self.save_dict("config_params", cfg, True)
				
			self.writer = SummaryWriter(self.runs_folder)

	def save_scalars(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(logging_dict['scalar'].keys()) != 0:

			for key in logging_dict['scalar'].keys():
				self.writer.add_scalar(label + key, logging_dict['scalar'][key], iteration)

	def save_dict(self, name, dictionary, yml_bool):
		if self.debugging_flag == False and yml_bool:
			with open(self.runs_folder + name + ".yml", 'w') as ymlfile2:
				yaml.dump(dictionary, ymlfile2)

		elif self.debugging_flag == False:
		    with open(self.runs_folder + name + '.pkl', 'wb') as f:
		        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

	def save_images2D(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(list(logging_dict['image'].keys())) != 0:
			for image_key in logging_dict['image']:

				image_list = logging_dict['image'][image_key]

				if len(image_list) != 0:
					for idx, image in enumerate(image_list):
						image_list[idx] = image.detach().cpu().numpy()

					image_array = np.rot90(np.concatenate(image_list, axis = 1), k = 3, axes=(1,2)).astype(np.uint8)

					self.writer.add_image(label + image_key + 'predicted_image' , image_array, iteration)

	def save_npimages2D(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(list(logging_dict['image'].keys())) != 0:
			for image_key in logging_dict['image']:

				image_list = logging_dict['image'][image_key]

				if len(image_list) != 0:
					image_array = np.expand_dims(np.concatenate(image_list, axis = 1), axis = 0)

					self.writer.add_image(label + image_key + 'visitation_freq' , image_array, iteration)

	def save_tsne(self, points, labels_list, iteration, label, tensor_bool):	
		if self.debugging_flag == False:
			perplexity = 30.0
			lr_rate = 500
			initial_dims = 30
			tsne = TSNE(n_components=2, perplexity = 30.0, early_exaggeration = 12.0, learning_rate = 200.0, n_iter = 1000, method='barnes_hut')
			print("Beginning TSNE")
			if tensor_bool:
				Y = tsne.fit_transform(points.detach().cpu().numpy())
			else:
				Y = tsne.fit_transform(points)

			print("Finished TSNE")

			for idx, label_tuple in enumerate(labels_list):
				description, labels = label_tuple
				plt.switch_backend('agg')
				fig = plt.figure()
				if tensor_bool:
					plt.scatter(Y[:,0], Y[:,1], c = labels.detach().cpu().numpy())
				else:
					plt.scatter(Y[:,0], Y[:,1], c = labels)

				self.writer.add_figure(label + '_tsne_' + description, fig, iteration)