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
from dataloader import *
import yaml

from models import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile


class Logger(object):

	def __init__(self, cfg, debugging_flag, device):

		self.debugging_flag = debugging_flag

		run_description = cfg['logging_params']['run_description'] 


		##### Code to keep track of runs during a day and create a unique path for logging each run
		with open("run_tracking.yml", 'r+') as ymlfile1:
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
			load_cfg['run_tracker']['debugging'] = 0
			load_cfg['run_tracker']['training'] = 0        
			load_cfg['run_tracker']['testing'] = 0

		with open("run_tracking.yml", 'r+') as ymlfile1:
			yaml.dump(load_cfg, ymlfile1)


		self.runs_folder = ""
		self.models_folder = ""

		if self.debugging_flag == False:

			run_tracking_num = load_cfg['run_tracker'][run_description]
			logging_folder = cfg['logging_params']['logging_folder']

			if os.path.isdir(logging_folder) == False:
				os.mkdir(logging_folder)

			if os.path.isdir(logging_folder + 'runs/') == False:
				os.mkdir(logging_folder + 'runs/')

			if os.path.isdir(logging_folder + 'models/') == False:
				os.mkdir(logging_folder + 'models/')

			trial_description = cfg['logging_params']['run_notes']

			self.runs_folder = logging_folder + 'runs/' + time_str + "_"+ run_description + "_" +\
			str(run_tracking_num) + trial_description + "/"

			self.models_folder = logging_folder + 'models/' + time_str + "_"+ run_description + "_" +\
			str(run_tracking_num) + trial_description + "/"

			print("Runs folder:   ", self.runs_folder)    

			os.mkdir(self.runs_folder)

			self.writer = SummaryWriter(self.runs_folder)

			self.save_dict("config_params", cfg)

	def save_scalars(self, scalar_dict, iteration, label):

		if self.debugging_flag == False and len(scalar_dict.keys()) != 0:

			for key in scalar_dict.keys():
				self.writer.add_scalar(label + key, scalar_dict[key], iteration)

	def save_dict(self, name, dictionary):

		if self.debugging_flag == False:
			with open(self.runs_folder + name + ".yml", 'w') as ymlfile2:
				yaml.dump(dictionary, ymlfile2)

	# def save_images1D(self, image_list, image_label, iteration, label):

	# 	#assumes all images are the same size
	# 	## needs work to standardize image size to give to writer
	# 	if self.debugging_flag == False and len(image_list) != 0:

	# 		upsampled_image_list = []

	# 		for image in image_list:
	# 			upsampled_image_list.append(np.repeat(np.expand_dims(np.repeat(np.repeat(fut_fb[image_index].cpu().detach().numpy(), 20, axis = 0)\
	# 			, 20, axis = 1), axis = 0), 3, axis = 0))

	# 			concat_image = np.concatenate(upsampled_image_list, 2)

	# 			full_label = label + image_label

	# 			self.writer.add_image(full_label, concat_image, iteration)

