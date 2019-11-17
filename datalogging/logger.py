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

sys.path.insert(0, "../datalogging/") 

from models import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile


class Logger(object):
	def __init__(self, cfg, debugging_flag, save_model_flag, device):
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
			load_cfg['run_tracker']['debugging'] = 0
			load_cfg['run_tracker']['training'] = 0        
			load_cfg['run_tracker']['testing'] = 0

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
				self.save_dict("config_params", cfg)
				
			self.writer = SummaryWriter(self.runs_folder)

	def save_scalars(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(logging_dict['scalar'].keys()) != 0:

			for key in logging_dict['scalar'].keys():
				self.writer.add_scalar(label + key, logging_dict['scalar'][key], iteration)

	def save_dict(self, name, dictionary):
		if self.debugging_flag == False:
			with open(self.runs_folder + name + ".yml", 'w') as ymlfile2:
				yaml.dump(dictionary, ymlfile2)

	def save_images2D(self, logging_dict, iteration, label):
		if self.debugging_flag == False and len(logging_dict['image']) != 0:
			image_list = logging_dict['image']

			if len(image_list) != 0:
				for idx, image in enumerate(image_list):
					image_list[idx] = image.detach().cpu().numpy()

				image_array = np.rot90(np.concatenate(image_list, axis = 1), k = 3, axes=(1,2)).astype(np.uint8)

				self.writer.add_image(label + 'predicted_image', image_array, iteration)
