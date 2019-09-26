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

	def __init__(self, cfg_0, cfg_1, cfg_2, model_dict, debugging_flag, save_model_flag, device):

		self.debugging_flag = debugging_flag
		self.save_model_flag = save_model_flag

		run_description = cfg_2['logging_params']['run_description'] 

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

		if self.debugging_flag == False:

			run_tracking_num = load_cfg['run_tracker'][run_description]
			logging_folder = cfg_2['logging_params']['logging_folder']

			if os.path.isdir(logging_folder) == False:
				os.mkdir(logging_folder)

			if os.path.isdir(logging_folder + 'runs/') == False:
				os.mkdir(logging_folder + 'runs/')

			if os.path.isdir(logging_folder + 'models/') == False:
				os.mkdir(logging_folder + 'models/')

			trial_description = cfg_2['logging_params']['run_notes']

			runs_folder = logging_folder + 'runs/' + time_str + "_"+ run_description + "_" +\
			str(run_tracking_num) + trial_description + "/"

			print("Runs folder:   ", runs_folder)    

			os.mkdir(runs_folder)

			self.runs_folder = runs_folder

			self.writer = SummaryWriter(runs_folder)

			with open(runs_folder + "config_params.yml", 'w') as ymlfile2:
				yaml.dump(cfg_0, ymlfile2)
				yaml.dump(cfg_1, ymlfile2)
				yaml.dump(cfg_2, ymlfile2)
				yaml.dump(load_cfg, ymlfile2)

			self.model_path_dict = {}

			for key in model_dict.keys():

				self.model_path_dict[key] = logging_folder + 'models/' + time_str + "_" + run_description + "_" +\
			str(run_tracking_num) + "_" + trial_description + '_' + key + '.ckpt'

	def save_images1D(self, image_list, image_label, iteration, label):

		#assumes all images are the same size
		## needs work to standardize image size to give to writer
		if self.debugging_flag == False and len(image_list) != 0:

			upsampled_image_list = []

			for image in image_list:
				upsampled_image_list.append(np.repeat(np.expand_dims(np.repeat(np.repeat(fut_fb[image_index].cpu().detach().numpy(), 20, axis = 0)\
				, 20, axis = 1), axis = 0), 3, axis = 0))

				concat_image = np.concatenate(upsampled_image_list, 2)

				full_label = label + image_label

				self.writer.add_image(full_label, concat_image, iteration)

	def save_scalars(self, scalar_dict, iteration, label):

		if self.debugging_flag == False and len(scalar_dict.keys()) != 0:

			for key in scalar_dict.keys():
				# print("Label: ", label + key)
				# print("Value: ", scalar_dict[key])
				# print("Iteration: ", iteration)
				self.writer.add_scalar(label + key, scalar_dict[key], iteration)

	def save_models(self, model_dict, epoch_num): #, dynamics_model, goal_provider,

		if self.debugging_flag == False and self.save_model_flag:

			for key in model_dict.keys():

				ckpt_path = '{}.{}'.format(self.model_path_dict[key], epoch_num)
				print(ckpt_path)

				print("Saving checkpoint after epoch #{}".format(epoch_num))
				torch.save(model_dict[key].state_dict(), ckpt_path)

