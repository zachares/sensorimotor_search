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

############ somehow vectorize model inputs, outputs and losses ##########

######### this model only supports ADAM for optimization at the moment
class Trainer(object):
	def __init__(self, models_folder, save_models_flag, device):

		self.device = device

		z_dim = cfg["model_params"]["z_dim"]

		batch_size = cfg['dataloading_params']['batch_size']
		regularization_weight = cfg['training_params']['regularization_weight']
		learning_rate = cfg['training_params']['lrn_rate']
		beta_1 = cfg['training_params']['beta1']
		beta_2 = cfg['training_params']['beta2']

		self.info_flow = cfg['info_flow']
		self.batch_size = batch_size

		### Initializing Model ####
		print("Initializing Neural Network Models")
		self.model_dict = {}
		self.model_inputs = {}
		self.model_outputs = {}

		###############################################
		##### Declaring models to be trained ##########
		#################################################
		##### Note if a path has been provided then the model will load a previous model
		self.model_dict["observation_encoder"] = Observations_Encoder(models_folder, "obs_enc").to(device) #### need to put in correct inputs)






		###############################################
		###### Code ends here ########################
		################################################

		#### Setting per step model update method ####
		# Adam is a type of stochastic gradient descent    
		parameters_list = []

		for key in self.model_dict.keys():
			parameters_list += list(self.model_dict[key].parameters())

		self.optimizer = optim.Adam(parameters_list, lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
		# self.optimizer = optim.Adam(list(self.obs_enc.parameters()) + list(self.dyn_mod.parameters()) + list(self.goal_prov.parameters()),\
		#  lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
		
		##### Common Loss Function ####
		self.mse_loss = nn.MSELoss()
		self.crent_loss = nn.CrossEntropyLoss()
		self.bcrent_loss = nn.BCEWithLogitsLoss()
		# self.crent_loss = nn.CrossEntropyLoss(reduction = 'none')
		# self.bcrent_loss = nn.BCEWithLogitsLoss(reduction = 'none')

		###############################################
		##### Declaring loss functions for every term in the training objective
		##############################################
		self.loss_dict = {}



		###################################
		####### Code ends here ###########
		####################################

		##### Scalar Dictionary for logger #############
		self.scalar_dict = {}

	def train(self, sample_batched):
		torch.enable_grad()
		for key in self.model_dict.keys():
			self.model_dict[key].train()

		loss = self.forward_pass(sample_batched)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return self.scalar_dict

	def eval(self, sample_batched):
		torch.no_grad()
		for key in self.model_dict.keys():
			self.model_dict[key].eval()

		loss = self.forward_pass(sample_batched)

		return self.scalar_dict

	def forward_pass(self, sample_batched):
		####################################################################
		### Useful commands in case your model ever contains flat parameters 
		### instead of just mappings. This part can usually be ignored
		###################################################################
		# loss.sum().backward(retain_graph=True)
		# flat_params_grad = flat_params.grad.clone()
		# flat_params.grad.zero_()
		#####################################################################
		self.model_outputs['dataset'] = {}

		for key in sample_batched.keys():
			self.model_outputs['dataset'][key] = sample_batched[key]

		for key in self.model_dict.keys():

			self.model_inputs[key] = []
			
			for input_key in self.info_flow[key]["inputs"].keys():
				self.model_inputs[key].append(self.model_outputs[self.info_flow[key]["inputs"][input_key]][input_key])

			model_outputs = self.model_dict[key](tuple(self.model_inputs[key]))

			output_index = 0

			self.model_outputs[key] = {}

			for output_key in self.info_flow[key]["outputs"].keys():
				self.model_outputs[key][output_key] = model_outputs[output_index]
				output_index += 1

		return self.loss()

	def loss(self):

		self.loss_inputs = {}

		for idx, key in enumerate(self.info_flow['losses'].keys()):

			self.loss_inputs[key] = []

			for input_key in self.info_flow['losses'][key]['inputs']:

				self.loss_inputs[key].append(self.model_outputs[self.info_flow[key]["inputs"][input_key]][input_key])

			if idx == 0:
				loss = self.loss_dict[key].loss(tuple(self.loss_inputs[key]), self.scalar_dict)

			else:
				loss += self.loss_dict[key].loss(tuple(self.loss_inputs[key]), self.scalar_dict) 

		return loss

	def load(self, path_dict = {}):
		for key in self.model_dict.keys():
			self.model_dict[key].load(path_dict)

	def save(self, epoch_num):
		for key in self.model_dict.keys():
			self.model_dict[key].save(epoch_num)		



