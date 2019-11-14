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
from torch.distributions import Normal
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from tensorboardX import SummaryWriter
from dataloader import *
import yaml

sys.path.insert(0, "../models/") 

from models import *
from losses import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile

############ somehow vectorize model inputs, outputs and losses ##########

######### this model only supports ADAM for optimization at the moment
class Trainer(object):
	def __init__(self, cfg, models_folder, save_models_flag, device):

		self.device = device

		z_dim = cfg["model_params"]["z_dim"]

		batch_size = cfg['dataloading_params']['batch_size']
		regularization_weight = cfg['training_params']['regularization_weight']
		learning_rate = cfg['training_params']['lrn_rate']
		beta_1 = cfg['training_params']['beta1']
		beta_2 = cfg['training_params']['beta2']

		self.info_flow = cfg['info_flow']
		image_size = self.info_flow['dataset']['outputs']['image']
		force_size =self.info_flow['dataset']['outputs']['force'] 
		action_dim =self.info_flow['dataset']['outputs']['action'] 
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
		self.model_dict["PlaNet_Multimodal"] = PlaNet_Multimodal(models_folder + "PlaNet_Multimodal", image_size, force_size, z_dim, action_dim, device = device).to(device) #### need to put in correct inputs)
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
		# self.mse_loss = Proto_Loss(nn.MSELoss(), "mean_square_error")
		# self.crent_loss = Proto_Loss(nn.CrossEntropyLoss(), "cross_ent")
		# self.bcrent_loss = Proto_Loss(nn.BCEWithLogitsLoss(), "binary_cross_ent")
		# self.kl_div = Gaussian_KL("kl_div")
		# self.crent_loss = nn.CrossEntropyLoss(reduction = 'none')
		# self.bcrent_loss = nn.BCEWithLogitsLoss(reduction = 'none')

		###############################################
		##### Declaring loss functions for every term in the training objective
		##############################################
		self.loss_dict = {}
		self.loss_dict["MSE_image"] = Image_Reconstruction(nn.MSELoss(), "mean_square_error_image")
		self.loss_dict["MSE_force"] = Proto_Loss(nn.MSELoss(), "mean_square_error_force")
		self.loss_dict["Cross_Ent"] = Proto_Loss(nn.CrossEntropyLoss(), "cross_ent")
		self.loss_dict["Binary_Cross_Ent"] = Proto_Loss(nn.BCEWithLogitsLoss(), "binary_cross_ent")
		self.loss_dict["KL_DIV"] = Gaussian_KL("kl_div")
		###################################
		####### Code ends here ###########
		####################################

		##### Scalar Dictionary for logger #############
		self.logging_dict = {}
		self.logging_dict['scalar'] = {}
		self.logging_dict['image'] = []

	def train(self, sample_batched):
		torch.enable_grad()
		for key in self.model_dict.keys():
			self.model_dict[key].train()

		loss = self.forward_pass(sample_batched)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return self.logging_dict

	def eval(self, sample_batched):
		torch.no_grad()
		for key in self.model_dict.keys():
			self.model_dict[key].eval()

		loss = self.forward_pass(sample_batched)

		return self.logging_dict

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
			self.model_outputs['dataset'][key] = sample_batched[key].to(self.device)

		for key in self.model_dict.keys():
			self.model_inputs[key] = {}
			
			for input_key in self.info_flow[key]["inputs"].keys():
				input_source = self.info_flow[key]["inputs"][input_key]

				if input_key in self.model_outputs[input_source].keys():
					self.model_inputs[key][input_key] = self.model_outputs[input_source][input_key]
				else:
					self.model_inputs[key][input_key] = None

			self.model_outputs[key] = self.model_dict[key](self.model_inputs[key])

		return self.loss()

	def loss(self):

		self.loss_inputs = {}

		for idx, key in enumerate(self.info_flow['losses'].keys()):
			self.loss_inputs[key] = []
			for input_key in self.info_flow['losses'][key]['inputs'].keys():
				input_source = self.info_flow['losses'][key]["inputs"][input_key]
				self.loss_inputs[key].append(self.model_outputs[input_source][input_key])

			loss_function = self.loss_dict[self.info_flow['losses'][key]['loss']]

			if idx == 0:
				loss = loss_function.loss(tuple(self.loss_inputs[key]), self.logging_dict, self.info_flow['losses'][key]['weight'] )
			else:
				loss += loss_function.loss(tuple(self.loss_inputs[key]), self.logging_dict, self.info_flow['losses'][key]['weight'] )

		return loss

	def load(self, path_dict = {}):
		for key in self.model_dict.keys():
			self.model_dict[key].load(path_dict)

	def save(self, epoch_num):
		for key in self.model_dict.keys():
			self.model_dict[key].save(epoch_num)		



