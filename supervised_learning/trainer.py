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
		proprio_size =self.info_flow['dataset']['outputs']['proprio'] 
		action_dim =self.info_flow['dataset']['outputs']['action']
		z_dim = cfg["model_params"]["z_dim"] 
		self.batch_size = batch_size

		if 'curriculum' in cfg['training_params'].keys():
			self.curriculum = cfg['training_params']['curriculum']
		else:
			self.curriculum = None

		### Initializing Model ####
		print("Initializing Neural Network Models")
		self.model_dict = {}
		self.model_inputs = {}
		self.model_outputs = {}
		###############################################
		##### Declaring models to be trained ##########
		#################################################
		##### Note if a path has been provided then the model will load a previous model
		self.model_dict["Relational_Multimodal"] = Relational_Multimodal(models_folder, "Relational_Multimodal", self.info_flow, image_size, proprio_size, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)
		self.model_dict["VAE_Multimodal"] = VAE_Multimodal(models_folder, "VAE_Multimodal", self.info_flow, image_size, proprio_size, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)
		self.model_dict["Selfsupervised_Multimodal"] = Selfsupervised_Multimodal(models_folder, "Selfsupervised_Multimodal", self.info_flow, image_size, proprio_size, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)
		self.model_dict["Selfsupervised_Dynamics"] = Dynamics_DetModel(models_folder, "Selfsupervised_Dynamics", self.info_flow, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)	
		self.model_dict["VAE_Dynamics"] = Dynamics_DetModel(models_folder, "VAE_Dynamics", self.info_flow, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)
		self.model_dict["Relational_Dynamics"] = Dynamics_DetModel(models_folder, "Relational_Dynamics", self.info_flow, z_dim,\
		 action_dim, device = device, curriculum = self.curriculum).to(device)				 	
		###############################################
		###### Code ends here ########################
		################################################

		#### Setting per step model update method ####
		# Adam is a type of stochastic gradient descent    
		self.opt_dict = {}

		for key in self.model_dict.keys():
			if self.info_flow[key]["train"] == 1:
				print("Training " , key)
				parameters_list = list(self.model_dict[key].parameters())
				self.opt_dict[key] = optim.Adam(parameters_list, lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
			else:
				print("Not Training ", key)
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
		self.loss_dict["Rec_image_multistep"] = Image_Reconstruction_MultiStep(nn.MSELoss(), offset = 0)
		self.loss_dict["Pred_image_multistep"] = Image_Reconstruction_MultiStep(nn.MSELoss())
		self.loss_dict["Pred_multistep_list"] = Proto_MultiStep_Loss_List(nn.MSELoss())
		self.loss_dict["Rec_multistep"] = Proto_MultiStep_Loss(nn.MSELoss(), offset = 0)
		self.loss_dict["KL_DIV_multistep"] = Gaussian_KL_MultiStep()
		self.loss_dict["Pred_eepos_multistep"] = Proto_MultiStep_Loss(nn.MSELoss(), max_idx = 3)
		self.loss_dict["Prior_multistep"] = Prior_Multistep()
		# self.loss_dict["Cross_Ent"] = Proto_Loss(nn.CrossEntropyLoss(), "cross_ent")
		self.loss_dict["BCE_multistep"] = BinaryEst_MultiStep(nn.BCEWithLogitsLoss())
		###################################
		####### Code ends here ###########
		####################################
		####################################
		##### Training Results Dictionary for logger #############
		##########################################
		self.logging_dict = {}
		self.logging_dict['scalar'] = {}
		self.logging_dict['image'] = {}

	def train(self, sample_batched):
		torch.enable_grad()
		for key in self.model_dict.keys():
			self.model_dict[key].train()

		loss = self.forward_pass(sample_batched)

		for key in self.opt_dict.keys():
			self.opt_dict[key].zero_grad()

		loss.backward()

		for key in self.opt_dict.keys():
			self.opt_dict[key].step()

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
		loss_idx = 0
		loss_bool = False
		for idx_model, model_key in enumerate(self.model_dict.keys()):
			for idx_output, output_key in enumerate(self.info_flow[model_key]['outputs'].keys()):
				if self.info_flow[model_key]['outputs'][output_key]['loss'] == "":
					loss_idx += 1
					continue

				input_list = [self.model_outputs[model_key][output_key]]

				if 'inputs' in list(self.info_flow[model_key]['outputs'][output_key].keys()):
					for input_key in self.info_flow[model_key]['outputs'][output_key]['inputs'].keys():
						input_source = self.info_flow[model_key]['outputs'][output_key]['inputs'][input_key]
						input_list.append(self.model_outputs[input_source][input_key])

				loss_function = self.loss_dict[self.info_flow[model_key]['outputs'][output_key]['loss']]
				loss_name = self.info_flow[model_key]["outputs"][output_key]["loss_name"]

				if loss_bool == False:
					loss = loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name)
					loss_bool = True
				else:
					loss += loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name)

		return loss

	def load(self, path_dict = {}):
		for key in self.model_dict.keys():
			self.model_dict[key].load(path_dict)

	def save(self, epoch_num):
		for key in self.model_dict.keys():
			self.model_dict[key].save(epoch_num)		



