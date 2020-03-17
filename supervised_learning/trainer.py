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
		parallel_bool = cfg['training_params']['parallel_bool']

		self.info_flow = cfg['info_flow']
		# image_size = self.info_flow['dataset']['outputs']['image']
		force_size =self.info_flow['dataset']['outputs']['force_hi_freq'] 
		proprio_size =self.info_flow['dataset']['outputs']['proprio'] 
		joint_size = self.info_flow['dataset']['outputs']['joint_pos']
		pose_size = 3
		# rgbd_size = self.info_flow['dataset']['outputs']['rgbd']

		action_dim =self.info_flow['dataset']['outputs']['action']
		z_dim = cfg["model_params"]["z_dim"]
		min_steps = cfg["model_params"]["min_steps"]
		num_options = cfg["model_params"]["num_options"]

		self.batch_size = batch_size

		self.curriculum = cfg['training_params']['curriculum']

		if len(self.curriculum) == 0:
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
		# self.model_dict["Options_ClassifierTransformer"] = Options_ClassifierTransformer(models_folder, "Options_ClassifierTransformer", self.info_flow,\
		#  force_size, proprio_size, action_dim, num_options, min_steps, device = device).to(device)
		self.model_dict["PosErr_DetectionTransformer"] = PosErr_DetectionTransformer(models_folder, "PosErr_DetectionTransformer", self.info_flow,\
		 force_size, proprio_size, action_dim, num_options, min_steps, device = device).to(device)
		# self.model_dict["Options_PredictionResNet"] = Options_PredictionResNet(models_folder, "Options_PredictionResNet", self.info_flow,\
		#  pose_size, num_options, device = device).to(device)
		# self.model_dict["PosErr_PredictionResNet"] = PosErr_PredictionResNet(models_folder, "PosErr_PredictionResNet", self.info_flow,\
		#  pose_size, num_options, device = device).to(device)
		print("Finished Initialization")

		if parallel_bool == 1:
			for key in self.model_dict.keys():
				self.model_dict[key].set_parallel(True)
				print(key, "is now parallelizable between GPUs")
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
		# nn.MSELoss()
		# nn.CrossEntropyLoss()
		# nn.BCEWithLogitsLoss()
		# nn.CrossEntropyLoss(reduction = 'none')
		# nn.BCEWithLogitsLoss(reduction = 'none')

		###############################################
		##### Declaring loss functions for every term in the training objective
		##############################################
		self.loss_dict = {}
		self.loss_dict["Image_multistep"] = Proto_MultiStep_Loss(record_function = record_image)
		self.loss_dict["MSE_multistep"] = Proto_MultiStep_Loss(record_function = record_diff)
		self.loss_dict["L1_multistep"] = Proto_MultiStep_Loss(loss_function = nn.L1Loss(), record_function = record_diff)
		self.loss_dict["Gaussian_KL_multistep"] = Gaussian_KL_MultiStep_Loss()
		self.loss_dict["Gaussian_KL"] = Gaussian_KL_Loss()
		self.loss_dict["MSE"] = Proto_Loss()
		self.loss_dict["L1"] = Proto_Loss(loss_function = nn.L1Loss())
		self.loss_dict["Image_loss"] = Proto_Loss(record_function = record_image)
		self.loss_dict["BCE_multistep"] = BinaryEst_MultiStep_Loss()
		self.loss_dict["CE_multistep"] = CrossEnt_MultiStep_Loss()
		self.loss_dict["CE"] = CrossEnt_Loss()
		self.loss_dict["GaussNegLogProb_multistep"] = GaussianNegLogProb_multistep_Loss()
		self.loss_dict["Mult_KL_multistep"] = Multinomial_KL_MultiStep_Loss()
		self.loss_dict["Mult_KL"] = Multinomial_KL_Loss()
		self.loss_dict["Mag_multistep"] = Proto_MultiStep_Loss(loss_function = nn.L1Loss(), record_function = record_mag)
		self.loss_dict["Angle_multistep"] = Proto_MultiStep_Loss(record_function = record_angle)
		self.loss_dict["Contrastive"] = Contrastive_Loss()
		self.loss_dict["Ranking_Loss"] = Ranking_Loss()
		self.loss_dict["CE_ensemble"] = CrossEnt_Ensemble_Loss()
		self.loss_dict["Multivariate_Normal_Logprob"] = Multivariate_GaussianNegLogProb_Loss()
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
			if self.info_flow[key]["train"] == 1:
				self.model_dict[key].train()
			else:
				self.model_dict[key].eval()

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
			# if self.info_flow[model_key]["train"] == 0:
			# 	continue
			for idx_output, output_key in enumerate(self.info_flow[model_key]['outputs'].keys()):
				if self.info_flow[model_key]['outputs'][output_key]['loss'] == "":
					loss_idx += 1
					continue

				input_list = [self.model_outputs[model_key][output_key]]

				if 'inputs' in list(self.info_flow[model_key]['outputs'][output_key].keys()):
					for input_key in self.info_flow[model_key]['outputs'][output_key]['inputs'].keys():
						input_source = self.info_flow[model_key]['outputs'][output_key]['inputs'][input_key]
						if input_source == "":
							input_source = model_key
							
						input_list.append(self.model_outputs[input_source][input_key])

				loss_function = self.loss_dict[self.info_flow[model_key]['outputs'][output_key]['loss']]
				loss_name = self.info_flow[model_key]["outputs"][output_key]["loss_name"]

				if loss_bool == False:
					if "offset" in self.info_flow[model_key]["outputs"][output_key].keys():
						offset = self.info_flow[model_key]["outputs"][output_key]["offset"]
						loss = loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name, offset)
					else:
						loss = loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name)
					loss_bool = True
				else:
					if "offset" in self.info_flow[model_key]["outputs"][output_key].keys():
						offset = self.info_flow[model_key]["outputs"][output_key]["offset"]
						loss += loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name, offset)
					else:
						loss += loss_function.loss(tuple(input_list), self.logging_dict, self.info_flow[model_key]['outputs'][output_key]['weight'], model_key + "/" + loss_name)

		return loss

	def load(self, path_dict = {}):
		for key in self.model_dict.keys():
			self.model_dict[key].load(path_dict)

	def save(self, epoch_num):
		for key in self.model_dict.keys():
			self.model_dict[key].save(epoch_num)		



