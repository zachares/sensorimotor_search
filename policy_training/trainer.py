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
class Trainer(object):

	def __init__(self, cfg_0, cfg_1, cfg_2, device):

		self.device = device

		cols = cfg_0['game_params']['cols']
		rows = cfg_0['game_params']['rows']
		pose_dim = cfg_0['game_params']['pose_dim']

		z_dim = cfg_2["model_params"]["model_1"]["z_dim"]

		batch_size = cfg_2['dataloading_params']['batch_size']
		self.batch_size = batch_size

		regularization_weight = cfg_2['training_params']['regularization_weight']
		learning_rate = cfg_2['training_params']['lrn_rate']
		beta_1 = cfg_2['training_params']['beta1']
		beta_2 = cfg_2['training_params']['beta2']

		### Initializing Model ####
		print("Initializing Neural Network Models")
		self.model_dict = {}
		self.model_input_keys = {}

		##### model name ##########
		### Specify model names ####
		self.model_dict[cfg_2["model_params"]["model_1"]["name"]] = Observations_Encoder(rows, cols, 1,  z_dim, device = device).to(device)
		self.model_input_keys[cfg_2["model_params"]["model_1"]["name"]] = cfg_2["model_params"]["model_1"]["input_keys"]

		### Loading Model Checkpoint ####
		for key in cfg_2["model_params"].keys():
			model_path = cfg_2['model_params'][key]['path']
			if model_path != "":
			    print('Loading model from {}...'.format(model_path))
			    ckpt1 = torch.load(model_path)
			    self.model_dict[cfg_2["model_params"][key]["name"]].load_state_dict(ckpt1)

		#### Setting per step model update method ####
		# Adam is a type of stochastic gradient descent    

		parameters_list = []

		self.bounds = torch.ones(2).to(self.device).requires_grad_(True)
# self.bounds, 
		##### logit size must always be an odd number
		self.modes = torch.ones(12).to(self.device).requires_grad_(True)

		# for key in self.model_dict.keys():
		# 	parameters_list += list(self.model_dict[key].parameters())

		self.optimizer = optim.Adam( [self.bounds, self.modes], lr= learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
		# self.optimizer = optim.Adam(list(self.obs_enc.parameters()) + list(self.dyn_mod.parameters()) + list(self.goal_prov.parameters()),\
		#  lr=learning_rate, betas=(beta_1, beta_2), weight_decay = regularization_weight)
		##### Defining Loss Function #####

		self.mse_loss = nn.MSELoss()
		self.crent_loss = nn.CrossEntropyLoss(reduction = 'none')
		self.bcrent_loss = nn.BCEWithLogitsLoss(reduction = 'none')

		######### variables requiring grad #########
		self.action = torch.zeros(batch_size, 1, rows,cols).to(self.device)

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

		points = sample_batched['points']
		params = sample_batched['params']
		samples = sample_batched['samples']

		self.model_inputs_dict = {}

		for model_key in self.model_input_keys.keys():
			
			self.model_inputs_dict[model_key] = []

			for input_key in self.model_input_keys[model_key]:

				if input_key == "obs_t0":
					self.model_inputs_dict[model_key].append(sample_batched[input_key].to(self.device).unsqueeze(1))
				else:
					self.model_inputs_dict[model_key].append(sample_batched[input_key].to(self.device))

		self.model_outputs_dict = {}

		for key in self.model_dict.keys():
			self.model_outputs_dict[key] = self.model_dict[key](self.model_inputs_dict[key])

		self.loss_targets = (samples.to(self.device), params.to(self.device))

		return self.calc_loss()

	def calc_loss(self):

		samples, params = self.loss_targets

		samples = samples.squeeze()

		# self.bounds = torch.tensor([-0.0001, 1.5001]).to(self.device)

		modes = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * (2 * (torch.sigmoid(torch.abs(self.modes)) - 0.5))

		bounds = self.bounds.to(self.device).unsqueeze(0).repeat(samples.size()[0], 1) #self.bounds.torch.tensor([-0.001, 0.8501])
		# bounds = torch.tensor([-0.0001, 1.5001]).to(self.device).unsqueeze(0).repeat(samples.size()[0], 1) #self.bounds.

		one = torch.ones_like(samples)

		neg_one = -torch.ones_like(samples)

		zero = torch.zeros_like(samples)

		lb_weight = torch.where(bounds[:,0] > samples, one, zero).sum() # need to think about accidentally initializing in the wrong direction
		ub_weight = torch.where(bounds[:,1] < samples, neg_one, zero).sum() # need to think about accidentally initializing in the wrong direction

		wb_weights = (torch.where((bounds[:,1] > samples), one, zero) * torch.where(bounds[:,0] < samples, one, zero))

		loss = lb_weight * self.bounds[0] + ub_weight * self.bounds[1]

		### way to measure how often it appears between samples or within a sample

		if wb_weights.sum() > 1:

			# min distance

			new_samples = samples[(wb_weights * samples).nonzero()].squeeze() # new_samples[new_samples.nonzero()]
			sorted_samples, indices = torch.sort(samples[(wb_weights * samples).nonzero()].squeeze())

			diff_samples = sorted_samples[1:] - sorted_samples[:-1]

			if diff_samples.size()[0] < modes.size()[0] / 2:
				range_max = int(diff_samples.size()[0] - (diff_samples.size()[0] % 2))
			else:
				range_max = int(modes.size()[0] // 2)

			# print(diff_samples)
			sorted_diff_samples, max_idxs = torch.sort(diff_samples)

			mode_loss = torch.tensor([0]).float().to(self.device)

			# print(sorted_samples.size())

			for idx in range(0,range_max):

				loss += torch.abs(modes[2 * idx] - sorted_samples[max_idxs[-(idx + 1)]]) +  torch.abs(modes[2 * idx + 1] - sorted_samples[max_idxs[-(idx + 1)] + 1])

			####the loss below only works with a torch.abs regularization, no mode collapse has yet been observed

		self.current_params = modes.tolist()


		# print("Difference in iteration")
		# print(sorted_samples[idx_max_diff].item())

		# print(sorted_samples[idx_max_diff + 1].item())

		# print("Mode prediction")
		# print(mode_0[0])
		# print(mode_0[1])

		# print("interval_sizes")
		# print((self.interval_sizes[0,:] - self.prev_interval_sizes)[0,:])
		# print(" ###########")
		# print("Samples")
		# print(samples.squeeze())
		# print("pred params")
		# print(self.current_params)
		# # print("params")
		# # print(params[0,:])
		# # print("interval_update_weights")
		# # print(interval_update_weights)

		# # print("Full Bounds Indexed")
		# # print(test_values)

		# # # print(full_bounds[np.arange(full_bounds.size()[0]), argmin_sq_bounds])
		# # # print("Full Bounds")
		# # # print(full_bounds)
		# print("#########################")
		# print("interval_update_weights")
		# print(interval_update_weights.squeeze())
		# print("argmin_sq_bounds")
		# print(argmin_sq_bounds)
		# for idx in range(6):
		# 	print(sorted_samples[max_idxs[-(idx + 1)]].item(), "    ", sorted_samples[max_idxs[-(idx + 1)] + 1].item()) 


		# a = input("Continue?")

		# self.scalar_dict['loss/loss'] = loss.item()
		# self.scalar_dict['metric/average_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).mean().item()
		# self.scalar_dict['metric/min_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).max().item()
		# self.scalar_dict['metric/max_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).min().item()

		return loss


	def output_goal(self, obs_t0):

		return torch.sigmoid(self.obs_enc_goal_image(obs_t0))

	def load_model(self, path):

		## Loading Model Checkpoint ####
	    print('Loading model from {}...'.format(path))
	    ckpt1 = torch.load(path)
	    self.model_dict["observation_encoder"].load_state_dict(ckpt1)

# 		# probs = (torch.abs(logits).transpose(0,1) / torch.abs(logits).sum(1)).transpose(0,1)

# 		probs = 2 * (torch.sigmoid(torch.abs(logits)) - 0.5)

# 		# probs = F.softmax(logits, dim = 1)

# 		# old_probs = torch.clone(probs)

# 		# for idx in range(probs.size()[1]):

# 		# 	probs[:,idx] = old_probs[:,:idx + 1].sum(1)

# 		# print("Probs size: ", probs.size())

# 		full_bounds = (bounds[:,1].unsqueeze(0) - probs.transpose(0,1) * (bounds[:,1].unsqueeze(0)  - bounds[:,0].unsqueeze(0))).transpose(0,1)

# 		ico_bool = False
# 		ico_list = [-1]

# 		for idx in range(full_bounds.size()[1] - 1):

# 			if probs[0,idx] > probs[0, idx + 1] and ico_bool == False:

# 				ico_list.append(-1)

# 			else:

# 				ico_bool = True
# 				ico_list.append(1)

# 		if ico_bool:
# 			ico_weights = torch.tensor(ico_list).to(self.device).float().unsqueeze(0).repeat(samples.size()[0], 1)			
# 		else:
# 			ico_weights = torch.zeros_like(logits)

# 		prob_weights = torch.zeros_like(probs)

# 		if ico_bool == False:

# 			for idx in range(full_bounds.size()[1]):

# 				if idx == 0:
# 					lt_weights_pos = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, one, zero).repeat(1, probs.size()[1])				
# 					lt_weights = torch.clone(lt_weights_pos)

# 					gt_weights_pos =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, one, zero).repeat(1, probs.size()[1] - 1)
# 					gt_weights_neg =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, neg_one, zero)	
# 					gt_weights = torch.cat((gt_weights_neg, gt_weights_pos), dim = 1)

# 				elif idx == (probs.size()[1] - 1):
# 					lt_weights_neg = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, neg_one, zero).repeat(1, probs.size()[1])				
# 					lt_weights = torch.clone(lt_weights_neg)

# 					gt_weights_pos =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, one, zero).repeat(1, probs.size()[1] - 1)
# 					gt_weights_neg =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, neg_one, zero)	
# 					gt_weights = torch.cat((gt_weights_pos, gt_weights_neg), dim = 1)

# 				elif idx % 2 == 0:
# 					lt_weights_pos = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, one, zero).repeat(1, idx)
# 					lt_weights_neg = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, neg_one, zero).repeat(1, probs.size()[1] - idx)				
# 					lt_weights = torch.cat((lt_weights_pos, lt_weights_neg), dim = 1)

# 					gt_weights_pos =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, one, zero).repeat(1, probs.size()[1] - idx - 1)
# 					gt_weights_neg =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, neg_one, zero).repeat(1, idx + 1)	
# 					gt_weights = torch.cat((gt_weights_neg, gt_weights_pos), dim = 1)

# 				elif idx % 2 == 1:
# 					lt_weights_pos = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, one, zero).repeat(1, probs.size()[1] - idx - 1)
# 					lt_weights_neg = torch.where(full_bounds[:, idx].unsqueeze(1) > samples, neg_one, zero).repeat(1, idx + 1)				
# 					lt_weights = torch.cat((lt_weights_neg, lt_weights_pos), dim = 1)

# 					gt_weights_pos =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, one, zero).repeat(1, idx)
# 					gt_weights_neg =  torch.where(full_bounds[:, idx].unsqueeze(1) < samples, neg_one, zero).repeat(1, probs.size()[1] - idx)	
# 					gt_weights = torch.cat((gt_weights_pos, gt_weights_neg), dim = 1)

# 				# print("Logit weights size: ")
# 				# print(logit_weights.size())
# 				# print("Wb weights size: ")
# 				# print(wb_weights.size())
# 				# print("Lt weights size: ")
# 				# print(lt_weights.size())
# 				# print("full_bounds size: ")
# 				# print(full_bounds[:, idx] .size())

# 				prob_weights += wb_weights * (lt_weights + gt_weights)

# 		self.current_params = (bounds[0,1] - probs[0,:] * (bounds[0,1] - bounds[0,0])).tolist() 

# 		# print("Logit weights size: ", logit_weights.size())
# 		# print("Logit weights")
# 		# print(logit_weights)
# 		# print("Samples")
# 		# print(samples)
# 		# print("Current Params")
# 		# print(self.current_params)
# 		# a  = input("Continue?")

# 		loss = (lb_weight * bounds[:,0].unsqueeze(1) - ub_weight * bounds[:1].unsqueeze(1) + ((ico_weights + prob_weights) * logits).mean(1).unsqueeze(1)).mean()
# 		self.loss = loss.detach().item()

# 		self.current_params = (bounds[0,1] - probs[0,:] * (bounds[0,1] - bounds[0,0])).tolist() # (self.bounds[1] - probs[0,:] * (self.bounds[1] - self.bounds[0])).tolist()

# 		# sq_bounds = full_bounds ** 2

# 		# _, argmin_sq_bounds = sq_bounds.min(1)

# 		# test_values = full_bounds[np.arange(full_bounds.size()[0]), argmin_sq_bounds].unsqueeze(1)

# 		# argmin_sq_bounds = torch.cat((torch.tensor(range(self.batch_size)).to(self.device).unsqueeze(1), argmin_sq_bounds.unsqueeze(1)), dim = 1)





# 		# pos_probs_weights = torch.where((bounds[:,1].unsqueeze(1) > samples), one, zero) \
# 		# * torch.where(bounds[:,0].unsqueeze(1) < samples, one, zero) \
# 		# * ((torch.where(test_values > 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 0, one, zero))   \
# 		# + (torch.where(test_values < 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 1, one, zero))) \

# 		# neg_probs_weights = torch.where((bounds[:,1].unsqueeze(1) > samples), one, zero) \
# 		# * torch.where(bounds[:,0].unsqueeze(1) < samples, one, zero) \
# 		# * ((torch.where(test_values < 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 0, one, zero))   \
# 		# + (torch.where(test_values > 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 1, one, zero))) \
			
# 		# the loss below only works with a torch.abs regularization, no mode collapse has yet been observed

# 		# loss = (lb_weight * bounds[:,0].unsqueeze(1) - ub_weight * bounds[:1].unsqueeze(1) +\
# 		#  pos_probs_weights * logits[np.arange(logits.size()[0]), argmin_sq_bounds].unsqueeze(1) \
# 		#  - neg_probs_weights * logits[np.arange(logits.size()[0]), argmin_sq_bounds].unsqueeze(1)).mean()

# 		# the loss below only works with softmax regularization and mode collapse was observed during an experiment 
# 		# it may learn faster using crent though, may merit further experiments to determine this

# 		# loss = (lb_weight * bounds[:,0].unsqueeze(1) - ub_weight * bounds[:1].unsqueeze(1) +\ 
# 		#  pos_probs_weights * self.crent_loss(logits, argmin_sq_bounds).unsqueeze(1) \
# 		#  - neg_probs_weights * self.crent_loss(logits, argmin_sq_bounds).unsqueeze(1)).mean()

# 		# print("logits one way")
# 		# print(logits[:, argmin_sq_bounds.tolist()].mean(1))
# 		# print("logits the other way")
# 		# print(logits[np.arange(logits.size()[0]), argmin_sq_bounds])
# #(lb_weight * bounds[:,0].unsqueeze(1)).mean()
# 			# - ub_weight * bounds[:1].unsqueeze(1)\
# 			#  + probs_weights * self.crent_loss(logits, argmin_sq_bounds).unsqueeze(1)).mean()

# 		# print("Samples")
# 		# print(samples)
# 		# print("params")
# 		# print(self.current_params)
# 		# print("Full Bounds Indexed")
# 		# print(test_values)

# 		# # print(full_bounds[np.arange(full_bounds.size()[0]), argmin_sq_bounds])
# 		# # print("Full Bounds")
# 		# # print(full_bounds)
# 		# print("argmin_sq_bounds")
# 		# print(argmin_sq_bounds)
# 		# # print("Odd")
# 		# # print(torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 1, one, zero))
# 		# # print("Even")
# 		# # print(torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 0, one, zero))
# 		# # print("Index bounds")
# 		# # print(test_values)
# 		# # print("Negative")
# 		# # print(torch.where(test_values < 0, one, zero))
# 		# # print("Positive")
# 		# # print(torch.where(test_values < 0, one, zero))
# 		# # print("Prob weights")
# 		# # print(probs_weights)
# 		# print("Odd and Neg")
# 		# print((torch.where(test_values < 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 1, one, zero)))
# 		# print("Even and Pos")
# 		# print((torch.where(test_values > 0, one, zero) * torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 0, one, zero)))


# 		# a = input("Continue?")

# 		self.scalar_dict['loss/loss'] = loss.item()
# 		self.scalar_dict['metric/average_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).mean().item()
# 		self.scalar_dict['metric/min_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).max().item()
# 		self.scalar_dict['metric/max_error'] = (bounds[:,0].unsqueeze(0) + probs.transpose(0,1) * (bounds[:,1].unsqueeze(0) - bounds[:,0].unsqueeze(0)) - params.transpose(0,1)).min().item()

# # 		return loss

# 		# the loss below only works with softmax regularization and mode collapse was observed during an experiment 
# 		# it may learn faster using crent though, may merit further experiments to determine this

# 		# loss = (lb_weight * bounds[:,0].unsqueeze(1) - ub_weight * bounds[:1].unsqueeze(1) +\ 
# 		#  pos_probs_weights * self.crent_loss(logits, argmin_sq_bounds).unsqueeze(1) \
# 		#  - neg_probs_weights * self.crent_loss(logits, argmin_sq_bounds).unsqueeze(1)).mean()

# 		samples, params = self.loss_targets

# 		logits = self.logits.unsqueeze(0).repeat(samples.size()[0], 1)
# 		# bounds = self.bounds.to(self.device).unsqueeze(0).repeat(samples.size()[0], 1) #self.bounds.torch.tensor([-0.001, 0.8501])
# 		bounds = torch.tensor([-0.0001, 1.5001]).to(self.device).unsqueeze(0).repeat(samples.size()[0], 1) #self.bounds.

# 		one = torch.ones_like(samples)

# 		neg_one = - torch.ones_like(samples)

# 		zero = torch.zeros_like(samples)

# 		lb_weight = torch.where(bounds[:,0].unsqueeze(1) > samples, one, zero)
# 		ub_weight = torch.where(bounds[:,1].unsqueeze(1) < samples, neg_one, zero)
# 		wb_weights = (torch.where((bounds[:,1].unsqueeze(1) > samples), one, zero) * torch.where(bounds[:,0].unsqueeze(1) < samples, one, zero))


# 		######### normalizing weights

# 		probs = (torch.abs(logits).transpose(0,1) / torch.abs(logits).sum(1)).transpose(0,1)

# 		# probs = 2 * (torch.sigmoid(torch.abs(logits)) - 0.5)

# 		# probs = F.softmax(logits, dim = 1)

# 		self.prev_interval_sizes = torch.clone(self.interval_sizes)

# 		interval_sizes = torch.clone(probs)

# 		self.interval_sizes = torch.clone(interval_sizes)

# 		######### cumulative weights instead of interval sizes

# 		cum_probs = torch.zeros_like(probs)

# 		for idx in range(probs.size()[1]):

# 			cum_probs[:,idx] = interval_sizes[:,:idx + 1].sum(1)

# 		######## calculating closest interval boundary

# 		full_bounds = (samples - (bounds[:,0].unsqueeze(0) + cum_probs.transpose(0,1) * (bounds[:,1].unsqueeze(0)  - bounds[:,0].unsqueeze(0))).transpose(0,1))

# 		sq_bounds = full_bounds ** 2

# 		_, argmin_sq_bounds = sq_bounds.min(1)

# 		#	#### value of the difference between the sample and the closest interval boundary

# 		test_values = full_bounds[np.arange(full_bounds.size()[0]), argmin_sq_bounds].unsqueeze(1)

# 		# four test cases 
# 		# sample must be within boundary
# 		# left interval boundary and positive test value -> decrease interval && add 1 to the argmin index
# 		# left interval boundary and negative test value -> decrease interval && keep index the same
# 		# right interval boundary and positive test value -> decrease interval && add 1 to the argmin index
# 		# right interval boundary and negative test value -> decrease interval && keep index the same

# 		neg_test_values = torch.where(test_values < 0, one, zero)
# 		pos_test_values = torch.where(test_values > 0, one, zero)

# 		# an even index means a right interval boundary
# 		# an odd index means a left interval boundary

# 		right_interval_boundary = torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 0, one, zero)
# 		left_interval_boundary = torch.where(argmin_sq_bounds.unsqueeze(1) % 2 == 1, one, zero)

# 		interval_update_weights = wb_weights * ((neg_test_values * right_interval_boundary +  pos_test_values * left_interval_boundary)\
# 		 + (neg_test_values * left_interval_boundary + pos_test_values * right_interval_boundary))

# 		argmin_sq_bounds = argmin_sq_bounds.float() + pos_test_values.float().squeeze()

# 		####the loss below only works with a torch.abs regularization, no mode collapse has yet been observed

# 		self.current_params = (bounds[0,0] + cum_probs[0,:] * (bounds[0,1] - bounds[0,0])).tolist() 

# 		loss = (lb_weight * bounds[:,0] + ub_weight * bounds[:,1] + interval_update_weights * interval_sizes[np.arange(interval_sizes.size()[0]), argmin_sq_bounds.long()]).sum()
