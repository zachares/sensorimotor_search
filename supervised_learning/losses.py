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

class Proto_Loss(object):
	def __init__(self, loss_function = None):
		self.loss_function = loss_function

	def loss(self, input_tuple, logging_dict, weight, label):
		net_est = input_tuple[0]
		target = input_tuple[1]

		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Proto_MultiStep_Loss(object):
	def __init__(self, loss_function, max_idx = -1, offset = 1):
		self.loss_function = loss_function
		self.max_idx = max_idx
		self.offset = offset

	def loss(self, input_tuple, logging_dict, weight, label):
		net_est_list = input_tuple[0]
		if self.max_idx == -1:
			targets = input_tuple[1]
		else:
			targets = input_tuple[1][:,:,:self.max_idx]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + self.offset]
			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Proto_MultiStep_Hist_Loss(object):
	def __init__(self, loss_function, max_idx = -1, offset = 1):
		self.loss_function = loss_function
		self.max_idx = max_idx
		self.offset = offset

	def loss(self, input_tuple, logging_dict, weight, label):
		net_est_list = input_tuple[0]
		if self.max_idx == -1:
			targets = input_tuple[1]
		else:
			targets = input_tuple[1][:,:,:self.max_idx]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + self.offset]
			errors = self.loss_function(net_est, target)

			mu_err = errors.mean()
			std_err = errors.std()

			num_batch = errors.size() * torch.max(0.1, torch.sigmoid(torch.log(std_err / mu_err)))

			print(num_batch)

			errors_sorted, indices = torch.sort(errors)

			if idx == 0:
				loss = weight * errors_sorted[-num_batch:]
			else:
				loss += weight * errors_sorted[-num_batch:]

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Proto_MultiStep_Loss_List(object):
	def __init__(self, loss_function, max_idx = -1, offset = 1):
		self.loss_function = loss_function
		self.max_idx = max_idx
		self.offset = offset
		
	def loss(self, input_tuple, logging_dict, weight, label):
		net_est_list = input_tuple[0]
		if self.max_idx == -1:
			targets = input_tuple[1]
		else:
			targets = input_tuple[1][:,:,:self.max_idx]

		for idx, net_est in enumerate(net_est_list):
			target = targets[idx + self.offset]
			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss


class Image_Reconstruction(Proto_Loss):
	def loss(self, input_tuple, logging_dict, weight, label):
		net_est = input_tuple[0]
		target = input_tuple[1]

		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['image'][label] = []		
		logging_dict['image'][label].append(net_est[0])
		logging_dict['image'][label].append(target[0])

		return loss

class Gaussian_KL(Proto_Loss):

	def __init__(self):
	    super().__init__(None)

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		mu_est, var_est, mu_tar, var_tar = params

		element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

		loss = weight * element_wise.sum(1).sum(0)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Prior_Multistep(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__(None)

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		z_list, p_list = params

		for idx in range(len(z_list)):
			z = z_list[idx]
			p = p_list[idx]

			z_dist = (z.unsqueeze(2).repeat(1,1,z.size(0)) - z.unsqueeze(2).repeat(1,1,z.size(0)).transpose(0,2)).norm(p=2, dim = 1)

			p_dist = (p.unsqueeze(2).repeat(1,1,p.size(0)) - p.unsqueeze(2).repeat(1,1,p.size(0)).transpose(0,2)).norm(p=2, dim = 1)

			element_wise = (z_dist - p_dist).norm(p=2, dim = 1)

			if idx == 0:
				loss = weight * element_wise.sum()
			else:
				loss += weight * element_wise.sum()  

		# print(label, loss)
		logging_dict['scalar']["loss/" + label] = torch.tensor([loss]).detach().item()

		return loss

class Image_Reconstruction_MultiStep(Proto_MultiStep_Loss):
	def loss(self, input_tuple, logging_dict, weight, label):
		net_est_list = input_tuple[0]
		targets = input_tuple[1]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + self.offset]

			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['image'][label] = []		
		logging_dict['image'][label].append(net_est[0])
		logging_dict['image'][label].append(target[0])

		# if self.offset == 1:
		# 	print(loss)

		return loss

class BinaryEst_MultiStep(Proto_MultiStep_Loss):
	def loss(self, input_tuple, logging_dict, weight, label):
		inputs_list = input_tuple[0]

		for idx, input_element in enumerate(inputs_list):
			logits = input_element[0]
			labels = input_element[1]
			probs = torch.sigmoid(logits)
			samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
			accuracy = torch.where(samples == labels, torch.ones_like(probs), torch.zeros_like(probs))
			# if accuracy.mean() != 0.5:
			# 	print(accuracy.mean())
			# print("Accuracy: ", accuracy.mean())

			if idx == 0:
				loss = weight * self.loss_function(logits, labels)
				accuracy_rate = accuracy.mean()
			else:
				loss += weight * self.loss_function(logits, labels)
				accuracy_rate += accuracy.mean()
		
		# print(accuracy_rate)
		accuracy_rate /= len(inputs_list)

		# # print(accuracy_rate)
		# a = input(" ")

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/' + label] = accuracy_rate.item()	

		return loss

class Gaussian_KL_MultiStep(Proto_Loss):

	def __init__(self):
	    super().__init__(None)

	def loss(self, input_tuple, logging_dict, weight, label):
		params_list = input_tuple[0]

		for idx, params in enumerate(params_list):
			mu_est, var_est, mu_tar, var_tar = params

			# print("Idx: ", idx)
			# print(mu_est.size())
			# print(var_est.size())
			# print("")
			# print(mu_tar.size())
			# print(var_tar.size())

			element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

			if idx == 0:
				loss = weight * element_wise.sum()
			else:
				loss += weight * element_wise.sum()

		logging_dict['scalar']["loss/" + label] = torch.tensor([loss]).detach().item()
		
		return loss
		
