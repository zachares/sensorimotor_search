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
	def __init__(self, loss_function, loss_name):
		self.loss_name = loss_name
		self.loss_function = loss_function

	def loss(self, input_tuple, logging_dict, weight, model_label):
		net_est = input_tuple[0]
		target = input_tuple[1]

		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()

		return loss

	def loss_metrics(self, input_tuple, logging_dict):
		pass

class Proto_MultiStep_Loss(object):
	def __init__(self, loss_function, loss_name, max_idx = -1):
		self.loss_name = loss_name
		self.loss_function = loss_function
		self.max_idx = max_idx

	def loss(self, input_tuple, logging_dict, weight, model_label):
		net_est_list = input_tuple[0]
		if self.max_idx == -1:
			targets = input_tuple[1]
		else:
			targets = input_tuple[1][:,:self.max_idx]		

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + 1]

			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()

		return loss

class Image_Reconstruction(Proto_Loss):
	def loss(self, input_tuple, logging_dict, weight, model_label):
		net_est = input_tuple[0]
		target = input_tuple[1]


		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()
		logging_dict['image'][model_label] = []		
		logging_dict['image'][model_label].append(net_est[0])
		logging_dict['image'][model_label].append(target[0])

		return loss

class Gaussian_KL(Proto_Loss):

	def __init__(self, loss_name):
	    super().__init__(None, loss_name)

	def loss(self, input_tuple, logging_dict, weight, model_label):
		params = input_tuple[0]
		mu_est, var_est, mu_tar, var_tar = params

		element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

		loss = weight * element_wise.sum(1).sum(0)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()

		return loss

class Image_Reconstruction_MultiStep(Proto_Loss):
	def loss(self, input_tuple, logging_dict, weight, model_label):
		net_est_list = input_tuple[0]
		targets = input_tuple[1]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + 1]

			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()
		logging_dict['image'][model_label] = []		
		logging_dict['image'][model_label].append(net_est[0])
		logging_dict['image'][model_label].append(target[0])

		return loss

class Pairing_MultiStep(Proto_MultiStep_Loss):
	def loss(self, input_tuple, logging_dict, weight, model_label):
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
		# print(accuracy_rate)

		# # print(accuracy_rate)
		# a = input(" ")

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()
		logging_dict['scalar']['accuracy/' + model_label + self.loss_name] = accuracy_rate.item()	

		return loss

class Gaussian_KL_MultiStep(Proto_Loss):

	def __init__(self, loss_name):
	    super().__init__(None, loss_name)

	def loss(self, input_tuple, logging_dict, weight, model_label):
		params_list = input_tuple[0]

		for idx, params in enumerate(params_list):
			mu_est, var_est, mu_tar, var_tar = params

			element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

			if idx == 0:
				loss = weight * element_wise.sum(1).sum(0)
			else:
				loss += weight * element_wise.sum(1).sum(0)

		logging_dict['scalar']["loss/" + model_label + self.loss_name] = loss.item()
		
		return loss
		
