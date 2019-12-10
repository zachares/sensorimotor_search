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
	def __init__(self, loss_function):
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

class Relational_Multistep(Proto_Loss):

	def __init__(self):
	    super().__init__(None)

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		z_list, p_list = params
		loss_L2 = torch.zeros_like(z_list[0])
		loss_reldist = torch.zeros_like(z_list[0])
		loss_reldir = torch.zeros_like(z_list[0])

		for idx in range(2,len(z_list)):
			z_0 = z_list[idx - 2]
			z_1 = z_list[idx - 1]
			z_2 = z_list[idx - 0]

			p_0 = p_list[idx - 2]
			p_1 = p_list[idx - 1]
			p_2 = p_list[idx - 0]

			p_02 = p_2 - p_0
			p_02_test = torch.where(p_2 == p_0, torch.zeros_like(p_2), torch.ones_like(p_2)).sum(1)
			p_02_mask = torch.where(p_02_test == 0, torch.zeros_like(p_02_test), torch.ones_like(p_02_test))

			p_01 = p_1 - p_0
			p_01_test = torch.where(p_1 == p_0, torch.zeros_like(p_1), torch.ones_like(p_1)).sum(1)
			p_01_mask = torch.where(p_01_test == 0, torch.zeros_like(p_01_test), torch.ones_like(p_01_test))

			p_dot = (p_01 * p_02).sum(dim=1) 
			p_01_norm = p_01.norm(p = 2, dim = 1) + (1 - p_01_mask)
			p_02_norm = p_02.norm(p = 2, dim = 1) + (1 - p_02_mask)

			z_02 = z_2 - z_0
			z_02_test = torch.where(z_2 == z_0, torch.zeros_like(z_2), torch.ones_like(z_2)).sum(1)
			z_02_mask = torch.where(z_02_test == 0, torch.zeros_like(z_02_test), torch.ones_like(z_02_test))

			z_01 = z_1 - z_0
			z_01_test = torch.where(z_1 == z_0, torch.zeros_like(z_1), torch.ones_like(z_1)).sum(1)
			z_01_mask = torch.where(z_01_test == 0, torch.zeros_like(z_01_test), torch.ones_like(z_01_test))

			z_dot = (z_01 * z_02).sum(dim=1) 
			z_01_norm = z_01.norm(p = 2, dim = 1) + (1 - z_01_mask)
			z_02_norm = z_02.norm(p = 2, dim = 1) + (1 - z_02_mask)

			mask = p_02_mask * p_01_mask * z_01_mask * z_02_mask

			### L2 regularization to make latent space more compact
			loss_L2[idx] = 1e-4 * weight * (z_0.pow(2).sum() + z_1.pow(2).sum() + z_2.pow(2).sum())

			### training objective to the relative magnitude between the points similar
			loss_reldist[idx] = 1e-5 * weight * (mask * (z_01_norm / z_02_norm - p_01_norm / p_02_norm)).pow(2).sum()

			### training objective to make the direction between progress points and latent points similar
			loss_reldir[idx] = weight * (mask * (z_dot / (z_01_norm * z_02_norm) - p_dot / (p_01_norm * p_02_norm))).pow(2).sum()

		loss = loss_L2.sum() + loss_reldist.sum() + loss_reldir.sum()
		# print(label, loss)

		logging_dict['scalar']["loss/" + label + "_L2"] = loss_L2.sum().item()
		logging_dict['scalar']["loss/" + label + "_reldist"] = loss_reldist.sum().item()
		logging_dict['scalar']["loss/" + label + "_reldir"] = loss_reldir.sum().item()

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

			element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

			if idx == 0:
				loss = weight * element_wise.sum(1).sum(0)
			else:
				loss += weight * element_wise.sum(1).sum(0)

		logging_dict['scalar']["loss/" + label] = loss.item()
		
		return loss
		
