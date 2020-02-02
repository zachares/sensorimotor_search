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

def record_image(net_est, target, label, logging_dict):
	logging_dict['image'][label] = []		
	logging_dict['image'][label].append(net_est[0])
	logging_dict['image'][label].append(target[0])

def record_angle(net_est, target, label, logging_dict):
	ang = torch.abs(torch.acos((net_est * target).sum(1) / (net_est.norm(p=2, dim =1) * target.norm(p=2, dim=1))))
	angle = (ang * 180 / np.pi)
	logging_dict['scalar']['accuracy/anglemean_err_' + label] = angle.mean().item()
	logging_dict['scalar']['accuracy/anglevar_err_' + label] = angle.var().item()

def record_mag(mag_est, mag_tar, label, logging_dict):
	magnitude_error = (1 - (torch.abs(mag_est - mag_tar) / mag_tar))
	logging_dict['scalar']['mean/mag_err_' + label] = magnitude_error.mean().item()
	logging_dict['scalar']['var/mag_err_' + label] = magnitude_error.var().item()


def record_diff(est, tar, prev_tar, label, logging_dict):
	tar_error = (1 - ((est - tar).norm(p=2, dim=1) / (tar).norm(p=2, dim=1)))
	diff_error = (1 - ((est - tar).norm(p=2, dim=1) / (tar - prev_tar).norm(p=2, dim=1)))
	logging_dict['scalar']['acc_mean/err_' + label] = tar_error.mean().item()
	logging_dict['scalar']['var/err_' + label] = tar_error.var().item()
	logging_dict['scalar']['z/differr_' + label] = diff_error.mean().item()
	
def log_normal(x, m, v):
    return -0.5 * ((x - m).pow(2)/ v + torch.log(2 * np.pi * v)).sum(-1).unsqueeze(-1)

def multinomial_KL(p, logq):
	p_soft = p + 1e-8
	p_soft_norm = p_soft / p_soft.sum(1).unsqueeze(1).repeat(1,p_soft.size(1))
	return - (p_soft_norm * (logq - torch.log(p_soft_norm))).sum(1).mean()

def gaussian_KL(mu_est, var_est, mu_tar, var_tar):
	return 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1).sum(1).mean()

class Proto_Loss(object):
	def __init__(self, loss_function = nn.MSELoss(), transform_target_function = None, record_function = None):
		self.loss_function = loss_function
		self.transform_target_function = transform_target_function
		self.record_function = record_function

	def loss(self, input_tuple, logging_dict, weight, label):
		net_est = input_tuple[0]
		if self.transform_target_function is not None:
			target = self.transform_target_function(input_tuple[1])
		else:
			target = input_tuple[1]

		loss = weight * self.loss_function(net_est, target)

		if self.record_function is not None:
			self.record_function(net_est, target, logging_dict)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Proto_MultiStep_Loss(object):
	def __init__(self, loss_function = nn.MSELoss(), transform_target_function = None, record_function = None):
		self.loss_function = loss_function
		self.transform_target_function = transform_target_function
		self.record_function = record_function

	def loss(self, input_tuple, logging_dict, weight, label, offset = 0):
		net_est_list = input_tuple[0]

		if self.transform_target_function is not None:
			targets = self.transform_target_function(input_tuple[1])
		else:
			targets = input_tuple[1]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:,idx + offset]
			prev_target = targets[:,idx + offset - 1]

			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

			if self.record_function is not None:
				self.record_function(net_est, target, prev_target, label, logging_dict)

		logging_dict['scalar']["loss/" + label] = loss.item() / len(net_est_list)

		return loss

class GaussianNegLogProb_multistep_Loss(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label, offset):
		logits_list = input_tuple[0]
		labels_array = input_tuple[1]

		for idx, logits in enumerate(logits_list):
			labels = labels_array[:,idx + offset]
			means, varis = logits

			mean_error_mag = 1 - ((labels - means).norm(p=2, dim = 1) / (labels).norm(p=2, dim =1)).mean()				

			if idx == 0:
				loss = -1.0 * weight * log_normal(labels, means, varis).sum(1).mean()
				mean_error = mean_error_mag.item()
			else:
				loss += -1.0 * weight * log_normal(labels, means, varis).sum(1).mean()
				mean_error = mean_error_mag.item()

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/average_error/' + label] = mean_error_mag / len(logits_list)
		logging_dict['scalar']['accuracy/max_variance/' + label] = torch.abs(varis).max().item()

		return loss

class Multinomial_KL_MultiStep_Loss(Proto_MultiStep_Loss):

	def __init__(self, record_function = None):
	    super().__init__(record_function = record_function)
	    self.logsoftmax = nn.LogSoftmax(dim = 1)

	def loss(self, input_tuple, logging_dict, weight, label, offset):
		logits_list = input_tuple[0]
		targets = input_tuple[1]

		for idx, logits in enumerate(logits_list):
			log_est = self.logsoftmax(logits)
			target = targets[:,idx + offset]

			if idx == 0:
				loss = weight * multinomial_KL(log_est, target)
			else:
				loss += weight * multinomial_KL(log_est, target)

			if self.record_function is not None:
				self.record_function(torch.exp(log_est), target, label, logging_dict)

		logging_dict['scalar']["loss/" + label] = loss.item() / len(logits_list)
		
		return loss

class BinaryEst_MultiStep_Loss(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__(loss_function = nn.BCEWithLogitsLoss())

	def loss(self, input_tuple, logging_dict, weight, label, offset):
		logits_list = input_tuple[0]
		labels_array = input_tuple[1]

		for idx, logits in enumerate(logits_list):
			labels = labels_array[:,idx + offset]

			probs = torch.sigmoid(logits).squeeze()
			samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
			accuracy = torch.where(samples == labels, torch.ones_like(probs), torch.zeros_like(probs))

			if idx == 0:
				loss = weight * self.loss_function(logits, labels)
				accuracy_rate = accuracy.mean()
			else:
				loss += weight * self.loss_function(logits, labels)
				accuracy_rate += accuracy.mean()

		accuracy_rate /= len(logits_list)

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/' + label] = accuracy_rate.item()	

		return loss / len(logits_list)

class CrossEnt_MultiStep_Loss(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__(loss_function = nn.CrossEntropyLoss())
	    self.softmax = nn.Softmax(dim=1)

	def loss(self, input_tuple, logging_dict, weight, label, offset):
		logits_list = input_tuple[0]
		labels_array = input_tuple[1]

		for idx, logits in enumerate(logits_list):
			labels = labels_array[:,idx + offset]

			probs = self.softmax(logits)
			samples = torch.zeros_like(labels)
			samples[torch.arange(samples.size(0)), probs.max(1)[1]] = 1.0
			test = torch.where(samples == labels, torch.zeros_like(probs), torch.ones_like(probs)).sum(1)
			accuracy = torch.where(test > 0, torch.zeros_like(test), torch.ones_like(test))

			if idx == 0:
				loss = weight * self.loss_function(logits, labels.max(1)[1])
				accuracy_rate = accuracy.mean()
			else:
				loss += weight * self.loss_function(logits, labels.max(1)[1])
				accuracy_rate += accuracy.mean()

		accuracy_rate /= len(logits_list)

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/' + label] = accuracy.mean().item()	

		return loss

class Histogram_MultiStep_Loss(object):
	def __init__(self, loss_function, transform_target_function = None, record_function = None, hyperparameter = 1.0):
		self.loss_function = loss_function
		self.transform_target_function = transform_target_function
		self.record_function = record_function
		self.hyperparameter = hyperparameter

	def loss(self, input_tuple, logging_dict, weight, label, offset = 0):
		net_est_list = input_tuple[0]

		if self.transform_target_function is not None:
			targets = self.transform_target_function(input_tuple[1])
		else:
			targets = input_tuple[1]

		for idx, net_est in enumerate(net_est_list):
			target = targets[:, idx + self.offset]
			errors = self.loss_function(net_est, target).sum(1)

			mu_err = errors.mean()
			std_err = errors.std()

			if 1 - torch.sigmoid(torch.log(self.hyperparameter * std_err / mu_err)) < 0.1:
				num_batch = np.round(0.1 * errors.size(0)).astype(np.int32)
			else:
				num_batch = torch.round((1 - torch.sigmoid(torch.log( self.hyperparameter * std_err / mu_err))) * errors.size(0)).type(torch.int32)

			errors_sorted, indices = torch.sort(errors)

			if idx == 0:
				loss = weight * errors_sorted[-num_batch:].mean()
			else:
				loss += weight * errors_sorted[-num_batch:].mean()

			if self.record_function is not None:
				self.record_function(net_est, target, label, logging_dict)

		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss

class Distance_Multistep_Loss(Proto_MultiStep_Loss):
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
				
		logging_dict['scalar']["loss/" + label] = torch.tensor([loss]).detach().item()

		return loss
		
class Gaussian_KL_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		mu_est, var_est, mu_tar, var_tar = params

		loss = weight * gaussian_KL(mu_est, var_est, mu_tar, var_tar)

		logging_dict['scalar']["loss/" + label] = loss.item()

		if self.record_function is not None:
			self.record_function((mu_est, var_est), (mu_tar, var_tar), label, logging_dict)

		return loss
		
class Gaussian_KL_MultiStep_Loss(Proto_Loss):

	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		params_list = input_tuple[0]

		for idx, params in enumerate(params_list):
			mu_est, var_est, mu_tar, var_tar = params
			if idx == 0:
				loss = weight * gaussian_KL(mu_est, var_est, mu_tar, var_tar)
			else:
				loss += weight * gaussian_KL(mu_est, var_est, mu_tar, var_tar)

			if self.record_function is not None:
				self.record_function((mu_est, var_est), (mu_tar, var_tar), label, logging_dict)

		logging_dict['scalar']["loss/" + label] = torch.tensor([loss]).detach().item()
		
		return loss