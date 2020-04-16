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
	# logging_dict['scalar']['accuracy/anglevar_err_' + label] = angle.var().item()

def record_mag(mag_est, mag_tar, label, logging_dict):
	magnitude_error = torch.abs(mag_est.squeeze().unsqueeze(1) / mag_tar.squeeze().unsqueeze(1))
	logging_dict['scalar']['mean/rel_mag_' + label] = magnitude_error.mean().item()
	logging_dict['scalar']['var/mag_err_' + label] = magnitude_error.var().item()

def normed_MSEloss(net_est, target):
	if len(list(net_est.size())) == 1:
		return ((net_est / target) - 1).pow(2).mean()
	else:
		return ((net_est / target) - 1).pow(2).sum(1).mean()

def record_diff(est, tar, label, logging_dict):
	if len(list(est.size())) == 1:
		tar_error = (1 - ((est.unsqueeze(1) - tar.unsqueeze(1)).pow(2).sum(1).pow(0.5)  / tar.unsqueeze(1)).pow(2).sum(1).pow(0.5))
	else:
		tar_error = (1 - ((est - tar).pow(2).sum(1).pow(0.5) / tar.pow(2).sum(1).pow(0.5)))
		
	logging_dict['scalar']['acc_mean/err_' + label] = tar_error.mean().item()
	# logging_dict['scalar']['var/err_' + label] = tar_error.var().item()
	
def log_normal(x, m, v):
    return -0.5 * ((x - m).pow(2)/ v + 4 * torch.log(2 * np.pi * v)).sum(-1).unsqueeze(-1)

def det3(mats):
    return mats[:,0,0] * (mats[:,1,1] * mats[:,2,2] - mats[:,1,2] *mats[:,2,1]) -\
     mats[:,0,1] * (mats[:,1,0] * mats[:,2,2] - mats[:,2,0] * mats[:,1,2]) +\
      mats[:,0,2] * (mats[:,1,0] * mats[:,2,1] - mats[:,1,1] * mats[:,2,0])

def multiv_gauss_logprob(samples, means, prec):
    # means dim - batch x samples
    # samples dim - batch x samples
    # var dim - batch x samples x samples
    prec_det = det3(prec)
    log_prob_const = 0.5 * torch.log(prec_det)
    err = (samples - means).unsqueeze(1) 
    log_prob_sample = -0.5 * torch.bmm(torch.bmm(err, prec), err.transpose(1,2))
    return log_prob_const + log_prob_sample

def multiv_gauss_KL(mean1, mean2, prec1, prec2):

	prec1_det = det3(prec1)
	prec2_det = det3(prec2)
	prec1_inv = torch.inverse(prec1)
	prec_mult = torch.bmm(prec2, prec1_inv)
	prec_mult_trace = torch.diagonal(prec_mult, dim1=1, dim2=2).sum(1)
	mean_error = (mean2 - mean1).unsqueeze(2)

	return 0.5 * (torch.log(prec1_det/prec2_det) + prec_mult_trace + torch.bmm(torch.bmm(mean_error.transpose(1,2), prec2), mean_error).squeeze())

def multinomial_KL(logits_q, logits_p):

	return -(F.softmax(logits_p, dim =1) * (F.log_softmax(logits_q, dim = 1) - F.log_softmax(logits_p, dim = 1))).sum(1).mean()

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

			if idx == 0:
				loss = weight * self.loss_function(net_est, target)
			else:
				loss += weight * self.loss_function(net_est, target)

			if self.record_function is not None:
				self.record_function(net_est, target, label, logging_dict)

		logging_dict['scalar']["loss/" + label] = loss.item() / len(net_est_list)

		return loss

class CrossEnt_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__(loss_function = nn.CrossEntropyLoss())
	    self.softmax = nn.Softmax(dim=1)

	def loss(self, input_tuple, logging_dict, weight, label):
		logits = input_tuple[0]
		labels = input_tuple[1]

		probs = self.softmax(logits)
		samples = torch.zeros_like(labels)
		samples[torch.arange(samples.size(0)), probs.max(1)[1]] = 1.0
		test = torch.where(samples == labels, torch.zeros_like(probs), torch.ones_like(probs)).sum(1)
		accuracy = torch.where(test > 0, torch.zeros_like(test), torch.ones_like(test))

		loss = weight * self.loss_function(logits, labels.max(1)[1])

		accuracy_rate = accuracy.mean()

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/' + label] = accuracy.mean().item()	

		return loss

class Multivariate_GaussianNegLogProb_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		labels = input_tuple[1]

		means, precs = params
		
		loss = -1.0 * weight * multiv_gauss_logprob(labels, means, precs).mean()

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['avg_err/' + label] =(means - labels).norm(p=2, dim =1).mean().item()

		return loss

class Multivariate_GaussianKL_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		params_est = input_tuple[0]
		params_tgt = input_tuple[1]

		means_est, precs_est = params_est
		means_tgt, precs_tgt = params_tgt
		
		loss = weight * multiv_gauss_KL(means_tgt, means_est, precs_tgt, precs_est).mean()

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['avg_err/' + label] =(means_est -  means_tgt).norm(p=2, dim =1).mean().item()

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

class GaussianNegLogProb_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		params = input_tuple[0]
		labels = input_tuple[1]

		means, varis = params			

		loss = -1.0 * weight * log_normal(labels, means, varis).sum(1).mean()

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['avg_err/' + label] =(means - labels).pow(2).sum(1).mean(0).item()
		logging_dict['scalar']['avg_var_err/' + label] =((means - labels).pow(2).sum(1) - varis.sum(1)).mean(0).item()

		return loss

class Multivariate_GaussianNegLogProb_multistep_Loss(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label, offset):
		logits_list = input_tuple[0]
		labels_array = input_tuple[1]

		for idx, logits in enumerate(logits_list):
			labels = labels_array[:,idx + offset]
			means, varis = logits			

			if idx == 0:
				loss = -1.0 * weight * multiv_gauss_logprob(labels, means, var).mean()
			else:
				loss += -1.0 * weight * multiv_gauss_logprob(labels, means, var).mean()
		
		logging_dict['scalar']["loss/" + label] = loss.item()

		return loss


class Multinomial_KL_Loss(Proto_Loss):

	def __init__(self, record_function = None):
	    super().__init__(record_function = record_function)
	    self.logsoftmax = nn.LogSoftmax(dim = 1)

	def loss(self, input_tuple, logging_dict, weight, label):
		logits = input_tuple[0]
		target_logits = input_tuple[1]

		l_guess = F.softmax(logits, dim =1).max(1)[1]
		t_guess = F.softmax(target_logits, dim =1).max(1)[1]
		match = torch.where(l_guess == t_guess, torch.ones_like(l_guess), torch.zeros_like(l_guess)).float()

		loss = weight * multinomial_KL(logits, target_logits)

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']["match/" + label] = match.mean().item()
		
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

class BinaryEst_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__(loss_function = nn.BCEWithLogitsLoss())

	def loss(self, input_tuple, logging_dict, weight, label):
		logits = input_tuple[0].squeeze()
		labels = input_tuple[1].squeeze()

		# probs = torch.sigmoid(logits).squeeze()
		# samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
		# accuracy = torch.where(samples == labels, torch.ones_like(probs), torch.zeros_like(probs))

		probs = torch.sigmoid(logits)
		ui_probs = torch.where(probs == torch.sigmoid(torch.tensor(-600).float().to(logits.device)), torch.zeros_like(probs), torch.ones_like(probs))

		# print(logits.size())
		# print(labels.size())
		logits_n = logits[ui_probs.nonzero(as_tuple=True)]
		labels_n = labels[ui_probs.nonzero(as_tuple=True)]

		probs = torch.sigmoid(logits_n).squeeze()

		pos_probs = probs[labels_n.nonzero(as_tuple=True)]

		samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
		accuracy = torch.where(samples == labels_n, torch.ones_like(probs), torch.zeros_like(probs))
		pos_accuracy = torch.where(pos_probs > 0.5, torch.ones_like(pos_probs), torch.zeros_like(pos_probs))


		loss = weight * self.loss_function(logits_n, labels_n)

		# if len(labels_n.nonzero(as_tuple=True)[0]) > 0:
		# 	logits_p = logits_n[labels_n.nonzero(as_tuple=True)]
		# 	labels_p = labels_n[labels_n.nonzero(as_tuple=True)]
		# 	loss += weight * self.loss_function(logits_p, labels_p) 

		logging_dict['scalar']["loss/" + label] = loss.item()
		logging_dict['scalar']['accuracy/' + label] = accuracy.mean().item()

		if len(labels_n.nonzero(as_tuple=True)[0]) > 0:
			logging_dict['scalar']['pos_accuracy/' + label] = pos_accuracy.mean().item()
		else:
			logging_dict['scalar']['pos_accuracy/' + label] = 1	

		return loss

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

class CrossEnt_Ensemble_Loss(Proto_MultiStep_Loss):
	def __init__(self):
	    super().__init__(loss_function = nn.CrossEntropyLoss())
	    self.softmax = nn.Softmax(dim=1)

	def loss(self, input_tuple, logging_dict, weight, label):
		logits_list = input_tuple[0]
		labels = input_tuple[1]

		logits_sum = torch.zeros_like(logits_list[0])

		for idx, logits in enumerate(logits_list):
			logits_sum += logits
			if idx == 0:
				loss = weight * self.loss_function(logits, labels.max(1)[1])
			else:
				loss += weight * self.loss_function(logits, labels.max(1)[1])

		probs = self.softmax(logits_sum)
		samples = torch.zeros_like(labels)
		samples[torch.arange(samples.size(0)), probs.max(1)[1]] = 1.0
		test = torch.where(samples == labels, torch.zeros_like(probs), torch.ones_like(probs)).sum(1)
		accuracy = torch.where(test > 0, torch.zeros_like(test), torch.ones_like(test))

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

class Contrastive_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__()

	def loss(self, input_tuple, logging_dict, weight, label):
		network_outputs = input_tuple[0]
		output_0, output_1 = network_outputs

		loss = -1.0 * weight * self.loss_function(output_0, output_1)
		logging_dict['scalar']["loss/" + label] = loss.item()

		# if self.record_function is not None:
		# 	self.record_function((mu_est, var_est), (mu_tar, var_tar), label, logging_dict)

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

class Ranking_Loss(Proto_Loss):
	def __init__(self):
	    super().__init__(loss_function = None)

	def loss(self, input_tuple, logging_dict, weight, label):
		values = input_tuple[0]
		targets = input_tuple[1]

		# print(targets)

		idxs = torch.argsort(targets)
		v_sorted = values[idxs] - values.min()

		# print(v_sorted)

		v_mat = v_sorted.unsqueeze(0).repeat_interleave(v_sorted.size(0), dim = 0)

		# print(v_mat)
		v_mat_diag = v_mat.diag().unsqueeze(1).repeat_interleave(v_sorted.size(0), dim = 1)

		# print(v_mat_diag)
		v = (v_mat - v_mat_diag)

		# print(v)

		w_bad = torch.where(v <= 0, torch.ones_like(v), torch.zeros_like(v)).triu()
		w_bad[torch.arange(w_bad.size(0)), torch.arange(w_bad.size(1))] *= 0

		w_good = (torch.where(v > 0, torch.ones_like(v), torch.zeros_like(v)) * torch.where(v <= 1, torch.ones_like(v), torch.zeros_like(v))).triu()
		w_good[torch.arange(w_good.size(0)), torch.arange(w_good.size(1))] *= 0

		# print(w_mat)
		# print(w_pos)
		# print(w_neg)
		# print("Weights: ", w)
		# print("Values: ", v_sorted)
		# print((w * torch.abs(w)))
		# print(torch.sign(v_sorted))

		loss = - weight * (w_good * v + w_bad * v).sum()

		# print(loss)

		# a = input("")

		logging_dict['scalar']["loss/" + label] = loss.mean().item()
		logging_dict['scalar']['accuracy/counts' + label] = w_bad.sum().item()	
		# logging_dict['scalar']['accuracy/w_neg' + label] = w_neg.sum().item()	

		return loss