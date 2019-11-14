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

	def loss(self, input_tuple, logging_dict, weight):
		net_est = input_tuple[0]
		target = input_tuple[1]

		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + self.loss_name] = loss.item()

		# self.loss_metrics(input_tuple, logging_dict)

		return loss

	def loss_metrics(self, input_tuple, logging_dict):
		pass

class Image_Reconstruction(Proto_Loss):
	def loss(self, input_tuple, logging_dict, weight):
		net_est = input_tuple[0]
		target = input_tuple[1]


		loss = weight * self.loss_function(net_est, target)

		logging_dict['scalar']["loss/" + self.loss_name] = loss.item()
		logging_dict['image'] = []		
		logging_dict['image'].append(net_est[0])
		logging_dict['image'].append(target[0])

		# self.loss_metrics(input_tuple, logging_dict)

		return loss

class Gaussian_KL(Proto_Loss):

	def __init__(self, loss_name):
	    super().__init__(None, loss_name)

	def loss(self, input_tuple, logging_dict, weight):
		params = input_tuple[0]
		mu_est, var_est, mu_tar, var_tar = params

		element_wise = 0.5 * (torch.log(var_tar) - torch.log(var_est) + var_est / var_tar + (mu_est - mu_tar).pow(2) / var_tar - 1)

		loss = weight * element_wise.sum(1).sum(0)

		logging_dict['scalar']["loss/" + self.loss_name] = loss.item()

		# self.loss_metrics(input_tuple, scalar_dict)

		return loss
		
# class Learning_Metrics(object):

# 	def __init__(self, constants = []):
# 		self.constants = constants

# 	def metrics(self, inputs, scalar_dict):
# 		pass

