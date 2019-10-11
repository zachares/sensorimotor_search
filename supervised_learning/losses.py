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

	def loss(self, inputs, scalar_dict):
		est, target = inputs

		self.loss_metrics(inputs, scalar_dict)

		return self.loss_function(est, target)

	def loss_metrics(self, inputs, scalar_dict):
		pass


class Learning_Metrics(object):

	def __init__(self, constants = []):
		self.constants = constants

	def metrics(self, inputs, scalar_dict):
		pass

