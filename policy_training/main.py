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
from trainer import Trainer
from logger import Logger
from dataloader import *
import yaml

from models import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile

######## git hash to know the state of the code when experiments are run

if __name__ == '__main__':

    ##################################################################################
    ##### Loading required config files
    ##################################################################################
    with open("../sim_env/game_params.yml", 'r') as ymlfile:
        cfg_0 = yaml.safe_load(ymlfile)

    cols = cfg_0['game_params']['cols']
    rows = cfg_0['game_params']['rows']
    pose_dim = cfg_0['game_params']['pose_dim']

    with open("../data_collection/datacollection_params.yml", 'r') as ymlfile:
        cfg_1 = yaml.safe_load(ymlfile)

    dataset_path = cfg_1['datacollection_params']['logging_folder']

    with open("goal_params.yml", 'r') as ymlfile:
        cfg_2 = yaml.safe_load(ymlfile)

    debugging_val = cfg_2['debugging_params']['debugging_val']
    save_model_val = cfg_2['debugging_params']['save_model_val']

    z_dim = cfg_2["model_params"]["model_1"]["z_dim"]
    use_cuda = cfg_2['training_params']['use_GPU'] and torch.cuda.is_available()

    seed = cfg_2['training_params']['seed']
    regularization_weight = cfg_2['training_params']['regularization_weight']
    learning_rate = cfg_2['training_params']['lrn_rate']
    beta_1 = cfg_2['training_params']['beta1']
    beta_2 = cfg_2['training_params']['beta2']
    max_epoch = cfg_2['training_params']['max_training_epochs']
    val_ratio = cfg_2['training_params']['val_ratio']
    logging_folder = cfg_2['logging_params']['logging_folder']
    run_description = cfg_2['logging_params']['run_description'] 

    num_workers = cfg_2['dataloading_params']['num_workers']

    if run_description == "testing":
        max_epoch = 1
        val_ratio = 0
        test_run = True
    else:
        test_run = False

    ##################################################################################
    ### Setting Debugging Flag and Save Model Flag
    ##################################################################################
    if debugging_val == 1.0:
        debugging_flag = True
        var = input("Debugging flag activated. No Results will be saved. Continue with debugging [y,n]: ")
        if var != "y":
            debugging_flag = False
    else:
        debugging_flag = False

    if debugging_flag:
        print("Currently Debugging")
    else:
        print("Training with debugged code")

    if save_model_val == 1.0:
        save_model_flag = True
    else:
        var = input("Save Model flag deactivated. No models will be saved. Are you sure[y,n]: ")
        if var == "y":
            save_model_flag = False
        else:
            save_model_flag = True

    ##################################################################################
    # hardware and low level training details
    ##################################################################################
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if use_cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    if use_cuda:
      print("Let's use", torch.cuda.device_count(), "GPUs!")

    ##################################################################################
    #### Training tool to train and evaluate neural networks
    ##################################################################################
    trainer = Trainer(cfg_0, cfg_1, cfg_2, device)

    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
    logger = Logger(cfg_0, cfg_1, cfg_2, trainer.model_dict, debugging_flag, save_model_flag, device)

    ##################################################################################
    #### Dataset creation function
    ##################################################################################
    data_loader = init_dataloader(cfg_0, cfg_1, cfg_2, debugging_flag, logger.runs_folder, device)

    np.random.seed(seed)
    ##################################################################################
    ####### Training ########
    ##################################################################################
    global_cnt = 0
    val_global_cnt = 0

    for i_epoch in range(max_epoch):
        print('Training epoch #{}...'.format(i_epoch))
        
        for i_iter, sample_batched in enumerate(data_loader):

            if global_cnt % 500 == 0:
                print(global_cnt, " Updates to the model have been performed ")

            scalar_dict = trainer.train(sample_batched)

            logger.save_scalars(scalar_dict, global_cnt, 'train/')

            global_cnt += 1

        logger.save_models(trainer.model_dict, i_epoch) #  trainer.dyn_mod, trainer.goal_prov,

        print("Epoch Params: ", trainer.current_params)
        print("Epoch bounds: ", trainer.bounds)
        # print("Last loss: ", trainer.loss)
