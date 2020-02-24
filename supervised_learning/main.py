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

sys.path.insert(0, "../models/") 
sys.path.insert(0, "../datalogging/") 

from trainer import Trainer
from logger import Logger
from models import *
from utils import *

import matplotlib.pyplot as plt 
from shutil import copyfile

######## git hash to know the state of the code when experiments are run

if __name__ == '__main__':

    ##################################################################################
    ##### Loading required config files
    ##################################################################################
    with open("learning_params.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    debugging_val = cfg['debugging_params']['debugging_val']
    save_model_val = cfg['debugging_params']['save_model_val']

    z_dim = cfg["model_params"]["z_dim"]
    use_cuda = cfg['training_params']['use_GPU'] and torch.cuda.is_available()

    seed = cfg['training_params']['seed']
    regularization_weight = cfg['training_params']['regularization_weight']
    learning_rate = cfg['training_params']['lrn_rate']
    beta_1 = cfg['training_params']['beta1']
    beta_2 = cfg['training_params']['beta2']
    max_epoch = cfg['training_params']['max_training_epochs']
    val_ratio = cfg['training_params']['val_ratio']
    logging_folder = cfg['logging_params']['logging_folder']
    run_description = cfg['logging_params']['run_description'] 

    num_workers = cfg['dataloading_params']['num_workers']
    dataset_path = cfg['dataloading_params']['dataset_path']
    idx_dict_path = cfg['dataloading_params']['idx_dict_path']

    if idx_dict_path == "":
        idx_dict_path = None

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
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
    logger = Logger(cfg, debugging_flag, save_model_flag)

    ##################################################################################
    #### Training tool to train and evaluate neural networks
    ##################################################################################
    trainer = Trainer(cfg, logger.models_folder, save_model_flag, device)

    ##################################################################################
    #### Dataset creation function
    ##################################################################################
    data_loader, val_data_loader, idx_dict = init_dataloader(cfg, device, idx_dict_path)

    if save_model_flag:
        logger.save_dict("val_train_split", idx_dict, False)
        trainer.save(0)
    ##################################################################################
    ####### Training ########
    ##################################################################################
    global_cnt = 0
    i_epoch = 0
    val_global_cnt = 0
    prev_time = time.time()

    if not test_run:
        for i_epoch in range(max_epoch):
            current_time = time.time()

            if i_epoch != 0:
                print("Epoch took ", current_time - prev_time, " seconds")
                prev_time = time.time()

            print('Training epoch #{}...'.format(i_epoch))
            
            for i_iter, sample_batched in enumerate(data_loader):
                sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

                logging_dict = trainer.train(sample_batched)
                global_cnt += 1

                if global_cnt % 50 == 0 or global_cnt == 1:
                    print(global_cnt, " Updates to the model have been performed ")
                    logger.save_images2D(logging_dict, global_cnt, 'train/')

                logger.save_scalars(logging_dict, global_cnt, 'train/')
            ##################################################################################
            ##### Validation #########
            ##################################################################################
            # performed at the end of each epoch
            if val_ratio is not 0 and (i_epoch % 25) == 0:
                
                print("Calculating validation results after #{} epochs".format(i_epoch))

                for i_iter, sample_batched in enumerate(val_data_loader):
                    sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                    sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

                    logging_dict= trainer.eval(sample_batched)

                    logger.save_scalars(logging_dict, val_global_cnt, 'val/')

                    val_global_cnt += 1

                logger.save_images2D(logging_dict, val_global_cnt, 'val/')

            ###############################################
            ##### Saving models every epoch ################
            ##############################################
            if save_model_flag and i_epoch == 0:
                if os.path.isdir(logger.models_folder) == False:
                    os.mkdir(logger.models_folder)
                trainer.save(i_epoch)

            elif save_model_flag and (i_epoch % 25) == 0:
                trainer.save(i_epoch)

    else:
        print("Starting Testing")

        for i_iter, sample_batched in enumerate(data_loader):
            sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
            sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

            logging_dict= trainer.eval(sample_batched)

            logger.save_scalars(logging_dict, global_cnt, 'test/')

            if (global_cnt + 1) % 50 == 0 or (global_cnt + 1) == 1:
                print(global_cnt + 1, " samples tested")

            global_cnt += 1

        logger.save_images2D(logging_dict, global_cnt, 'test/')

        print("Finished Testing")