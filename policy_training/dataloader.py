import h5py
import torch
import numpy as np
import time
import datetime
import os
import sys
import yaml
import copy
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

sys.path.append('/scr-ssd/gg_rl/data_analysis/')

from data_analysis import *

############## make sure points are ordered ##############

class H5_DataLoader(Dataset):
    """h5py file type dataloader with block stacking dataset."""

    def __init__(self, filename, device = None, transform=None):

        self.device = device
        self.transform = transform

        dataset = h5py.File(filename, 'r', swmr=True, libver = 'latest')

        self.points = np.array(dataset['points'])
        self.obs = np.array(dataset['obs'])
        self.trans = np.array(dataset['trans'])
        self.corr_points = np.array(dataset['corr_points'])
        self.known_length = self.points.shape[0]

        dataset.close()

        # self.multinomial_dict = compute_multinomial_dis(self.filename_list)

    def __len__(self):      

        return self.known_length

    def __getitem__(self, idx):


        params = np.array([0.1, 0.4, 0.5, 0.85, 0.95, 1.5, 2.5, 3, 4])
        point = np.random.uniform(0.0, 1.0, 1)
        sample = point * (0.1 + 0.1 + 0.1 + 1 + 1) 

        if sample <= 0.1:

            sample = sample

        elif sample > 0.1 and sample <= 0.2:

            sample = sample + 0.3

        elif sample > 0.2 and sample <= 0.3:

            sample = sample + 0.65

        elif sample > 0.3 and sample <= 1.3:

            sample = sample + 1.2

        else:
            sample = sample + 1.7


        sample = {
                'points': point,
                'params' :  params,
                'samples': sample,
                }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device = None):
        
        self.device = device

    def __call__(self, sample):

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if self.device == None:
                new_dict[k] = torch.FloatTensor(v)#torch.tensor(v, device = self.device, dtype = torch.float32)
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict

def init_dataloader(cfg_0, cfg_1, cfg_2, debugging_flag, runs_folder, device):


    ###############################################
    ########## Loading dataloader parameters #######
    ###############################################

    dataset_path = cfg_2['dataloading_params']['dataset_path']
    num_workers = cfg_2['dataloading_params']['num_workers']
    batch_size = cfg_2['dataloading_params']['batch_size']

    run_description = cfg_2['logging_params']['run_description'] 

    if run_description == "testing":
        max_epoch = 1
        val_ratio = 0
        test_run = True
    else:
        test_run = False
    ###############################################
    ######### Dataset Loading ###########
    ###############################################
    print("Dataset path: ", dataset_path)

    ###############################################
    #### Create Filename list of Datasets #####
    ###############################################

    dataset = H5_DataLoader(dataset_path, device = device, transform=transforms.Compose([ToTensor(device = device)]))

    train_sampler = SubsetRandomSampler(range(dataset.__len__()))    
    data_loader = DataLoader(dataset, batch_size= batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory = True) 

    return data_loader
