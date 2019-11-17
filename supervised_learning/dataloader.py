import h5py
import torch
import numpy as np
import time
import datetime
import os
import sys
import yaml
import copy

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

## ADD support for single file, multiset dataset and multifile multiset dataset
## ADD support for other file types besides H5

### dataloader only supports loading of H5 files
### dataloader only supports single file datasets or multifile datasets where each file 
### only contains a single subset of the full dataset

class H5_DataLoader(Dataset):
    ### h5py file type dataloader dataset ###
    def __init__(self, filename_list, loading_dict, val_ratio, num_steps = 1, idx_dict = None, device = None, transform=None):
        self.device = device
        self.transform = transform
        self.loading_dict = loading_dict
        self.filename_list = filename_list
        self.num_steps = num_steps

        self.val_bool = False
        self.val_ratio = val_ratio

        self.train_length = 0
        self.val_length = 0

        if idx_dict == None:
            self.idx_dict = {}
            self.idx_dict['val'] = {}
            self.idx_dict['train'] = {}

            #### assumes one dataset per file
            for idx, filename in enumerate(filename_list):
                dataset = self.load_file(filename)
                traj_nums = np.array(dataset['traj_num'])
                dataset_length = traj_nums.shape[0]

                for dataset_idx in range(dataset_length):
                    min_idx = dataset_idx
                    max_idx = dataset_idx + self.num_steps + 1

                    if max_idx >= dataset_length or (traj_nums[min_idx] != traj_nums[max_idx]):
                        continue

                    train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 1) ### 1 is train, 0 is val

                    if train_val_bool == 1:
                        self.idx_dict['train'][self.train_length] = (filename, (min_idx, max_idx)) 
                        self.train_length += 1  

                    else:
                        self.idx_dict['val'][self.val_length] = (filename, (min_idx, max_idx)) 
                        self.val_length += 1                         

                dataset.close()

            print("Total data points: ", self.train_length + self.val_length)
            print("Total training points: ", self.train_length)
            print("Total validation points: ", self.val_length)

        else:
            self.idx_dict = idx_dict

            self.train_length = len(self.idx_dict['train'].keys())
            self.val_length = len(self.idx_dict['val'].keys())

    def __len__(self):      
        if self.val_bool:
            return self.val_length
        else:
            return self.train_length

    def __getitem__(self, idx):
        if self.val_bool:
            dataset = self.load_file(self.idx_dict['val'][idx][0])
            idx_bounds = self.idx_dict['val'][idx][1] 
        else:
            dataset = self.load_file(self.idx_dict['train'][idx][0])
            idx_bounds = self.idx_dict['train'][idx][1] 
        
        sample = {}

        for key in self.loading_dict.keys():
            if key == "action":
                sample[key] = np.array(dataset[key])[idx_bounds[0]:idx_bounds[1] - 1]                
            else:
                sample[key] = np.array(dataset[key])[idx_bounds[0]:idx_bounds[1]]

        dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_file(self, path):
        return h5py.File(path, 'r', swmr=True, libver = 'latest')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device = None):
        
        self.device = device

    def __call__(self, sample):

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            new_dict[k] = torch.from_numpy(v).float()
        return new_dict

def init_dataloader(cfg, device):
    ###############################################
    ########## Loading dataloader parameters ######
    ###############################################
    dataset_path = cfg['dataloading_params']['dataset_path']
    num_workers = cfg['dataloading_params']['num_workers']
    batch_size = cfg['dataloading_params']['batch_size']
    num_steps = cfg['dataloading_params']['num_steps']

    val_ratio = cfg['training_params']['val_ratio']

    run_description = cfg['logging_params']['run_description']

    sample_keys = cfg['info_flow']['dataset']['outputs']

    saved_dataset_split = cfg['dataloading_params']['saved_dataset_split']

    if run_description == "testing":
        val_ratio = 0

    ###############################################
    ######### Dataset Loading ###########
    ###############################################
    print("Dataset path: ", dataset_path)

    ###############################################
    #### Create Filename list of Datasets #####
    ###############################################
    filename_list = []
    for file in os.listdir(dataset_path):
        if file.endswith(".h5"):
            filename_list.append(dataset_path + file)

    #### loading previous val_train split to continue training a model
    if saved_dataset_split != "":
        with open(saved_dataset_split, 'r') as ymlfile_dl:
            cfg_dataloader = yaml.safe_load(ymlfile_dl)

        idx_dict = cfg_dataloader['idx_dict']

    else:
        idx_dict = None

    dataset = H5_DataLoader(filename_list, sample_keys, val_ratio, num_steps = num_steps, idx_dict = idx_dict, device = device, transform=transforms.Compose([ToTensor(device = device)]))

    if val_ratio == 0:
        print("No validation set")

    else:
        dataset.val_bool = False

        val_dataset = copy.copy(dataset)

        val_dataset.val_bool = True

    train_sampler = SubsetRandomSampler(range(dataset.__len__()))    
    data_loader = DataLoader(dataset, batch_size= batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory = True) 

    val_data_loader = None  

    if val_ratio != 0:
        val_sampler = SubsetRandomSampler(range(val_dataset.__len__()))
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, sampler= val_sampler, pin_memory = True)

    return data_loader, val_data_loader, dataset.idx_dict

