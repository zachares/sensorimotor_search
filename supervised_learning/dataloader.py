import h5py
import torch
import numpy as np
import time
import datetime
import os
import sys
import yaml
import copy
import random
import pickle


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

## ADD support for single file, multiset dataset and multifile multiset dataset
## ADD support for other file types besides H5

### dataloader only supports loading of H5 files
### dataloader only supports single file datasets or multifile datasets where each file 
### only contains a single subset of the full dataset

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(angle)
    zeros = np.zeros_like(angle)

    case1 = np.where(angle < -TWO_PI, angle + TWO_PI * np.ceil(abs(angle) / TWO_PI), zeros)
    case2 = np.where(angle > TWO_PI, angle - TWO_PI * np.floor(angle / TWO_PI), zeros)
    case3 = np.where(angle > -TWO_PI, ones, zeros) * np.where(angle < 0, TWO_PI + angle, zeros)
    case4 = np.where(angle < TWO_PI, ones, zeros) * np.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4

def calc_angerr(target, current):
    TWO_PI = 2 * np.pi
    ones = np.ones_like(target)
    zeros = np.zeros_like(target)

    targ = np.where(target < 0, TWO_PI + target, target)
    curr = np.where(current < 0, TWO_PI + current, current)

    curr0 = np.where(abs(targ - (curr + TWO_PI)) < abs(targ - curr), ones , zeros) # curr + TWO_PI
    curr1 = np.where(abs(targ - (curr - TWO_PI)) < abs(targ - curr), ones, zeros) # curr - TWO_PI
    curr2 = ones - curr0 - curr1

    curr_fin = curr0 * (curr + TWO_PI) + curr1 * (curr - TWO_PI) + curr2 * curr

    error = targ - curr_fin

    error0 = np.where(abs(error + TWO_PI) < abs(error), ones, zeros)
    error1 = np.where(abs(error - TWO_PI) < abs(error), ones, zeros)
    error2 = ones - error0 - error1

    return error * error2 + (error + TWO_PI) * error0 + (error - TWO_PI) * error1

def selfsupervised_filenum(filenum):
    offset = 633
    idx_list = [634, 1267, 1900, 2533, 3166, 3799, 4432, 5065,  5698]
    choice = np.random.choice(2) + 1

    if filenum < 1900:
        if filenum < 634:
            return filenum + offset * choice

        elif filenum >= 634 and filenum < 1267:
            if choice == 1:
                return filenum - offset
            else:
                return filenum + offset
        else:
            return filenum - offset * choice

    elif filenum >= 1900 and filenum < 3799:
        if filenum < 2533:
            return filenum + offset * choice

        elif filenum >= 2533 and filenum < 3166:
            if choice == 1:
                return filenum - offset
            else:
                return filenum + offset
        else:
            return filenum - offset * choice
    else:
        if filenum < 4432:
            return filenum + offset * choice

        elif filenum >= 4432 and filenum < 5065:
            if choice == 1:
                return filenum - offset
            else:
                return filenum + offset
        else:
            return filenum - offset * choice

class H5_DataLoader(Dataset):
    ### h5py file type dataloader dataset ###
    def __init__(self, filename_list, loading_dict, val_ratio, num_steps = 1, idx_dict = None, device = None, transform=None):
        self.device = device
        self.transform = transform
        self.loading_dict = loading_dict
        self.sample = {}

        self.filename_list = filename_list     

        self.num_steps = num_steps

        self.val_bool = False
        self.val_ratio = val_ratio

        self.train_length = 0
        self.val_length = 0

        self.up_thresh = 0.0003

        if idx_dict == None:
            self.idx_dict = {}
            self.idx_dict['val'] = {}
            self.idx_dict['train'] = {}
            val_eepos_list = []
            train_eepos_list = []

            #### assumes one dataset per file

            print("Starting Train Val Split")
            for idx, filename in enumerate(self.filename_list):
                if ((idx + 1) % 500) == 0:
                    print("Processed ", idx + 1, "files out of", len(self.filename_list))
                dataset = self.load_file(filename)
                proprios = np.array(dataset['proprio'])
                dataset_length = proprios.shape[0]

                filenum = int(filename_list[-7:-3])
                ss_filenum = selfsupervised_filenum(filenum)

                ss_filename = filename[:-7] + str(ss_filenum) + ".h5"
                dataset_ss = 

                for dataset_idx in range(dataset_length):
                    min_idx = dataset_idx
                    max_idx = dataset_idx + self.num_steps + 1

                    if max_idx >= dataset_length:
                        continue

                    train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 1) ### 1 is train, 0 is val

                    if train_val_bool == 1:
                        self.idx_dict['train'][self.train_length] = (filename, (min_idx, max_idx))
                        self.train_length += 1  

                    else:
                        self.idx_dict['val'][self.val_length] = (filename, (min_idx, max_idx)) 
                        self.val_length += 1                         

                dataset.close()                  

        else:
            self.idx_dict = idx_dict
            self.train_length = len(list(self.idx_dict['train'].keys()))
            self.val_length = len(list(self.idx_dict['val'].keys()))

        self.prev_time = time.time()

        print("Total data points: ", self.train_length + self.val_length)
        print("Total training points: ", self.train_length)
        print("Total validation points: ", self.val_length)

    def return_idxs(self, curr_idx):
        file_idx = int(np.floor(curr_idx / self.dataset_length))
        return (file_idx, int(curr_idx - file_idx * self.dataset_length))

    def __len__(self):      
        if self.val_bool:
            return self.val_length
        else:
            return self.train_length

    def __getitem__(self, idx):
        if self.val_bool:
            key_set = 'val'
        else:
            key_set = 'train'

        dataset = self.load_file(self.idx_dict[key_set][idx][0])
        idxs_p = self.idx_dict[key_set][idx][1]
        # dataset_up = self.load_file(self.idx_dict[key_set][idx][2])
        # idxs_up = self.idx_dict[key_set][idx][3]

        sample = {}

        for key in self.loading_dict.keys():
            # print( key , " took ", time.time() - self.prev_time, " seconds")
            # self.prev_time = time.time()
            if key == 'image':
                sample[key] = dataset['image_s'][idxs_p[0]:idxs_p[1]]  
            elif key == 'depth':
                sample[key] = dataset['depth_s'][idxs_p[0]:idxs_p[1]]  
            elif key == 'action':
                sample[key] = np.array(dataset[key][idxs_p[0]:(idxs_p[1]-1)])
                # sample[key + '_m'] = np.linalg.norm(sample[key], axis = 1)
                # sample[key + '_d'] = sample[key] / np.repeat(np.expand_dims(sample[key + '_m'], axis = 1), sample[key].shape[1], axis = 1)
            # elif key == 'force':   
            #     sample[key] = np.array(dataset[key][idxs_p[0]:idxs_p[1]])  
            elif key == 'proprio':   
                sample[key] = np.array(dataset[key])[idxs_p[0]:idxs_p[1]]
                sample["pos"] = np.array(dataset[key])[idxs_p[0]:idxs_p[1], :3] 
                # sample["pos_m"] = np.linalg.norm(sample["pos"], axis = 1)
                # sample["pos_d"] = sample["pos"] / np.repeat(np.expand_dims(sample["pos_m"], axis = 1), sample["pos"].shape[1], axis = 1)

                sample["pos_diff"] = sample["pos"][1:] - sample["pos"][:-1]
                sample["pos_diff_m"] = np.linalg.norm(sample["pos_diff"], axis = 1)
                sample["pos_diff_d"] = sample["pos_diff"] / np.repeat(np.expand_dims(sample["pos_diff_m"], axis = 1), sample["pos_diff"].shape[1], axis = 1)

                sample["ang"] = T_angle(np.array(dataset[key])[idxs_p[0]:idxs_p[1], 3:6])
                # sample["ang_m"] = np.linalg.norm(sample["ang"], axis = 1)
                # sample["ang_d"] = sample["ang"] / np.repeat(np.expand_dims(sample["ang_m"], axis = 1), sample["ang"].shape[1], axis = 1)

                # sample["ang_diff"] = calc_angerr(sample["ang"][:-1], sample["ang"][1:])
                # sample["ang_diff_m"] = np.linalg.norm(sample["ang_diff"], axis = 1)
                # sample["ang_diff_d"] = sample["ang_diff"] / np.repeat(np.expand_dims(sample["ang_diff_m"], axis = 1), sample["ang_diff"].shape[1], axis = 1)

                sample["vel"] = np.array(dataset[key])[idxs_p[0]:idxs_p[1], 6:9] 
                # sample["vel_m"] = np.linalg.norm(sample["vel"], axis = 1)
                # sample["vel_d"] = sample["vel"] / np.repeat(np.expand_dims(sample["vel_m"], axis = 1), sample["vel"].shape[1], axis = 1)

                # sample["vel_diff"] = sample["vel"][1:] - sample["vel"][:-1]
                # sample["vel_diff_m"] = np.linalg.norm(sample["vel_diff"], axis = 1)
                # sample["vel_diff_d"] = sample["vel_diff"] / np.repeat(np.expand_dims(sample["vel_diff_m"], axis = 1), sample["vel_diff"].shape[1], axis = 1)

                sample["ang_vel"] = np.array(dataset[key])[idxs_p[0]:idxs_p[1], 9:12] 
                # sample["ang_vel_m"] = np.linalg.norm(sample["ang_vel"], axis = 1)
                # sample["ang_vel_d"] = sample["ang_vel"] / np.repeat(np.expand_dims(sample["ang_vel_m"], axis = 1), sample["ang_vel"].shape[1], axis = 1)

                # sample["ang_vel_diff"] = sample["ang_vel"][1:] - sample["ang_vel"][:-1]
                # sample["ang_vel_diff_m"] = np.linalg.norm(sample["ang_vel_diff"], axis = 1)
                # sample["ang_vel_diff_d"] = sample["ang_vel_diff"] / np.repeat(np.expand_dims(sample["ang_vel_diff_m"], axis = 1), sample["ang_vel_diff"].shape[1], axis = 1)

            elif key == 'joint_pos':
                sample[key] = T_angle(np.array(dataset[key][idxs_p[0]:idxs_p[1]]))
                # sample[key + '_m'] = np.linalg.norm(sample[key], axis = 1)
                # sample[key + '_d'] = sample[key] / np.repeat(np.expand_dims(sample[key + '_m'], axis = 1), sample[key].shape[1], axis = 1)

                # sample[key + "_diff"] = calc_angerr(sample[key][:-1], sample[key][1:])
                # sample[key + "_diff_m"] = np.linalg.norm(sample[key + "_diff"], axis = 1)
                # sample[key + "_diff_d"] = sample[key + "_diff"] / np.repeat(np.expand_dims(sample[key + "_diff_m"], axis = 1), sample[key + "_diff"].shape[1], axis = 1)

            # elif key == 'joint_vel':
            #     sample[key] = dataset[key][idxs_p[0]:idxs_p[1]]
            #     # sample[key + '_m'] = np.linalg.norm(sample[key], axis = 1)
            #     # sample[key + '_d'] = sample[key] / np.repeat(np.expand_dims(sample[key + '_m'], axis = 1), sample[key].shape[1], axis = 1)

            #     # sample[key + "_diff"] = sample[key][1:] - sample[key][:-1]
            #     # sample[key + "_diff_m"] = np.linalg.norm(sample[key + "_diff"], axis = 1)
            #     # sample[key + "_diff_d"] = sample[key + "_diff"] / np.repeat(np.expand_dims(sample[key + "_diff_m"], axis = 1), sample[key + "_diff"].shape[1], axis = 1)

            elif key == 'peg_type' or key == 'hole_type' or key == 'fit':
                sample[key] = np.array(dataset[key])
            else:
                sample[key] = dataset[key][idxs_p[0]:idxs_p[1]]  

        dataset.close()
        # dataset_up.close()

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

def init_dataloader(cfg, device, idx_dict_path = None):
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
        if file.endswith(".h5"): # and len(filename_list) < 20:
            filename_list.append(dataset_path + file)


    #### loading previous val_train split to continue training a model
    if idx_dict_path is not None:
        with open(idx_dict_path, 'rb') as f:
            idx_dict = pickle.load(f)

        print("Loaded Train Val split dictionary from path")
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

