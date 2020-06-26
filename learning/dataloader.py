import h5py
import torch
import numpy as np
import copy
import time
import os
import yaml

from torch.utils.data import Dataset
from utils import read_h5
from sklearn.linear_model import LinearRegression

## ADD data normalization
class Custom_DataLoader(Dataset):
    def __init__(self, cfg, idx_dict = None, device = None, transform=None):
        dataset_path = cfg['dataset_path']
        dataset_keys = cfg['dataset_keys']

        val_ratio = cfg['training_params']['val_ratio']

        dev_num = cfg['training_params']['dev_num']

        ###############################################
        ######### Dataset Loading ###########
        ###############################################
        print("Dataset path: ", dataset_path)

        ###############################################
        #### Create Filename list of Datasets #####
        ###############################################
        # filename_list = []
        # for file in os.listdir(dataset_path):
        #     if file.endswith(".h5"): # and len(filename_list) < 20:
        #         filename_list.append(dataset_path + file)

        self.device = device
        self.transform = transform
        self.dataset_path = dataset_path     
        self.dataset_keys = dataset_keys

        self.val_bool = False
        self.dev_bool = False
        self.val_ratio = val_ratio
        self.dev_ratio = 0

        self.train_length = 0
        self.val_length = 0
        self.dev_length = 0

        if idx_dict == None:
            self.idx_dict = {}
            self.idx_dict['val'] = {}
            self.idx_dict['train'] = {}
            self.idx_dict['dev'] = {} 
            print("Starting Train Val Split")

            ###########################################################
            ##### Project Specific Code Here ##########################
            ###########################################################            
            self.max_length = 0

            with open(self.dataset_path + "datacollection_params.yml", 'r') as ymlfile:
                cfg1 = yaml.safe_load(ymlfile)

            tool_types = cfg1['peg_names']
            hole_types = cfg1['hole_names']
            option_types = cfg1['fit_names']

            num_trajectories = cfg['custom_params']['num_trajectories']
            min_length = cfg['custom_params']['min_length']

            self.num_trajectories = num_trajectories
            self.min_length = min_length
            self.option_types = option_types
            self.tool_types = tool_types
            self.hole_types = hole_types

            self.dev_ratio = dev_num / (9 * num_trajectories)

            self.idx_dict["min_length"] = self.min_length
            self.idx_dict["option_types"] = self.option_types
            self.idx_dict["tool_types"] = self.tool_types
            self.idx_dict["hole_types"] = self.hole_types

            for idx in range(num_trajectories):
                for peg in tool_types:
                    
                    train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 1) ### 1 is train, 0 is val
                    dev_bool = np.random.binomial(1, 1 - self.dev_ratio, 1) ### 1 is train, 0 is val

                    for hole in hole_types:

                        filename = self.dataset_path + peg + "_" + hole + "_" + str(idx + 1).zfill(4) + ".h5" 

                        dataset = read_h5(filename)

                        if "proprio" not in dataset.keys():
                            continue

                        proprios = np.array(dataset['proprio'])
                        length = proprios.shape[0]
                        dataset.close()

                        if length < self.min_length:
                            continue

                        if self.max_length < length:
                            self.max_length = int(length)

                        if train_val_bool == 1:
                            self.idx_dict['train'][self.train_length] = filename
                            self.train_length += 1
                        else:
                            self.idx_dict['val'][self.val_length] = filename
                            self.val_length += 1

                        if dev_bool == 0:
                            self.idx_dict['dev'][self.dev_length] = filename
                            self.dev_length += 1

            self.idx_dict["max_length"] = self.max_length
            ##########################################################    
            ##### End of Project Specific Code #######################
            ##########################################################
        else:

            self.idx_dict = idx_dict
            
            self.train_length = len(list(self.idx_dict['train'].keys()))
            self.val_length = len(list(self.idx_dict['val'].keys()))
            self.dev_length = len(list(self.idx_dict['dev'].keys()))

        print("Total data points: ", self.train_length + self.val_length)
        print("Total training points: ", self.train_length)
        print("Total validation points: ", self.val_length)
        print("Total development points: ", self.dev_length)

    def __len__(self):
        if self.dev_bool:
            return self.dev_length     
        elif self.val_bool:
            return self.val_length
        else:
            return self.train_length

    def __getitem__(self, idx):
        if self.dev_bool:
            key_set = 'dev'
        elif self.val_bool:
            key_set = 'val'
        else:
            key_set = 'train'

        ###########################################################
        ##### Project Specific Code Here ##########################
        ###########################################################
        dataset = read_h5(self.idx_dict[key_set][idx])
        max_traj_length = np.array(dataset['proprio']).shape[0]

        max_length = self.idx_dict["max_length"]
        min_length = self.idx_dict["min_length"]

        if max_traj_length <= min_length:
            idx0 = 0
            idx1 = max_traj_length
        else:
            idx0 = np.random.choice(max_traj_length - min_length) # beginning idx
            idx1 = np.random.choice(range(idx0 + min_length, max_traj_length)) # end idx

        padded = max_length - idx1 + idx0 + 1
        unpadded = idx1 - idx0 - 1
        sample = {}

        for key in self.dataset_keys:
            if key == 'action':
                sample[key] = np.array(dataset[key][idx0:(idx1 - 1)]) # each action corresponds to the action causing the difference recorded
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)             
            elif key == 'force_hi_freq':
                sample[key] = np.array(dataset[key][(idx0 + 1):idx1])
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1], sample[key].shape[2]))], axis = 0)
            elif key == 'proprio':   
                sample[key] = np.array(dataset[key][idx0:idx1])
                init_proprio = sample[key][0,:6]
                sample[key + "_diff"] = sample[key][1:] - sample[key][:-1]
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)
                sample[key + "_diff"] = np.concatenate([sample[key + "_diff"], np.zeros((padded, sample[key + "_diff"].shape[1]))], axis = 0)
            elif key == 'contact':
                sample[key] = np.array(dataset[key][idx0:idx1])
                sample[key + "_diff"] = sample[key][1:].astype(np.int16) - sample[key][:-1].astype(np.int16) 
                sample[key] = np.concatenate([sample[key], np.zeros((padded))], axis = 0)
                sample[key + "_diff"] = np.concatenate([sample[key + "_diff"], np.zeros((padded))], axis = 0)  
            elif key == 'peg_type' or key == 'hole_type' or key == 'macro_action' or key == 'fit_type':
                sample[key] = np.array(dataset[key])

                if key == "fit_type":
                    sample["option_type"] = sample[key]
                    sample["fit_label"] = sample[key].argmax(0)
                elif key == "peg_type":
                    sample["tool_type"] = sample[key]
                elif key == "hole_type":
                    sample["state_type"] = sample[key]
                   

        dataset.close()

        sample["padding_mask"] = np.concatenate([np.zeros(unpadded), np.ones(padded)])
        sample["macro_action"] = np.concatenate([init_proprio[:2], sample['macro_action'][6:]]) #, np.array([unpadded])])

        ##########################################################    
        ##### End of Project Specific Code #######################
        ##########################################################

        return self.transform(sample)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device = None):
        
        self.device = device

    def __call__(self, sample):

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():

        ###########################################################
        ##### Project Specific Code Here ##########################
        ###########################################################

            if k == "padding_mask":
                new_dict[k] = torch.from_numpy(v).bool()
            elif k[-5:] == "label":
                new_dict[k] = v
            else:
                new_dict[k] = torch.from_numpy(v).float()
            '''
            Default code is
            new_dict[k] = torch.from_numpy(v).float()
            '''
        ##########################################################    
        ##### End of Project Specific Code #######################
        ##########################################################

        return new_dict
