import h5py
import torch
import numpy as np
import copy
import time
import os
import yaml
from torch.utils.data import Dataset
import random

## ADD data normalization
def read_h5(path):
    return h5py.File(path, 'r', swmr=True, libver = 'latest')

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

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
        filename_list = []
        if type(dataset_path) is list:
            print("Multi directory dataset")
            for path in dataset_path:
                for file in os.listdir(path):
                    if file.endswith(".h5"): # and len(filename_list) < 20:
                        filename_list.append(path + file)
        else:
            for file in os.listdir(dataset_path):
                if file.endswith(".h5"): # and len(filename_list) < 20:
                    filename_list.append(dataset_path + file)

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
            min_length = 20
            max_length = 100

            self.min_length = min_length
            self.max_length = max_length

            self.dev_ratio = dev_num / len(filename_list)

            self.idx_dict["min_length"] = self.min_length

            print("Min Length: ", min_length)

            self.idx_dict['max_length'] = self.max_length

            self.idx_dict['force_mean'] = np.zeros(6)
            self.idx_dict['force_std'] = np.zeros(6)
            self.count = 0
            self.num_objects = cfg['dataloading_params']['num_objects']
            self.idx_dict['num_objects'] = self.num_objects

            for filename in filename_list:
                # print(filename)
                train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 1) ### 1 is train, 0 is val
                dev_bool = np.random.binomial(1, 1 - self.dev_ratio, 1) ### 1 is train, 0 is val
                dataset = read_h5(filename)

                forces = np.array(dataset['force_hi_freq'][:,:,:6])

                if 'proprio' in dataset.keys():
                    proprio = np.array(dataset['proprio'])
                else:
                    proprio = np.array(dataset['rel_proprio'])
                    
                length = proprio.shape[0]
 
                dataset.close()               

                if length <= self.min_length:
                    continue

                # if self.max_length < length:
                #     self.max_length = int(length)

                if train_val_bool == 1:
                    self.idx_dict['force_mean'] += np.sum(np.sum(forces, axis = 0), axis = 0)
                    self.idx_dict['force_std'] += np.sum(np.sum(np.square(forces), axis=0), axis=0)
                    self.count += forces.shape[0] * forces.shape[1]

                    self.idx_dict['train'][self.train_length] = filename
                    self.train_length += 1
                else:
                    self.idx_dict['val'][self.val_length] = filename
                    self.val_length += 1

                if dev_bool == 0:
                    self.idx_dict['dev'][self.dev_length] = filename
                    self.dev_length += 1

            self.idx_dict["max_length"] = self.max_length

            self.idx_dict['force_std'] = np.sqrt((self.idx_dict['force_std'] -\
             np.square(self.idx_dict['force_mean']) / self.count ) / self.count)

            self.idx_dict['force_mean'] = self.idx_dict['force_mean'] / self.count

            # print("Mean ", self.idx_dict['force_std'])
            # print("std ", self.idx_dict['force_mean'])
            ##########################################################    
            ##### End of Project Specific Code #######################
            ##########################################################
        else:

            self.idx_dict = idx_dict
            
            self.train_length = len(list(self.idx_dict['train'].keys()))
            self.val_length = len(list(self.idx_dict['val'].keys()))
            self.dev_length = len(list(self.idx_dict['dev'].keys()))

        for key in cfg['info_flow'].keys():
            # cfg['info_flow'][key]['init_args']['num_policies'] = len(self.idx_dict["policies"])
            cfg['info_flow'][key]['init_args']['force_mean'] = self.idx_dict['force_mean'].tolist()
            cfg['info_flow'][key]['init_args']['force_std'] = self.idx_dict['force_std'].tolist()

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
        # prev_time = time.time()
        dataset = read_h5(self.idx_dict[key_set][idx])

        # max_length = 200
        # min_length = self.idx_dict["min_length"]
        # idx0 = 0
        # idx1 = np.random.choice(range(min_length, max_traj_length)) # end idx

        max_traj_length = np.array(dataset['proprio']).shape[0]

        max_length = self.idx_dict["max_length"]
        min_length = self.idx_dict["min_length"]

        idx0 = np.random.choice(max_traj_length - min_length) # beginning idx
        idx1 = np.random.choice(range(idx0 + min_length, min(idx0 + max_length, max_traj_length))) # end idx
        # print('\n', idx0, ' ', idx1)

        padded = max_length - idx1 + idx0 + 1
        unpadded = idx1 - idx0 - 1

        sample = {}
        sample['input_length'] = np.array(unpadded)

        for key in self.dataset_keys:
            if key == 'action':
                sample[key] = np.array(dataset[key][idx0:(idx1 - 1)]) # each action corresponds to the action causing the difference recorded
                sample['final_action'] = np.array(dataset[key][idx1 - 1])

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)

            elif key == 'force_hi_freq':
                sample[key] = np.array(dataset[key][(idx0 + 1):idx1])

                sample['final_force'] = sample[key][-1,-1]
                sample['next_force'] = np.array(dataset[key][idx1,-1])

                sample[key + '_unpaired'] = shuffle_along_axis(sample[key], 0)

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1], sample[key].shape[2]))], axis = 0)

                sample[key + '_unpaired'] = np.concatenate([sample[key + '_unpaired'],\
                 np.zeros((padded, sample[key + '_unpaired'].shape[1], sample[key + '_unpaired'].shape[2]))], axis = 0)

            elif key == 'proprio' or key == 'rel_proprio':   
                sample[key] = np.array(dataset[key][idx0:idx1])

                # this code is here if you are using the dataset used for the paper where the position recorded
                # for the robot endeffector in proprio is the same as the position recorded for the relative proprio
                # they both record the relative position instead of proprio recording the global position
                # this bug is fixed in the data collection code and a new data set has already been collected
                if key == 'rel_proprio':
                    sample['final_rel_pos'] =  100 * sample[key][-1,:2]
                    hole_info = np.array(dataset['hole_info'])
                    hole_idxs = np.array(dataset['hole_info'])[:,0]

                    # print(hole_idxs == sample['state_idx'])
                    cand_idx = np.where(hole_idxs == sample['state_idx'], np.ones_like(hole_idxs), np.zeros_like(hole_idxs)).argmax(0)
                    # print(cand_idx)
                    sample['object_pos'] = hole_info[cand_idx][4:6]
                    sample['final_pos'] = sample['final_rel_pos'] + sample['object_pos']
                else:
                    sample['object_pos'] = 100 * np.array(dataset['object_pos'])[:2] 
                    sample['final_pos'] = 100 * sample[key][-1,:2]

                # this is the relative position estimate according to the initial position belief from the simulated
                # vision based object detector used to collect this trajectory
                sample['rel_pos_estimate'] = 100 * np.array(dataset[key][idx1-1,:2] - dataset[key][0,:2]) #100 * np.random.uniform(low=-0.03, high = 0.03, size = 2) #(sample[key][-1,:2] - sample[key][0,:2])
                # sample['rel_pos_estimate'] = 100 * np.array(dataset[key][idx1-1,:2] - dataset[key][idx0,:2]) #100 * np.random.uniform(low=-0.03, high = 0.03, size = 2) #(sample[key][-1,:2] - sample[key][0,:2])

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)

            elif key == 'contact':
                sample[key] = np.array(dataset[key][(idx0 + 1):idx1]).astype(np.int32)

                sample['final_contact_idx'] = np.array(dataset[key][idx1-1,0]).astype(np.int32)
                sample['next_contact_idx'] = np.array(dataset[key][idx1,0]).astype(np.int32)

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)

            elif key == 'peg_vector':
                # print("peg vector: ", np.array(dataset[key]))
                sample[key] = np.array(dataset[key])[:self.idx_dict['num_objects']]
                sample["tool_vector"] = sample[key]
                sample["tool_idx"] = np.array(sample[key].argmax(0))

            elif key == 'hole_vector':
                # print("hole vector: ", np.array(dataset[key]))
                sample[key] = np.array(dataset[key])[:self.idx_dict['num_objects']]
                sample["state_vector"] = sample[key]
                sample["state_idx"] = np.array(sample[key].argmax(0))
                
                sample["state_prior"] = np.random.uniform(low=0, high=1, size = sample[key].shape[0])
                sample['state_prior'] = sample['state_prior'] / np.sum(sample['state_prior'])

        if sample['tool_idx'] == sample['state_idx']:
            tool_list = list(range(self.idx_dict['num_objects']))
            tool_list.remove(sample["tool_vector"].argmax(0))
            sample['fit_idx'] = np.array(0)

            sample['new_tool_idx'] = np.array(random.choice(tool_list))
        else:
            sample['new_tool_idx'] = sample['state_idx']
            sample['fit_idx'] = np.array(1)

        if sample['new_tool_idx'] == sample['state_idx']:
            sample['new_fit_idx'] = np.array(0)
        else:
            sample['new_fit_idx'] = np.array(1)

        sample["padding_mask"] = np.concatenate([np.zeros(unpadded), np.ones(padded)])     

        # theta = np.random.uniform(low=0.0, high=2*np.pi)
        # r = np.random.uniform(low=0.0, high =2.0)

        # sample['pos_prior_mean'] = np.array([r * np.cos(theta), r * np.sin(theta)]) + sample['object_pos']
        sample['pos_prior_mean'] = np.random.uniform(low=-2.0, high =2.0, size=2) + sample['object_pos']

        
        sample['pos_prior_var'] = np.ones(2)

        # print(sample['pos_prior_mean'] - sample['object_pos'])
        # print(sample['initial_rel_pos'])
        # print(sample['final_pos'])
        # print(sample['final_rel_pos'])
        # print(sample['object_pos'])

        dataset.close()
        # dataset_unpaired.close()

        # a = input(" ")
        ##########################################################    
        ##### End of Project Specific Code #######################
        ##########################################################
        # print(time.time() - prev_time)
        return self.transform(sample)
