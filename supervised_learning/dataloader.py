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
## ADD data normalization

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

class H5_DataLoader(Dataset):
    ### h5py file type dataloader dataset ###
    def __init__(self, model_folder, loading_dict, val_ratio, num_trajectories, min_length, idx_dict = None, device = None, transform=None):
        self.device = device
        self.transform = transform
        self.loading_dict = loading_dict
        self.sample = {}

        self.model_folder = model_folder     

        self.val_bool = False
        self.val_ratio = val_ratio

        self.train_length = 0
        self.val_length = 0

        self.min_length = int(min_length)

        if idx_dict == None:
            self.idx_dict = {}
            self.idx_dict['val'] = {}
            self.idx_dict['train'] = {}
            val_eepos_list = []
            train_eepos_list = []
  
            self.max_length = 0

            print("Starting Train Val Split")

            for idx in range(num_trajectories):
                lengths = np.zeros(3)
                cross_cross = model_folder + "Cross_Cross_" + str(idx + 1).zfill(4) + ".h5" 
                dataset = self.load_file(cross_cross)
                proprios = np.array(dataset['proprio'])
                lengths[0] = proprios.shape[0]
                dataset.close()

                cross_rect = model_folder + "Cross_Rect_" + str(idx + 1).zfill(4) + ".h5" 
                dataset = self.load_file(cross_rect)
                proprios = np.array(dataset['proprio'])
                lengths[1] = proprios.shape[0]
                dataset.close()

                cross_square = model_folder + "Cross_Square_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(cross_square)
                proprios = np.array(dataset['proprio'])
                lengths[2] = proprios.shape[0]
                dataset.close() 

                cross_length = lengths.min()

                rect_cross = model_folder + "Rect_Cross_" + str(idx + 1).zfill(4) + ".h5" 
                dataset = self.load_file(rect_cross)
                proprios = np.array(dataset['proprio'])
                lengths[0] = proprios.shape[0]
                dataset.close()

                rect_rect = model_folder + "Rect_Rect_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(rect_rect)
                proprios = np.array(dataset['proprio'])
                lengths[1] = proprios.shape[0]
                dataset.close()

                rect_square = model_folder + "Rect_Square_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(rect_square)
                proprios = np.array(dataset['proprio'])
                lengths[2] = proprios.shape[0]
                dataset.close()

                rect_length = lengths.min()

                square_cross = model_folder + "Square_Cross_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(square_cross)
                proprios = np.array(dataset['proprio'])
                lengths[0] = proprios.shape[0]
                dataset.close()

                square_rect = model_folder + "Square_Rect_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(square_rect)
                proprios = np.array(dataset['proprio'])
                lengths[1] = proprios.shape[0]
                dataset.close()

                square_square = model_folder + "Square_Square_" + str(idx + 1).zfill(4) + ".h5"
                dataset = self.load_file(square_square)
                proprios = np.array(dataset['proprio'])
                lengths[2] = proprios.shape[0]
                dataset.close()

                square_length = lengths.min()

                if self.max_length < cross_length:
                    self.max_length = int(cross_length)

                if self.max_length < rect_length:
                    self.max_length = rect_length

                if self.max_length < square_length:
                    self.max_length = square_length

                train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 3) ### 1 is train, 0 is val

                if train_val_bool[0] == 1:
                    self.idx_dict['train'][self.train_length] = (cross_cross, int(cross_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 1] = (cross_rect, int(cross_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 2] = (cross_square, int(cross_length)) #, filename_ss)
                    self.train_length += 3  
                else:
                    self.idx_dict['val'][self.val_length] = (cross_cross, int(cross_length)) #, filename_ss)
                    self.idx_dict['val'][self.val_length + 1] = (cross_rect, int(cross_length)) #, filename_ss)
                    self.idx_dict['val'][self.val_length + 2] = (cross_square, int(cross_length)) #, filename_ss) 
                    self.val_length += 3          

                if train_val_bool[1] == 1:
                    self.idx_dict['train'][self.train_length] = (rect_cross, int(rect_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 1] = (rect_rect, int(rect_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 2] = (rect_square, int(rect_length)) #, filename_ss)
                    self.train_length += 3  
                else:
                    self.idx_dict['val'][self.val_length] = (rect_cross, int(rect_length)) #, filename_ss) 
                    self.idx_dict['val'][self.val_length + 1] = (rect_rect, int(rect_length)) #, filename_ss) 
                    self.idx_dict['val'][self.val_length + 2] = (rect_square, int(rect_length)) #, filename_ss)
                    self.val_length += 3   

                if train_val_bool[2] == 1:
                    self.idx_dict['train'][self.train_length] = (square_cross, int(square_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 1] = (square_rect, int(square_length)) #, filename_ss)
                    self.idx_dict['train'][self.train_length + 2] = (square_square, int(square_length)) #, filename_ss)
                    self.train_length += 3  
                else:
                    self.idx_dict['val'][self.val_length] = (square_cross, int(square_length)) #, filename_ss) 
                    self.idx_dict['val'][self.val_length + 1] = (square_rect, int(square_length)) #, filename_ss) 
                    self.idx_dict['val'][self.val_length + 2] = (square_square, int(square_length)) #, filename_ss) 
                    self.val_length += 3


            self.idx_dict["max_length"] = self.max_length
            self.idx_dict["min_length"] = self.min_length

        else:

            switch_bool = False

            if switch_bool:
                self.idx_dict = {}
                self.idx_dict['train'] = idx_dict['val']
                self.idx_dict['val'] = idx_dict['train']
                self.idx_dict["max_length"] = idx_dict["max_length"]
                self.idx_dict["min_length"] = idx_dict["min_length"]
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
        max_traj_length = self.idx_dict[key_set][idx][1]
        max_length = self.idx_dict["max_length"]
        min_length = self.idx_dict["min_length"]
        # print(max_traj_length)
        # print(max_length)
        # print(min_length)

        if max_traj_length <= min_length:
            idx0 = 0
            idx1 = max_traj_length
        else:
            idx0 = np.random.choice(max_traj_length - min_length) # beginning idx
            idx1 = np.random.choice(range(idx0 + min_length, max_traj_length)) # end idx

        padded = max_length - idx1 + idx0 + 1
        unpadded = idx1 - idx0 - 1
        # print(idx0)
        # print(idx1)
        # print(padded)
        # print(unpadded)
        # print(idx1 - idx0)
        sample = {}

        for key in self.loading_dict.keys():
            if key == 'pose_err':
                continue

            # print( key , " took ", time.time() - self.prev_time, " seconds")
            # self.prev_time = time.time()
            if key == 'action':
                sample[key] = np.array(dataset[key][idx0:(idx1 - 1)])
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)
                # print("Action size", np.array(sample[key]).shape)
            elif key == 'insertion':
                if np.sum(np.array([dataset[key]])) > 0:
                    sample[key] = np.array([1.0])
                else:
                    sample[key] = np.array([0.0])
                     
            elif key == 'force_hi_freq':
                sample[key] = np.array(dataset[key][(idx0 + 1):idx1])
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1], sample[key].shape[2]))], axis = 0)
                # print("Force size", np.array(sample[key]).shape)
            elif key == 'proprio':   
                sample[key] = 100 * np.array(dataset[key][idx0:idx1])
                sample[key + "_diff"] = sample[key][1:] - sample[key][:-1]
                error = np.random.normal(0.0, 0.75, 3)
                sample['init_pos'] = sample[key][0,:3]
                sample['final_pos'] = sample[key][-1,:3]
                sample['pose_delta'] = sample['final_pos'] - sample['init_pos']
                sample['pose_vect'] = np.concatenate([sample['init_pos'] + error, sample['final_pos'] + error, sample['pose_delta']])
                sample['errorinit_pos'] = sample['init_pos'] + error
                sample['pos_err'] = error
                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)
                sample[key + "_diff"] = np.concatenate([sample[key + "_diff"], np.zeros((padded, sample[key + "_diff"].shape[1]))], axis = 0)
                # print("Proprio size", np.array(sample[key]).shape)
                # print("Proprio diff size", np.array(sample[key + "_diff"]).shape)
            elif key == 'contact':
                sample[key] = np.array(dataset[key][idx0:idx1])
                sample[key + "_diff"] = sample[key][1:].astype(np.int16) - sample[key][:-1].astype(np.int16) 
                sample[key] = np.concatenate([sample[key], np.zeros((padded))], axis = 0)
                sample[key + "_diff"] = np.concatenate([sample[key + "_diff"], np.zeros((padded))], axis = 0)   
                # print("contact size", np.array(sample[key]).shape)
                # print("contact diff size", np.array(sample[key + "_diff"]).shape)            
            elif key == 'peg_type' or key == 'hole_type':
                sample[key] = np.array(dataset[key])
            elif key == 'final_point':
                sample["command_pos"] =  np.array(dataset[key][:3])


            # else:
            #     sample[key] = dataset[key][idx0:idx1] 
            #     print(key, " size", np.array(sample[key]).shape)

        dataset.close()

        sample["padding_mask"] = np.concatenate([np.zeros(unpadded), np.ones(padded)])
        sample["macro_action"] = np.concatenate([sample["init_pos"], sample["command_pos"]])
        # print(sample["padding_mask"].shape)
        # sample["length"] = np.array([unpadded - 1])
        # print("Length size: ", sample["length"].shape)

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
            if k == "padding_mask":
                new_dict[k] = torch.from_numpy(v).bool()
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict

def init_dataloader(cfg, device, idx_dict_path = None):
    ###############################################
    ########## Loading dataloader parameters ######
    ###############################################
    dataset_path = cfg['dataloading_params']['dataset_path']
    num_workers = cfg['dataloading_params']['num_workers']
    batch_size = cfg['dataloading_params']['batch_size']
    num_trajectories = cfg['dataloading_params']['num_trajectories']
    min_steps = cfg['model_params']['min_steps']

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

    dataset = H5_DataLoader(dataset_path, sample_keys, val_ratio, num_trajectories, min_steps,\
     idx_dict = idx_dict, device = device, transform=transforms.Compose([ToTensor(device = device)]))

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

