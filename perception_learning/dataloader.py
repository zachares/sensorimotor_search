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

            self.idx_dict["policies"] = []

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

                # if np.array(dataset['peg_idx'])[0] >= self.num_objects or np.array(dataset['hole_idx'])[0] >= self.num_objects:
                #     continue

                # policy = dataset["policy"][-1,0]

                forces = np.array(dataset['force_hi_freq'][:,:,:6])

                rel_proprio = np.array(dataset['rel_proprio'])
                length = rel_proprio.shape[0]

                # tolerance = 0.025

                # if np.linalg.norm(rel_proprio[0,:2]) > tolerance:
                #     print(filename)
                #     continue
 
                dataset.close()               

                # if policy not in self.idx_dict["policies"]:
                #     self.idx_dict["policies"].append(policy)

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

        # if idx == 0:
        #     idx_unpaired = idx + 1
        # elif idx == (len(self.idx_dict[key_set].keys()) - 1):
        #     idx_unpaired = idx - 1
        # else:
        #     idx_unpaired = idx + random.choice([-1,1])

        # dataset_unpaired = read_h5(self.idx_dict[key_set][idx_unpaired])

        max_traj_length = np.array(dataset['proprio']).shape[0]

        max_length = self.idx_dict["max_length"]
        min_length = self.idx_dict["min_length"]

        idx0 = np.random.choice(max_traj_length - min_length) # beginning idx
        idx1 = np.random.choice(range(idx0 + min_length, min(idx0 + max_length, max_traj_length))) # end idx

        padded = max_length - idx1 + idx0 + 1
        unpadded = idx1 - idx0 - 1

        sample = {}
        sample['input_length'] = np.array(unpadded)

        # pixel_shift = np.random.choice(range(-10,11), 2)

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

            elif key == 'rel_proprio':   
                sample[key] = np.array(dataset[key][idx0:idx1])

                # sample['rel_pos_prior_mean'] = 100 * (sample[key][-1,:2] - np.array(dataset[key])[0,:2])
                # sample['rel_pos_prior_var'] = np.square(np.random.uniform(low=1e-1, high =2, size = 2)) 

                sample['rel_pos_prior_mean'] = 100 * np.array(dataset[key][idx1-1,:2] - dataset[key][0,:2]) #np.random.uniform(low=-0.03, high = 0.03, size = 2)
                sample['rel_pos_prior_var'] = np.ones(2) #np.square(np.random.uniform(low=1e-1, high =3, size = 2))

                sample['final_rel_pos'] = 100 * sample[key][-1,:2]

                sample['rel_pos_estimate'] = 100 * np.array(dataset[key][idx1-1,:2] - dataset[key][0,:2]) #100 * np.random.uniform(low=-0.03, high = 0.03, size = 2) #(sample[key][-1,:2] - sample[key][0,:2])

                sample['next_rel_pos'] = 100 * np.array(dataset[key][idx1, :2])

                sample['reference_pos'] = 100 * np.array(dataset[key][idx1-1,:2] - dataset[key][0,:2])

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)

            elif key == 'proprio':   
                sample[key] = np.array(dataset[key][idx0:idx1])

                sample[key + '_unpaired'] = sample[key] #shuffle_along_axis(sample[key], 0)

                sample['final_pos'] = sample[key][-1,:2]
                sample['next_pos'] = np.array(dataset[key][idx1, :2])
                
                sample['final_pos_change'] = 1000 * sample[key][-1,:2] - sample[key][-2,:2]
                sample['next_pos_change'] = 1000 * np.array(dataset[key][idx1, :2]) - sample[key][-1,:2]

                sample[key] = np.concatenate([sample[key], np.zeros((padded, sample[key].shape[1]))], axis = 0)
                sample[key + '_unpaired'] = np.concatenate([sample[key + '_unpaired'], np.zeros((padded, sample[key + '_unpaired'].shape[1]))], axis = 0)

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

                tool_list = list(range(self.idx_dict['num_objects']))
                tool_list.remove(sample[key].argmax(0))

                sample['new_tool_idx'] = np.array(random.choice(tool_list))

            elif key == 'hole_vector':
                # print("hole vector: ", np.array(dataset[key]))
                sample[key] = np.array(dataset[key])[:self.idx_dict['num_objects']]
                sample["state_vector"] = sample[key]
                sample["state_idx"] = np.array(sample[key].argmax(0))
                
                sample["state_prior"] = np.random.uniform(low=0, high=1, size = sample[key].shape[0])
                sample['state_prior'] = sample['state_prior'] / np.sum(sample['state_prior'])

                # sample['type_particles_idx'] = np.random.randint(sample[key].shape[0], size = num_particles)

                # sample['weights'] = np.random.uniform(low=0, high=1, size = num_particles)
                # sample['weights_particles'] = sample['weights'] / np.sum(sample['weights'])

            #     if np.sum(np.array(dataset['done'][(idx0 + 1):idx1])) > 0:
            #         sample['done_mask'] = np.zeros((sample[key].size))
            #     else:
            #         sample['done_mask'] = np.ones((sample[key].size))

            elif key == 'fit_vector':
                sample[key] = np.array(dataset[key])
                # print("Fit Vector: ", sample[key])
                sample["obs_vector"] = sample[key]
                sample["fit_idx"] = np.array(sample[key].argmax(0))
                # sample["obs_idx"] = np.array(sample[key].argmax(0))
                
                # if np.sum(np.array(dataset['done'][(idx0 + 1):idx1])) > 0:
                #     sample['done_mask'] = np.zeros((sample[key].size))
                # else:
                #     sample['done_mask'] = np.ones((sample[key].size))

        if sample['new_tool_idx'] == sample['state_idx']:
            sample['new_fit_idx'] = np.array(0)
        else:
            sample['new_fit_idx'] = np.array(1)

        sample["padding_mask"] = np.concatenate([np.zeros(unpadded), np.ones(padded)])
        # sample["pol_idx"] = np.array(dataset['policy'][-1,0])

        dataset.close()
        # dataset_unpaired.close()

        # a = input(" ")
        ##########################################################    
        ##### End of Project Specific Code #######################
        ##########################################################
        # print(time.time() - prev_time)
        return self.transform(sample)


            # elif key == 'hole_info':
            #     sample['hole_sites'] = np.array(dataset[key])[:,:2] - np.array([[0.5, 0.0],[0.5, 0.0],[0.5, 0.0]])

            #     pixels = np.array(dataset[key])[:,-2:]

            #     sample['pixels'] = pixels + pixel_shift

            #     sample['heat_map_idx'] = np.zeros((3,128,128))
            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]), int(pixels[0,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]), int(pixels[1,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]), int(pixels[2,1] + pixel_shift[1])] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) + 1, int(pixels[0,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) + 1, int(pixels[1,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) + 1, int(pixels[2,1] + pixel_shift[1]) + 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) - 1, int(pixels[0,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) - 1, int(pixels[1,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) - 1, int(pixels[2,1] + pixel_shift[1]) + 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) + 1, int(pixels[0,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) + 1, int(pixels[1,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) + 1, int(pixels[2,1] + pixel_shift[1]) - 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) - 1, int(pixels[0,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) - 1, int(pixels[1,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) - 1, int(pixels[2,1] + pixel_shift[1]) - 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]), int(pixels[0,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]), int(pixels[1,1] + pixel_shift[1]) + 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]), int(pixels[2,1] + pixel_shift[1]) + 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) + 1, int(pixels[0,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) + 1, int(pixels[1,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) + 1, int(pixels[2,1] + pixel_shift[1])] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]), int(pixels[0,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]), int(pixels[1,1] + pixel_shift[1]) - 1] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]), int(pixels[2,1] + pixel_shift[1]) - 1] = 1.0

            #     sample['heat_map_idx'][0, int(pixels[0,0] + pixel_shift[0]) - 1, int(pixels[0,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][1, int(pixels[1,0] + pixel_shift[0]) - 1, int(pixels[1,1] + pixel_shift[1])] = 1.0
            #     sample['heat_map_idx'][2, int(pixels[2,0] + pixel_shift[0]) - 1, int(pixels[2,1] + pixel_shift[1])] = 1.0

            #     # print(np.array(dataset[key])[:,:3])

            # elif key == 'reference_image':
            #     image = np.array(dataset[key]) / 255.0
            #     sample[key] = add_noise_and_shift(image, noise = 0.03, shift_x = pixel_shift[0] , shift_y = pixel_shift[1])

            # elif key == 'reference_point_cloud':
            #     sample[key] = add_noise_and_shift(np.array(dataset[key]), noise = 0.005, shift_x = pixel_shift[0] , shift_y = pixel_shift[1])

            # elif key == 'reference_depth':
            #     sample[key] = add_noise_and_shift(np.array(dataset[key]), noise = 0.03, shift_x = pixel_shift[0] , shift_y = pixel_shift[1])
        # sample["correct_site"] = sample['hole_sites'][sample['state_idx']]
        # sample['point_cloud_point_est'] = sample['reference_point_cloud']\
        # [int(pixels[sample['state_idx'],0] + pixel_shift[0]), int(pixels[sample['state_idx'],1] + pixel_shift[1]), :2]

        # sample['point_cloud_point_ests'] = np.concatenate([\
        #     np.expand_dims(sample['reference_point_cloud'][int(pixels[0,0] + pixel_shift[0]), int(pixels[0,1] + pixel_shift[1]), :2], axis = 0),\
        #     np.expand_dims(sample['reference_point_cloud'][int(pixels[1,0] + pixel_shift[0]), int(pixels[1,1] + pixel_shift[1]), :2], axis = 0),\
        #     np.expand_dims(sample['reference_point_cloud'][int(pixels[2,0] + pixel_shift[0]), int(pixels[2,1] + pixel_shift[1]), :2], axis = 0)\
        #     ], axis = 0)
        # print("point cloud est 0", sample['reference_point_cloud'][int(pixels[0,0] + pixel_shift[0]), int(pixels[0,1] + pixel_shift[1])])
        # print("point cloud est 1", sample['reference_point_cloud'][int(pixels[1,0] + pixel_shift[0]), int(pixels[1,1] + pixel_shift[1])])
        # print("point cloud est 2", sample['reference_point_cloud'][int(pixels[2,0] + pixel_shift[0]), int(pixels[2,1] + pixel_shift[1])])

        # print("point cloud est 0", np.array(dataset['reference_point_cloud'])[int(pixels[0,0]), int(pixels[0,1])])
        # print("point cloud est 1", np.array(dataset['reference_point_cloud'])[int(pixels[1,0]), int(pixels[1,1])])
        # print("point cloud est 2", np.array(dataset['reference_point_cloud'])[int(pixels[2,0]), int(pixels[2,1])])

        # plot_image(np.rot90(np.transpose(sample["rgbd_last"][:-1], (1,2,0)), k = 1))
        # a = input("")