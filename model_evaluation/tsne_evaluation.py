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
from torch.distributions import Normal


from tensorboardX import SummaryWriter
import yaml

sys.path.insert(0, "../learning/models/") 
sys.path.insert(0, "../robosuite/") 
sys.path.insert(0, "../learning/datalogging/") 
sys.path.insert(0, "../learning/supervised_learning/") 

from models import *
from dataloader import *
from logger import Logger

from scipy.misc import imresize as resize
from scipy.cluster.vq import vq, kmeans, whiten

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device, key_list):
        self.device = device
        self.key_list = key_list

    def convert(self, sample):
        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if k in self.key_list:
                # print(v.shape)
                if k == "image":
                    v = np.rot90(v, k = 2)
                    v = resize(v, (128, 128))
                    v = np.transpose(v, (2, 1, 0))

                new_dict[k] = torch.from_numpy(v).float().to(device).unsqueeze(0).unsqueeze(1)

                # print(new_dict[k].size())

        return new_dict

def sample_gaussian(m, v, device):
    epsilon = Normal(0, 1).sample(m.size())
    return m + torch.sqrt(v) * epsilon.to(device)

class Generative_Action_Model(object):
    def __init__(self, num_steps, action_dim, mean, var, device, num_elite_samples):
        self.size = (num_steps, action_dim[0])
        self.mean = (torch.ones(self.size) * mean).to(device)
        self.var = (torch.ones(self.size) * var).to(device)
        self.device = device 
        self.nes = num_elite_samples
        self.obj_funct = nn.MSELoss(reduction = 'none')

    def sample(self, num_samples):
        return sample_gaussian(self.mean.unsqueeze(0).repeat(num_samples, 1, 1),\
            self.var.unsqueeze(0).repeat(num_samples, 1, 1), self.device)

    def cem_update(self, samples, pos, goal):
        scores = self.obj_funct(pos.cpu(), goal.cpu()).sum(1)
        sorted_scores, idxs = torch.sort(scores)

        self.top_score = sorted_scores[0]
        self.opt_sample = samples[idxs[0]]

        self.mean = samples[idxs[:self.nes]].mean(dim = 0)
        self.var = samples[idxs[:self.nes]].var(dim = 0)

def print_size(tensor, label):
    print(label, " size: ", tensor.size())

def find_closest_centroid(centroids, points):
    distance_list = []

    for idx in range(centroids.shape[0]):
        centroid = np.expand_dims(centroids[idx], axis = 0)
        distance_list.append(np.expand_dims(np.linalg.norm(points - centroid, axis = 1), axis = 1))

    return np.argmin(np.concatenate(distance_list, axis = 1),axis = 1)

def sort_centroids(centroids):
    norms = np.linalg.norm(centroids, axis = 1)
    print(norms.shape)
    idxs = np.argsort(norms)
    return centroids[idxs]

if __name__ == '__main__':
	#####################################################################
	### Loading run parameters
	#####################################################################
    with open("tsne_params.yml", 'r') as ymlfile:
        cfg1 = yaml.safe_load(ymlfile)

    use_cuda = cfg1['evaluation_params']['use_GPU'] and torch.cuda.is_available()
    seed = cfg1['evaluation_params']['seed']

    debugging_val = cfg1['debugging_params']['debugging_val']

    model_folder = cfg1['evaluation_params']['model_folder']
    epoch_num = cfg1['evaluation_params']['epoch_num']

    parameters_path = cfg1['evaluation_params']['parameters_dictionary']
    train_val_split_path = cfg1['evaluation_params']['train_val_split_path']

    small_k = cfg1['evaluation_params']['small_k']
    medium_k = cfg1['evaluation_params']['medium_k']
    large_k = cfg1['evaluation_params']['large_k']

    with open(parameters_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    info_flow = cfg['info_flow']
    image_size = info_flow['dataset']['outputs']['image']
    force_size =info_flow['dataset']['outputs']['force'] 
    action_dim =info_flow['dataset']['outputs']['action']
    z_dim = cfg["model_params"]["z_dim"] 
    ##################################################################################
    ### Setting Debugging Flag
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
    ##################################################################################
    ### Hardware and Low Level Training Details
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
    #################################################################################
    ### Defining and Loading Latent Space Encoder Model
    #################################################################################
    encoder = PlaNet_Multimodal(model_folder + "PlaNet_Multimodal", image_size, force_size, z_dim, action_dim, device = device).to(device)
    encoder.load(epoch_num)
    encoder.eval()
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
    save_model_flag = False
    logger = Logger(cfg1, debugging_flag, save_model_flag, device)
    logging_dict = {}
    #################################################################################
    ### Loading Dataset
    #################################################################################
    train_val_split_path = None #### REMOVE THIS WHEN POSSIBLE ####
    data_loader, val_data_loader, idx_dict = init_dataloader(cfg, device, train_val_split_path)
    train_z_list = []
    train_eepos_list = []
    # val_z_list = []
    # val_eepos_list = []
    for i_iter, sample_batched in enumerate(data_loader):
        if (i_iter + 1) % 10 == 0:
            print(i_iter + 1, " iterations have been run")

        z = encoder.encode(sample_batched)['latent_state']
        ee_pos = sample_batched['proprio'][:,0,:3]
        train_z_list.append(z.detach().cpu().numpy())
        train_eepos_list.append(ee_pos.detach().cpu().numpy())

    print("Done encoding training set")
        # if i_iter == 0:
        #     print_size(ee_pos, "EEPOS")
        #     print_size(z, "Z")

    # for i_iter, sample_batched in enumerate(val_data_loader):
    #     z = encoder.encode(sample_batched)
    #     ee_pos = sample_batched['proprio'][:,0,:3]
    #     val_z_list.append(z)
    #     val_eepos_list.append(ee_pos)

    train_z_array = np.concatenate(train_z_list, axis = 0)
    train_eepos_array = np.concatenate(train_eepos_list, axis = 0)

    # val_z_array = torch.cat(val_z_list, dim = 0).detach().cpu().numpy()
    # val_eepos_array = torch.cat(val_eepos_list, dim = 0).detach().cpu().numpy()

    whitened_train_eepos_array = whiten(train_eepos_array)
    small_centroids, distortion = kmeans(whitened_train_eepos_array, small_k)
    medium_centroids, distortion = kmeans(whitened_train_eepos_array, medium_k)
    large_centroids, distortion = kmeans(whitened_train_eepos_array, large_k)
    small_centroids = sort_centroids(small_centroids)
    medium_centroids = sort_centroids(medium_centroids)
    large_centroids = sort_centroids(large_centroids)

    print("Finished K means calculations")

    small_labels = find_closest_centroid(small_centroids, whitened_train_eepos_array)
    medium_labels = find_closest_centroid(medium_centroids, whitened_train_eepos_array)
    large_labels = find_closest_centroid(large_centroids, whitened_train_eepos_array)

    print("Finished labelling data points")

    logger.save_tsne(train_z_array, [("global", small_labels), ("meso", medium_labels), ("local", large_labels), ("local", large_labels)], 0,'evaluation', False)

    print("Finished Evaluation")













