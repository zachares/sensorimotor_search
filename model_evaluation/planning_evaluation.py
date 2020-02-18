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
from logger import Logger

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
from scipy.misc import imresize as resize

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
    def __init__(self, num_steps, action_dim, mean, var, device, num_elite_samples, env):
        self.env = env
        self.size = (num_steps, action_dim[0])
        self.mean = (torch.zeros(self.size) * mean).to(device)
        self.var = (torch.ones(self.size) * var).to(device)
        self.device = device 
        self.nes = num_elite_samples
        self.obj_funct = nn.MSELoss(reduction = 'none')

    def clip_actions(self, action_samples):
        # print("Before")
        # print(action_samples[0,0,:])
        # print("After")
        # print(self.env.controller.transform_action(action_samples)[0,0,:])
        return self.env.controller.transform_action(action_samples)

    def sample(self, num_samples):
        action_samples =  sample_gaussian(self.mean.unsqueeze(0).repeat(num_samples, 1, 1),\
            self.var.unsqueeze(0).repeat(num_samples, 1, 1), self.device)
        return torch.tensor(self.clip_actions(action_samples.detach().cpu().numpy())).to(self.device).float()

    def cem_update(self, samples, pos, goal):
        # print(pos.size())
        # print(goal.size())
        scores = self.obj_funct(pos.cpu(), goal.cpu()).sum(1)
        sorted_scores, idxs = torch.sort(scores)

        self.top_score = sorted_scores[0]
        self.opt_sample = samples[idxs[0]]

        self.mean = samples[idxs[:self.nes]].mean(dim = 0)
        self.var = samples[idxs[:self.nes]].var(dim = 0)

if __name__ == '__main__':
	#####################################################################
	### Loading run parameters
	#####################################################################
    with open("planning_params.yml", 'r') as ymlfile:
        cfg1 = yaml.safe_load(ymlfile)

    use_cuda = cfg1['evaluation_params']['use_GPU'] and torch.cuda.is_available()
    seed = cfg1['evaluation_params']['seed']

    debugging_val = cfg1['debugging_params']['debugging_val']

    parameters_path = cfg1['evaluation_params']['parameters_dictionary']

    num_samples = cfg1['evaluation_params']['num_samples']
    num_elite_samples = cfg1['evaluation_params']['num_elite_samples']
    num_steps = cfg1['evaluation_params']['num_steps'] 
    num_trials = cfg1['evaluation_params']['num_trials']
    num_iterations = cfg1['evaluation_params']['num_iterations']

    peg_type = cfg1['evaluation_params']['peg_type']

    info_flow = cfg1['info_flow']

    with open(parameters_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    image_size = info_flow['dataset']['outputs']['image']
    force_size =info_flow['dataset']['outputs']['force'] 
    action_dim =info_flow['dataset']['outputs']['action']
    proprio_size = info_flow['dataset']['outputs']['proprio']
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
    # model = Simple_Multimodal("", "Simple_Multimodal_Hist1", info_flow, image_size, proprio_size, z_dim,\
    #  action_dim, device = device).to(device)
    # model_name = "Simple_Multimodal_Hist1"

    # model = Simple_Multimodal("", "Simple_Multimodal", info_flow, image_size, proprio_size, z_dim,\
    #  action_dim, device = device).to(device)
    # model_name = "Simple_Multimodal"

    # model = Simple_Multimodal("", "Simple_Multimodal_Reg", info_flow, image_size, proprio_size, z_dim,\
    #  action_dim, device = device).to(device)
    # model_name = "Simple_Multimodal_Reg"

    model = LSTM_Multimodal("", "LSTM_Multimodal", info_flow, image_size, proprio_size, z_dim,\
     action_dim, device = device).to(device)
    model_name = "LSTM_Multimodal"

    # encoder = VAE_Multimodal("", "VAE_Multimodal", info_flow, image_size, proprio_size, z_dim,\
    #  action_dim, device = device).to(device)
    # dynamics = Dynamics_DetModel("", "VAE_Dynamics", info_flow, z_dim,\
    #  action_dim, device = device).to(device)  
    # model_name = "VAE_Multimodal"

    model.eval()
    encoder = model
    dynamics = model
    ##################################################################################
    #### Logging tool to save scalars, images and models during training#####
    ##################################################################################
    save_model_flag = False
    logger = Logger(cfg1, debugging_flag, save_model_flag)
    logging_dict = {}
    #################################################################################
    ### Loading Environment
    #################################################################################
    env = robosuite.make("PandaPegInsertion", has_renderer=True, ignore_done=True,\
     use_camera_obs=True, gripper_visualization=True, control_freq=10,\
      gripper_type = peg_type + "PegwForce", controller='position')

    obs = env.reset()
    env.viewer.set_camera(camera_id=2)

    peg_bottom_site = peg_type + "Peg_bottom_site"
    peg_top_site = peg_type + "Peg_top_site"
    offset = np.array([0, 0, 0.05])
    top_goal = np.concatenate([env._get_sitepos(peg_top_site) + offset, np.array([np.pi, 0, np.pi])])
    goal = np.concatenate([top_goal + np.array([0,0,0.04, 0,0,0]), np.array([np.pi, 0, np.pi])])

    # obs_keys = [ "image", "force", "proprio", "action_sequence"]
    obs_keys = [ "image", "force_hi_freq", "proprio", "action_sequence"]
    to_tensor = ToTensor(device, obs_keys)

    tol = 0.001
    tol_ang = 100
    kp = 10

    prev_time = time.time()
    for global_cnt in range(1, num_trials + 1):
        current_time = time.time()
        logging_dict['scalar'] = {}
        logging_dict['image'] = {}
        logging_dict['image'][model_name]  = []       

        if global_cnt != 1:
            print("Trial took ", current_time - prev_time, " seconds")
            prev_time = time.time()

        while env._check_poserr(goal, tol, tol_ang) == False:
            action, action_euler = env._pose_err(goal)
            pos_err = kp * action_euler[:3]
            obs, reward, done, info = env.step(pos_err)

        print("Reached Goal Position")

        goal_pos = obs['proprio'][:3]
        goal_z = encoder.encode(to_tensor.convert(obs))['latent_state'].squeeze().unsqueeze(0)

        random_action_model = Generative_Action_Model(num_steps, action_dim, 0, 1, device, num_elite_samples, env)

        action_sequence = random_action_model.sample(1).squeeze().detach().cpu().numpy()

        for idx in range(action_sequence.shape[0]):
            obs, reward, done, info = env.step(action_sequence[0])

        init_pos = obs['proprio'][:3]

        tensor_obs = to_tensor.convert(obs)

        init_z = encoder.encode(to_tensor.convert(obs))['latent_state'].squeeze().unsqueeze(0)
        tensor_obs['z'] = init_z

        if global_cnt == 1:
            init_score = random_action_model.obj_funct(init_z.cpu(), goal_z.cpu()).sum(1)

        dest_z = torch.zeros((num_samples, init_z.size(1)))
        # print("Dest z: ", dest_z.size())
        for idx_steps in range(num_steps):
            action_model = Generative_Action_Model(num_steps, action_dim, 0, 10, device, num_elite_samples, env)

            for iteration in range(num_iterations):
                action_sequences = action_model.sample(num_samples)

                for idx in range(action_sequences.size(0)):
                    action_sequence = action_sequences[idx]
                    tensor_obs['action_sequence'] = action_sequence
                    dest_z[idx]  = dynamics.trans(tensor_obs)['latent_state'].detach()

                action_model.cem_update(action_sequences, dest_z, goal_z.repeat(num_samples, 1))

            if idx_steps == 0:
                top_score = action_model.top_score

            opt_action_sequence = action_model.opt_sample.detach().cpu().numpy()
            obs, reward, done, info = env.step(opt_action_sequence[0]) 

        final_pos = obs['proprio'][:3]
        tensor_obs = to_tensor.convert(obs)
        final_z = encoder.encode(tensor_obs)['latent_state'].squeeze().unsqueeze(0)

        if not debugging_flag:
            z_goal = goal_z.squeeze().detach().cpu().numpy()
            z_final = final_z.squeeze().detach().cpu().numpy()
            z_init = init_z.squeeze().detach().cpu().numpy()

            logging_dict['scalar'][model_name + "/norm_distance_error"] = np.linalg.norm(final_pos - goal_pos) / np.linalg.norm(init_pos - goal_pos)
            # logging_dict['scalar'][model_name + "/norm_distance_travelled"] = np.linalg.norm(final_pos - init_pos) / np.linalg.norm(init_pos - goal_pos)

            logging_dict['scalar'][model_name + "/norm_latent_distance_error"] = np.linalg.norm(z_goal - z_final) / np.linalg.norm(z_init - z_goal)
            # logging_dict['scalar'][model_name + "/norm_latent_distance_travelled"] = np.linalg.norm(z_init - z_final) / np.linalg.norm(z_init - z_goal)

            logging_dict['scalar'][model_name + "/Minimum Mean Squared Error"] = top_score / init_score            

            logger.save_scalars(logging_dict, global_cnt, 'evaluation/')
            logger.save_images2D(logging_dict, global_cnt, 'evaluation/')

        if global_cnt % 10 == 0:
            print("Number of trials conducted: ", global_cnt)

    print("Finished Evaluation")













