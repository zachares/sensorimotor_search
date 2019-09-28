import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
import models_utils


def sample_gaussian(m, v, device):
    
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z

def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

#######################################
# Defining Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel

#### observation encoder for mapping from the image data and force data to low dimensional latent space
class Observation_Encoder(Proto_Macromodel):
    def __init__(self, model_name, load_bool, image_size, force_size, z_dim, device = None):
        super().__init__()
        ##### the Proto_Macromodel class initializes a model list

        #### image size = (RGB Values, Image Height, Image Width)
        #### force size = (Forces and Moments, Time Series)

        self.device = device

        self.image_encoder = CONV2DN(model_name + "_image_encoder", load_bool, image_size[0], z_dim, image_size[1], 1, image_size[2], 1, False, True, 3, device = device)

        self.force_encoder = CONV1DN(model_name + "_force_encoder", load_bool, force_size[0], z_dim, force_size[1], 1, False, True, 3, device = device)

        self.modality_fusion = FCN(model_name + "_modality_fusion", load_bool, z_dim * 2, z_dim, 3, device = device)

        self.model_list.append(self.image_encoder)
        self.model_list.append(self.force_encoder)
        self.model_list.append(self.modality_fusion)

    def forward(self, inputs):

        image, forces = inputs
        
        z_image = self.image_encoder(image)
        z_force = self.force_encoder(forces)

        return self.modality_fusion(torch.cat((z_image.transpose(0,1), z_force.transpose(0,1))).transpose(0,1))

class Confidence_Predictor(Proto_Macromodel):
    def __init__(self, model_name, load_bool, z_dim, device = None):
        super().__init__()
        ##### the Proto_Macromodel class initializes a model list

        self.device = device

        self.confidence_predictor = FCN(model_name + "_confidence_prediction", load_bool, z_dim + 1, 1, 3, device = device)

        self.model_list.append(self.confidence_predictor)

    def forward(self, inputs):

        z, goals = inputs

        logits = self.confidence_predictor(torch.cat((z.transpose(0,1), goals.transpose(0,1))).transpose(0,1))

        return logits, torch.sigmoid(logits)

class Switching_Policy(Proto_Macromodel):
    def __init__(self, model_name, load_bool, z_dim, num_goals, threshold = 0.5, device = None):
        super().__init__()
        ##### the Proto_Macromodel class initializes a model list

        self.device = device

        self.threshold = threshold
        self.num_goals = num_goals

        self.switching_policy = FCN(model_name + "_switching_policy", load_bool, z_dim, num_goals - 1, 3, device = device)

        self.model_list.append(self.switching_policy)

    def forward(self, inputs):

        z, goals, confidence_probs = inputs

        ##### setting new exploration area if confidence score is less than the confidence score        
        if self.num_goals == 2:
            delta_goal = torch.where(confidence_probs < self.threshold, torch.ones_like(goals), torch.zeros_like(goals))

        else:
            delta_goal = torch.where(confidence_probs < self.threshold, self.switching_policy(z).multinomial(1) + 1, torch.zeros_like(goals))

        return (delta_goal + goals) % self.num_goals #### defining the new goal areas for each datapoint in the batch

class Motion_Policy(Proto_Macromodel):
    def __init__(self, model_name, load_bool, z_dim, num_goals, threshold = 0.5, device = None):
        super().__init__()

    def forward(self, inputs)

        z, goals, pos, goal_volumes = inputs

        delta_params = self.motion_policy(torch.cat((z.transpose(0,1), new_goals.transpose(0,1))).transpose(0,1))

        delta_mu, delta_var = gaussian_parameters(delta_params, dim = 1)

        delta_sample = sample_gaussian(delta_mu, delta_var, self.device)

        

        lower_goal_bound = goal_bounds[np.arange(goal_bounds.size()[0]), new_goals, 0].unsqueeze(1).repeat(1, self.num_goals)
        upper_goal_bound = goal_bounds[np.arange(goal_bounds.size()[0]), new_goals, 1].unsqueeze(1).repeat(1, self.num_goals)

        action_probs_lb_clipped = torch.where(x_pos_large < lower_goal_bound, right_action, action_probs_pred)
        action_probs_clipped = torch.where(x_pos_large > upper_goal_bound, left_action, action_probs_lb_clipped)

        return action_probs_clipped

