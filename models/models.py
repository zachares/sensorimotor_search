import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
from models_utils import *

### test a deterministic autoencoder with dynamics model - moving forward with this
### test with contact and without contact dynamics model - bad results
### test with contact and without contact dynamics model and forces - bad results

### test dynamics model regularization method - 2 hyperparameters
### test latent space regularization method

### test with both

### test with selfsupervised training objectives
### test variational version
### test bayes filter structure
### test bayes filter with model uncertainty

### test training dynamics model seperately
### collect a dataset with random actions and decrease the size of the state space

def sample_gaussian(m, v, device):
    epsilon = Normal(0, 1).sample(m.size())
    return m + torch.sqrt(v) * epsilon.to(device)

def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m.unsqueeze(2), v.unsqueeze(2)

def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect ## sigma^-2

    mu = (m_vect*T_vect).sum(2) * (1/T_vect.sum(2))
    var = (1/T_vect.sum(2))

    return mu, var

#######################################
# Defining Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel


class EEFRC_Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, proprio_size, force_size, action_dim, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device
        self.image_size = image_size

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size[0]
        self.z_dim = z_dim

        self.ee_nc = ResNetFCN(folder + "_ee_resnet_nc", self.action_dim + self.proprio_size + self.force_size, self.proprio_size, 5, device = self.device)

        self.ee_c = ResNetFCN(folder + "_ee_resnet_c", self.action_dim + self.proprio_size + self.force_size, self.proprio_size, 5, device = self.device)

        self.frc_nc = ResNetFCN(folder + "_frc_resnet_nc", self.action_dim + self.proprio_size + self.force_size, self.force_size, 5, device = self.device) 

        self.frc_c = ResNetFCN(folder + "_frc_resnet_c", self.action_dim + self.proprio_size + self.force_size, self.force_size, 5, device = self.device) 

        self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + self.force_size, 1, 3, device = self.device)    

        self.model_list = [self.ee_c, self.ee_nc, self.frc_nc, self.frc_c, self.contact_class]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_ee_pred(self, proprio, force, action):

        cont = torch.sigmoid(self.contact_class(torch.cat([proprio, force], dim = 1)))

        contact = torch.where(cont > 0.5, torch.ones_like(cont), torch.zeros_like(cont))
        
        proprio_pred = self.ee_nc(torch.cat([proprio, force, action], dim = 1)) + contact * self.ee_c(torch.cat([proprio, force action], dim = 1)) + proprio

        force_pred = self.frc_nc(torch.cat([proprio, force, action], dim = 1)) + contact * self.frc_c(torch.cat([proprio, force action], dim = 1)) + force

        return proprio_pred, force_pred

    def forward(self, input_dict):
        proprios = (input_dict["proprio"]).to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        prop_list = []
        force_list = []
        contact_list = []

        if self.curriculum is not None:
            steps = images.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = images.size(1)

            force = forces[:,0]
            proprio = proprios[:,0]

        for idx in range(steps):
            action = actions[:,idx]

            cont_class = self.contact_class(torch.cat([proprio, force], dim = 1))
            prop, force = self.get_ee(proprio, force, action)

            prop_list.append(prop)
            force_list.append(force)
            contact_list.append(cont_class)

        return {
            'contact': contact_list,
            'prop_pred': prop_list,
            'force_pred': force_list,
        }

    def trans(self, input_dict):
        proprio = input_dict["proprio"].to(self.device).squeeze().unsqueeze(0)
        force = input_dict["force"].to(self.device).squeeze().unsqueeze(0)
        actions = (input_dict["action_sequence"]).to(self.device)

        steps = actions.size(0)

        for idx in range(steps):
            action = actions[idx].unsqueeze(0)
            proprio, force = self.get_ee_pred(proprio, force, action)           

        return {
            'proprio': proprio,
            'force': force,
        }
