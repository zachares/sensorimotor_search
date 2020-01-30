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

def sample_gaussian(m, h, device):
    v = F.softplus(h) + 1e-8
    epsilon = Normal(0, 1).sample(m.size())
    return m + torch.sqrt(v) * epsilon.to(device)

def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect ## sigma^-2

    mu = (m_vect*T_vect).sum(2) * (1/T_vect.sum(2))
    var = (1/T_vect.sum(2))

    return mu, var

def simple_normalization(vector):
    return vector / vector.norm(p=2, dim = 1).unsqueeze(1).repeat(1, vector.size(1))
#######################################
# Defining Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel
class DynamicswForce(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, z_dim, num_options, offset, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim
        self.force_size = force_size
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16

        self.normalization = simple_normalization #nn.Softmax(dim=1)

        self.state_size = (self.frc_enc_size + self.proprio_size + self.contact_size) 

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + 2 * self.num_options, 1, 3, device = self.device)    

        self.idyn = ResNetFCN(folder + "_inverse_dynamics", 2 * self.state_size + 2 * self.num_options, self.z_dim, 4, device = self.device)
        self.direction_model = FCN(folder + "_direction_est", self.z_dim, self.action_dim, 3, device = self.device)
        self.magnitude_model = FCN(folder + "_magnitude_est", self.z_dim, 1, 3, device = self.device)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.state_size - self.contact_size, 5, device = self.device)

        self.frc_rep = ResNetFCN(folder + "_frc_rep", self.proprio_size + self.contact_size, self.frc_enc_size, 4, device = self.device)
        self.dyn_noise = Params(folder + "_dynamics_noise", (self.state_size - self.contact_size), device = self.device) 
        self.idyn_noise = Params(folder + "_inv_dynamics_noise", (self.action_dim), device = self.device) 
        self.frc_rep_noise = Params(folder + "_inv_dynamics_noise", (self.frc_enc_size), device = self.device) 

        self.model_list.append(self.idyn)
        self.model_list.append(self.dyn)
        self.model_list.append(self.direction_model)
        self.model_list.append(self.magnitude_model)

        self.model_list.append(self.frc_enc)
        self.model_list.append(self.contact_class)

        self.model_list.append(self.frc_rep)
        self.model_list.append(self.dyn_noise)
        self.model_list.append(self.idyn_noise)
        self.model_list.append(self.frc_rep_noise)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_action_est(self, state, next_state, peg_type, hole_type):
        # print("State size: ", state.size(), "   ", self.state_size)
        action_latent = self.idyn(torch.cat([state, next_state, peg_type, hole_type], dim = 1))
        return self.normalization(self.direction_model(action_latent)), torch.abs(self.magnitude_model(action_latent))

    def get_state_pred(self, state, action, peg_type, hole_type):
        next_state = self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))
        return next_state[:, :self.proprio_size], next_state[:, self.proprio_size:]

    def forward(self, input_dict):
        actions = input_dict["action"].to(self.device)
        proprios = input_dict["proprio"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        action_mag_list = []
        action_dir_list = []
        frc_enc_list = []
        contact_list = []
        proprio_pred_list = []
        frc_enc_pred_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            proprio = proprios[:,idx]
            proprio_next = proprios[:,idx+1]

            force = forces[:,idx]
            force_next = forces[:,idx+1]

            peg_type = peg_types[:,idx]
            hole_type = hole_types[:,idx]

            contact = contacts[:,idx]
            contact_next = contacts[:,idx+1]

            action = actions[:,idx]

            contact_logits = self.contact_class(torch.cat([proprio, peg_type, hole_type], dim = 1))
            
            frc_enc = self.frc_enc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))

            frc_enc_next = self.frc_enc(force_next)

            if idx == steps - 1:
                frc_enc_list.append(frc_enc_next.unsqueeze(1))

            state = torch.cat([proprio, frc_enc, contact], dim =1)
            state_next = torch.cat([proprio_next, frc_enc_next, contact_next], dim = 1)

            action_dir_probs, action_mag = self.get_action_est(state, state_next, peg_type, hole_type)

            proprio_pred, frc_enc_pred = self.get_state_pred(state, action, peg_type, hole_type)

            if idx >= self.offset:
                action_mag_list.append(action_mag.squeeze())
                action_dir_list.append(action_dir_probs)
                contact_list.append(contact_logits.squeeze())
                proprio_pred_list.append(proprio_pred)
                frc_enc_pred_list.append(frc_enc_pred)

        return {
            "action_mag": action_mag_list,
            "action_dir": action_dir_list,
            "frc_enc": torch.cat(frc_enc_list, dim = 1),
            "contact_class": contact_list,
            "frc_enc_pred": frc_enc_pred_list,
            "proprio_pred": proprio_pred_list,
        }

class Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, proprio_size, action_dim, z_dim, num_options, offset, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1

        self.normalization = simple_normalization #nn.Softmax(dim=1)

        self.state_size = (self.proprio_size + self.contact_size) 

        self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + 2 * self.num_options, 1, 3, device = self.device)    

        self.idyn = ResNetFCN(folder + "_inverse_dynamics", 2 * self.state_size + 2 * self.num_options, self.z_dim, 4, device = self.device)
        self.direction_model = FCN(folder + "_direction_est", self.z_dim, self.action_dim, 3, device = self.device)
        self.magnitude_model = FCN(folder + "_magnitude_est", self.z_dim, 1, 3, device = self.device)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.proprio_size, 4, device = self.device)

        self.dyn_noise = Params(folder + "_dynamics_noise", (self.proprio_size), device = self.device) 
        self.idyn_noise = Params(folder + "_inv_dynamics_noise", (self.action_dim), device = self.device) 

        self.model_list.append(self.idyn)
        self.model_list.append(self.dyn)
        self.model_list.append(self.direction_model)
        self.model_list.append(self.magnitude_model)
        self.model_list.append(self.contact_class)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_action_est(self, state, next_state, peg_type, hole_type):
        action_latent = self.idyn(torch.cat([state, next_state, peg_type, hole_type], dim = 1))
        return self.normalization(self.direction_model(action_latent)), torch.abs(self.magnitude_model(action_latent))

    def get_state_pred(self, state, action, peg_type, hole_type):
        return self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))

    def forward(self, input_dict):
        actions = input_dict["action"].to(self.device)
        proprios = input_dict["proprio"].to(self.device)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        action_mag_list = []
        action_dir_list = []
        proprio_list = []
        contact_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            proprio = proprios[:,idx]
            proprio_next = proprios[:,idx+1]

            peg_type = peg_types[:,idx]
            hole_type = hole_types[:,idx]

            contact = contacts[:,idx]
            contact_next = contacts[:,idx+1]

            action = actions[:,idx]

            contact_logits = self.contact_class(torch.cat([proprio, peg_type, hole_type], dim = 1))

            state = torch.cat([proprio, contact], dim =1)
            state_next = torch.cat([proprio_next, contact_next], dim = 1)

            action_dir, action_mag = self.get_action_est(state, state_next, peg_type, hole_type)
            proprio_pred = self.get_state_pred(state, action, peg_type, hole_type)

            if idx >= self.offset:
                action_mag_list.append(action_mag.squeeze())
                action_dir_list.append(action_dir)
                contact_list.append(contact_logits.squeeze())
                proprio_list.append(proprio_pred)

        return {
            "action_mag": action_mag_list,
            "action_dir": action_dir_list,
            "contact_class": contact_list,
            "proprio_pred": proprio_list,
        }

class Options_ClassifierLSTM(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, z_dim, num_options, offset, learn_rep = True, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size
        self.z_dim = z_dim
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16
        self.learn_rep = learn_rep
        
        self.softmax = nn.Softmax(dim=1)

        if self.learn_rep:  
            self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + 3 * self.num_options

            self.cross_params = Params(folder + "_cross_rep", (2 * self.num_options), device = self.device)
            self.rect_params = Params(folder + "_rect_rep", (2 * self.num_options), device = self.device)
            self.square_params = Params(folder + "_square_rep", (2 * self.num_options), device = self.device)

            self.model_list.append(self.cross_params)
            self.model_list.append(self.rect_params)
            self.model_list.append(self.square_params)
        else:
            self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + 2 * self.num_options

        self.options_lstm = LSTMCell(folder + "_options_lstm", self.state_size, self.z_dim, device = self.device)
        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)
        self.options_class = FCN(folder + "_options_class", z_dim, 3, 3, device = self.device)

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.model_list.append(self.options_lstm)
        self.model_list.append(self.pre_lstm)
        self.model_list.append(self.options_class)

        self.model_list.append(self.frc_enc)


        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pegtype(self, peg_type_idx):

        if self.learn_rep:
            cross_idx = peg_type_idx[:,0].unsqueeze(1).repeat(1, self.num_options * 2)
            rect_idx = peg_type_idx[:,1].unsqueeze(1).repeat(1, self.num_options * 2)
            square_idx = peg_type_idx[:,2].unsqueeze(1).repeat(1, self.num_options * 2)

            cross_rep = self.cross_params.params.unsqueeze(0).repeat(peg_type_idx.size(0), 1)
            rect_rep = self.rect_params.params.unsqueeze(0).repeat(peg_type_idx.size(0), 1)
            square_rep = self.square_params.params.unsqueeze(0).repeat(peg_type_idx.size(0), 1)

            return cross_idx * cross_rep + rect_idx * rect_rep + square_idx * square_rep
        else:
            return peg_type_idx

    def get_options_class(self, proprio, frc_enc, contact, action, peg_type, hole_type, h = None, c = None):
        prestate = torch.cat([proprio, frc_enc, contact.unsqueeze(1), action, peg_type, hole_type], dim = 1)
        state = self.pre_lstm(prestate)

        if h is None or c is None:
            h_pred, c_pred = self.options_lstm(state)
        else:
            h_pred, c_pred = self.options_lstm(state, h, c) 

        options_logits = self.options_class(h_pred)

        return options_logits, h_pred, c_pred

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)

        # print("Forces size: ", forces.size())
        contacts = input_dict["contact"].to(self.device)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        options_list = []
        hole_accuracy_list = []
        hole_probs_list = []
        frc_enc_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        hole_probs = torch.ones_like(peg_types[:,0]) / peg_types[:,0].size(0)

        for idx in range(steps):
            action = actions[:,idx]
            proprio = proprios[:,idx]
            force = forces[:,idx]
            contact = contacts[:,idx]
            peg_type_idx = peg_types[:,idx]
            hole_type = hole_types[:,idx]

            peg_type = self.get_pegtype(peg_type_idx)

            frc_enc = self.frc_enc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))

            if idx == 0:
                options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs)
                hole_probs = self.softmax(options_logits)
            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 8 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs, h_clone, c_clone)
                hole_probs = self.softmax(options_logits)
            else:
                options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs, h, c)
                hole_probs = self.softmax(options_logits)

            samples = torch.zeros_like(hole_type)
            samples[torch.arange(samples.size(0)), hole_probs.max(1)[1]] = 1.0
            test = torch.where(samples == hole_type, torch.zeros_like(hole_probs), torch.ones_like(hole_probs)).sum(1)
            accuracy = torch.where(test > 0, torch.zeros_like(test), torch.ones_like(test))

            hole_accuracy_list.append(accuracy.squeeze().unsqueeze(1))
            hole_probs_list.append(hole_probs.unsqueeze(1))

            if idx >= self.offset:
                options_list.append(options_logits)

        return {
            'hole_accuracy': torch.cat(hole_accuracy_list, dim = 1),
            'hole_probs': torch.cat(hole_probs_list, dim = 1),
            'frc_enc': torch.cat(frc_enc_list, dim = 1),
            'options_class': options_list,
        }


