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

def log_normal(x, m, v):
    return -0.5 * ((x - m).pow(2)/ v + torch.log(2 * np.pi * v)).sum(-1).unsqueeze(-1)

#######################################
# Defining Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel
class Fit_ClassifierLSTM(Proto_Macromodel):
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

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size
        self.z_dim = z_dim
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16

        self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + self.num_options

        self.fit_lstm = LSTMCell(folder + "_fit_lstm", self.state_size, self.z_dim, device = self.device) 

        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)

        self.fit_class = FCN(folder + "_fit_class", z_dim, 1, 3, device = self.device)  

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.model_list = [self.fit_lstm, self.fit_class, self.frc_enc, self.pre_lstm]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_fit_class(self, proprio, force, contact, action, peg_type, h = None, c = None):
        # print("Force size: ", force.size())
        frc_enc = self.frc_enc(force)

        # print("Proprio size: ", proprio.size())
        # print("Force size: ", frc_enc.size())
        # print("contact size: ", contact.size())
        # print("action size: ", action.size())
        # print("peg type size: ", peg_type.size())
        prestate = torch.cat([proprio, frc_enc, contact.unsqueeze(1), action, peg_type], dim = 1)
        state = self.pre_lstm(prestate)

        if h is None or c is None:
            h_pred, c_pred = self.fit_lstm(state)
        else:
            h_pred, c_pred = self.fit_lstm(state, h, c) 

        fit_logits = self.fit_class(h_pred)

        return fit_logits, h_pred, c_pred

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)

        # print("Forces size: ", forces.size())
        contacts = input_dict["contact"].to(self.device)
        peg_types = input_dict["peg_type"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        fit_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]
            proprio = proprios[:,idx]
            force = forces[:,idx]
            contact = contacts[:,idx]
            peg_type = peg_types[:,idx]

            if idx == 0:
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type)

            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 8 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type, h_clone, c_clone)
            else:
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type, h, c)

            if idx >= self.offset:
                fit_list.append(fit_logits)

        return {
            'fit_class': fit_list,
        }

class Fit_ClassifierParticle(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, offset, num_particles = 20, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size
        self.num_particles = num_particles
        self.num_options = num_options
        self.frc_enc_size = 16
        self.offset = offset

        self.state_size_nc = self.action_dim + 2 * self.proprio_size + self.num_options
        self.state_size_c = self.action_dim + 2 * self.proprio_size + self.frc_enc_size + self.num_options

        self.ee_nc = ResNetFCN(folder + "_ee_resnet_nc", self.state_size_nc, self.proprio_size, 3, device = self.device)
        self.ee_c = ResNetFCN(folder + "_ee_resnet_c", self.state_size_c, self.proprio_size, 3, device = self.device)

        # self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + self.force_size, 1, 2, device = self.device)  

        self.proprio_noise = Params(folder + "_noise", self.proprio_size, device = self.device)  

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size , 1), False, True, 1, device = self.device)

        self.fit_lstm = LSTMCell(folder + "_fit_lstm", 16, 16, device = self.device) 

        self.pre_lstm = FCN(folder + "_pre_lstm", 4, 16, 3, device = self.device)

        self.fit_class = FCN(folder + "_fit_class", 16, 1, 3, device = self.device)  

        self.model_list = [self.ee_c, self.ee_nc, self.proprio_noise, self.frc_enc, self.fit_lstm, self.pre_lstm, self.fit_class] #self.frc_nc, self.frc_c, self.joint_nc, self.joint_c, ]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, proprio, force, action, contact, peg_type):
        frc_enc = self.frc_enc(force)
        
        nc_state = torch.cat([proprio, action, peg_type], dim = 1).unsqueeze(1).repeat(1, self.num_particles, 1)
        c_state = torch.cat([proprio, frc_enc, action, peg_type], dim = 1).unsqueeze(1).repeat(1, self.num_particles, 1)

        proprio_variance = self.proprio_noise.params.unsqueeze(0).unsqueeze(1).repeat(nc_state.size(0), nc_state.size(1), 1)
        proprio_mean = torch.zeros_like(proprio_variance)

        noise_samples = sample_gaussian(proprio_mean, proprio_variance, self.device)

        nc_state = torch.cat([nc_state, noise_samples], dim = 2)
        c_state = torch.cat([c_state, noise_samples], dim = 2)

        proprio_particles = self.ee_nc(nc_state) + contact.unsqueeze(1).unsqueeze(2).repeat(1, c_state.size(1), proprio.size(1)) * self.ee_c(c_state) #+ proprio

        proprio_mean = proprio_particles.mean(1).squeeze()
        proprio_var = proprio_particles.var(1).squeeze()

        return proprio_mean, proprio_var

    def get_fit_class(self, log_prob, peg_type, h = None, c = None):
        prestate = torch.cat([log_prob, peg_type], dim = 1)
        state = self.pre_lstm(prestate)

        if h is None or c is None:
            h_pred, c_pred = self.fit_lstm(state)
        else:
            h_pred, c_pred = self.fit_lstm(state, h, c) 

        fit_logits = self.fit_class(h_pred)

        return fit_logits, h_pred, c_pred

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        peg_types = input_dict["peg_type"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        fit_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            force = forces[:, idx]
            proprio = proprios[:,idx]
            proprio_next = proprios[:,idx + 1]
            action = actions[:,idx]
            contact = contacts[:,idx]
            peg_type = peg_types[:,idx]

            proprio_mean, proprio_var = self.get_pred(proprio, force, action, contact, peg_type)

            log_prob = log_normal(proprio_next, proprio_mean, proprio_var)

            # print("Log prob size: ", log_prob.size())
            # print("peg type size: ", peg_type.size())

            if idx == 0:
                fit_logits, h, c = self.get_fit_class(log_prob, peg_type)

            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 4 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                fit_logits, h, c = self.get_fit_class(log_prob, peg_type, h_clone, c_clone)
            else:
                fit_logits, h, c = self.get_fit_class(log_prob, peg_type, h, c)

            if idx >= self.offset:
                fit_list.append(fit_logits)

        return {
            'fit_class': fit_list,
        }

class Options_ClassifierLSTM(Proto_Macromodel):
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

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size
        self.z_dim = z_dim
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16
        
        self.softmax = nn.Softmax(dim=1)

        self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + 2 * self.num_options

        self.options_lstm = LSTMCell(folder + "_options_lstm", self.state_size, self.z_dim, device = self.device) 

        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)

        self.options_class = FCN(folder + "_options_class", z_dim, 3, 3, device = self.device)  

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.model_list = [self.options_lstm, self.options_class, self.frc_enc, self.pre_lstm]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_options_class(self, proprio, force, contact, action, peg_type, hole_type, h = None, c = None):
        # print("Force size: ", force.size())
        frc_enc = self.frc_enc(force)

        # print("Proprio size: ", proprio.size())
        # print("Force size: ", frc_enc.size())
        # print("contact size: ", contact.size())
        # print("action size: ", action.size())
        # print("peg type size: ", peg_type.size())
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
        epoch =  int(input_dict["epoch"].detach().item())

        options_list = []

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
            peg_type = peg_types[:,idx]

            if idx == 0:
                options_logits, h, c = self.get_options_class(proprio, force, contact, action, peg_type, hole_probs)
                hole_probs = self.softmax(options_logits)
            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 8 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                options_logits, h, c = self.get_options_class(proprio, force, contact, action, peg_type, hole_probs, h_clone, c_clone)
                hole_probs = self.softmax(options_logits)
            else:
                options_logits, h, c = self.get_options_class(proprio, force, contact, action, peg_type, hole_probs, h, c)
                hole_probs = self.softmax(options_logits)

            if idx >= self.offset:
                options_list.append(options_logits)

        return {
            'options_class': options_list,
        }

class Options_ClassifierParticle(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, offset, num_particles = 20, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = False
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.force_size = force_size
        self.num_particles = num_particles
        self.num_options = num_options
        self.frc_enc_size = 16
        self.offset = offset

        self.state_size_nc = self.action_dim + 2 * self.proprio_size + 2 * self.num_options
        self.state_size_c = self.action_dim + 2 * self.proprio_size + self.frc_enc_size + 2 * self.num_options

        self.ee_nc = ResNetFCN(folder + "_ee_resnet_nc", self.state_size_nc, self.proprio_size, 3, device = self.device)
        self.ee_c = ResNetFCN(folder + "_ee_resnet_c", self.state_size_c, self.proprio_size, 3, device = self.device)

        # self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + self.force_size, 1, 2, device = self.device)  

        self.proprio_noise = Params(folder + "_noise", (self.proprio_size), device = self.device)  

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.options_lstm = LSTMCell(folder + "_fit_lstm", 16, 16, device = self.device) 

        self.pre_lstm = FCN(folder + "_pre_lstm", 7, 16, 3, device = self.device)

        self.options_class = FCN(folder + "_fit_class", 16, 3, 3, device = self.device)  

        self.softmax = nn.Softmax(dim=1)

        self.model_list = [self.ee_c, self.ee_nc, self.proprio_noise, self.frc_enc, self.options_lstm, self.pre_lstm, self.options_class] #self.frc_nc, self.frc_c, self.joint_nc, self.joint_c, ]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, proprio, force, action, contact, peg_type, hole_type):
        frc_enc = self.frc_enc(force)
        
        nc_state = torch.cat([proprio, action, peg_type, hole_type], dim = 1).unsqueeze(1).repeat(1, self.num_particles, 1)
        c_state = torch.cat([proprio,frc_enc, action, peg_type, hole_type], dim = 1).unsqueeze(1).repeat(1, self.num_particles, 1)

        proprio_variance = self.proprio_noise.params.unsqueeze(0).unsqueeze(1).repeat(nc_state.size(0), nc_state.size(1), 1)
        proprio_mean = torch.zeros_like(proprio_variance)

        noise_samples = sample_gaussian(proprio_mean, proprio_variance, self.device)

        nc_state = torch.cat([nc_state, noise_samples], dim = 2)
        c_state = torch.cat([c_state, noise_samples], dim = 2)

        proprio_particles = self.ee_nc(nc_state) + contact.unsqueeze(1).unsqueeze(2).repeat(1, c_state.size(1), proprio.size(1)) * self.ee_c(c_state) #+ proprio

        proprio_mean = proprio_particles.mean(1).squeeze()
        proprio_var = proprio_particles.var(1).squeeze()

        return proprio_mean, proprio_var

    def get_options_class(self, log_prob, peg_type, hole_type, h = None, c = None):
        prestate = torch.cat([log_prob, peg_type, hole_type], dim = 1)
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
        peg_types = input_dict["peg_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        options_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        hole_probs = torch.ones_like(peg_types[:,0]) / peg_types[:,0].size(0)

        for idx in range(steps):
            force = forces[:, idx]
            proprio = proprios[:,idx]
            proprio_next = proprios[:,idx + 1]
            action = actions[:,idx]
            contact = contacts[:,idx]
            peg_type = peg_types[:,idx]

            proprio_mean, proprio_var = self.get_pred(proprio, force, action, contact, peg_type, hole_probs)

            log_prob = log_normal(proprio_next, proprio_mean, proprio_var)

            if idx == 0:
                options_logits, h, c = self.get_options_class(log_prob, peg_type, hole_probs)
                hole_probs = self.softmax(options_logits)
            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 4 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                options_logits, h, c = self.get_options_class(log_prob, peg_type, hole_probs, h_clone, c_clone)
                hole_probs = self.softmax(options_logits)
            else:
                options_logits, h, c = self.get_options_class(log_prob, peg_type, hole_probs, h, c)
                hole_probs = self.softmax(options_logits)

            if idx >= self.offset:
                options_list.append(options_logits)

        return {
            'options_class': options_list,
        }


    # def trans(self, input_dict):
    #     proprio = input_dict["proprio"].to(self.device)
    #     force = input_dict["force"].to(self.device)
    #     joint_pos = input_dict["joint_pos"].to(self.device).squeeze()
    #     joint_vel = input_dict["joint_vel"].to(self.device).squeeze()
    #     joint_proprio = torch.cat([joint_pos, joint_vel], dim = 0)
    #     actions = (input_dict["action_sequence"]).to(self.device)

    #     steps = actions.size(0)

    #     for idx in range(steps):
    #         action = actions[idx].unsqueeze(0)
    #         proprio, joint_proprio, force = self.get_pred(proprio.squeeze().unsqueeze(0),\
    #         joint_proprio.squeeze().unsqueeze(0), force.squeeze().unsqueeze(0), action)           

    #     return {
    #         'proprio': proprio.squeeze().unsqueeze(0),
    #         'joint_pos': joint_proprio.squeeze().unsqueeze(0)[:, :self.joint_size],
    #         'joint_vel': joint_proprio.squeeze().unsqueeze(0)[:, self.joint_size:],
    #         'force': force.squeeze().unsqueeze(0),
    #     }
