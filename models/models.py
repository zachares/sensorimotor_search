import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
from models_utils import *


def sample_gaussian(m, v, device):
    
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z

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

#### observation encoder for mapping from the image data and force data to low dimensional latent space
# Add network processing to determine latent space paramenters instead of product of experts to compare performance
class PlaNet_Multimodal(Proto_Macromodel):
    def __init__(self, model_name, image_size, force_size, z_dim, action_dim, device = None):
        super().__init__()

        self.device = device
        self.image_size = image_size

        self.force_size = force_size[0]
        self.action_dim = action_dim[0]
        self.z_dim = z_dim

        self.frc_enc = FCN(model_name + "_frc_enc", self.force_size, 2 * z_dim, 4, device = self.device)
        self.frc_dec = FCN(model_name + "_frc_dec", z_dim, self.force_size, 4, device = self.device)

        self.img_enc = CONV2DN(model_name + "_img_enc", image_size, (2 * z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(model_name + "_img_dec", (z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.hidden_enc = FCN(model_name + "_hidden_enc", z_dim, 2 * z_dim, 3, device = self.device)

        self.det_state_model = LSTM(model_name + "_det_state_model", z_dim + self.action_dim, z_dim, device = self.device)

        self.trans_model = FCN(model_name + "_trans_model", z_dim, 2 * z_dim, 3, device = self.device)

        self.model_list = [ self.frc_enc, self.frc_dec, self.img_enc, self.img_dec, self.hidden_enc, self.det_state_model, self.trans_model]
 
    def forward(self, input_dict):
        image = input_dict["image"] / 255.0
        force = input_dict["force"]
        image_fut = input_dict["image_fut"] / 255.0
        force_fut = input_dict["force_fut"]
        prev_state = input_dict["z"]
        prev_rnn_state = input_dict["rnn_state"]
        action = input_dict["action"]

        if prev_rnn_state == None:
            prev_hidden = None
            prev_cell = None
        else:
            prev_hidden, prev_cell = prev_rnn_state

        if prev_state == None:
            img_mu, img_var = gaussian_parameters(self.img_enc(image))
            frc_mu, frc_var = gaussian_parameters(self.frc_enc(force)) 

            mu_vect = torch.cat([img_mu, frc_mu], dim = 2)
            var_vect = torch.cat([img_var, frc_var], dim = 2)
            mu_z, var_z = product_of_experts(mu_vect, var_vect)

            prev_state = sample_gaussian(mu_z, var_z, self.device)

        belief_state, cell_state = self.det_state_model(torch.cat([prev_state, action], dim = 1), prev_hidden, prev_cell)

        hid_mu, hid_var = gaussian_parameters(self.hidden_enc(belief_state))
        img_mu, img_var = gaussian_parameters(self.img_enc(image_fut))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(force_fut)) 
        trans_mu, trans_var = gaussian_parameters(self.trans_model(belief_state))

        mu_vect = torch.cat([hid_mu, img_mu, frc_mu, trans_mu], dim = 2)
        var_vect = torch.cat([hid_var, img_var, frc_var, trans_var], dim = 2)

        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z_enc = sample_gaussian(mu_z, var_z, self.device)
        z_state = sample_gaussian(trans_mu, trans_var, self.device)

        img_dec = 255.0 * torch.sigmoid(self.img_dec(z_enc))
        img_dec = F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')
        frc_dec = self.frc_dec(z_enc)

        params = (mu_z, var_z, trans_mu.squeeze(), trans_var.squeeze())

        rnn_state = (belief_state, cell_state)

        return {
            'params': params,
            'rnn_state': rnn_state,
            'image_pred': img_dec,
            'force_pred': frc_dec,
            'z': z_enc,
        }

    def trans(self, inputs):

        image = input_dict["image"]
        force = input_dict["force"]
        prev_state = input_dict["z"]
        prev_rnn_state = input_dict["rnn_state"]
        action = input_dict["action"]

        if prev_rnn_state == None:
            prev_hidden = None
            prev_cell = None
        else:
            prev_hidden, prev_cell = prev_rnn_state

        if prev_state == None:
            img_mu, img_var = gaussian_parameters(self.img_enc(image))
            frc_mu, frc_var = gaussian_parameters(self.frc_enc(force)) 

            mu_vect = torch.cat([img_mu, frc_mu], dim = 2)
            var_vect = torch.cat([img_var, frc_var], dim = 2)
            mu_z, var_z = product_of_experts(m_vect, var_vect)

            prev_state = sample_gaussian(mu_z, var_z, self.device)

        belif_state, cell_state = self.det_state_model(torch.cat([prev_state, action], dim = 1), prev_hidden, prev_cell)

        rnn_state = (belief_state, cell_state)
        
        trans_mu, trans_var = gaussian_parameters(self.trans_model(belief_state))        

        return {
            'next_state': sample_gaussian(trans_mu, trans_var, self.device),
            'rnn_state': rnn_state,
        }