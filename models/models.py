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

#### observation encoder for mapping from the image data and force data to low dimensional latent space
# Add network processing to determine latent space paramenters instead of product of experts to compare performance
class Simple_Multimodal(Proto_Macromodel):
    def __init__(self, model_name, image_size, force_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
        super().__init__()

        self.curriculum = curriculum
        self.device = device
        self.image_size = image_size

        self.force_size = force_size[0]
        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim

        self.frc_enc = FCN(model_name + "_frc_enc", self.force_size, 2 * self.z_dim, 5, device = self.device)
        self.frc_dec = FCN(model_name + "_frc_dec", self.z_dim, self.force_size, 5, device = self.device)

        self.proprio_enc = FCN(model_name + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.proprio_dec = FCN(model_name + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.img_enc = CONV2DN(model_name + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(model_name + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.trans_model = FCN(model_name + "_trans_model", self.z_dim + self.action_dim, 2 * self.z_dim, 5, device = self.device)

        self.model_list = [ self.frc_enc, self.frc_dec, self.img_enc, self.img_dec, self.proprio_enc, self.proprio_dec, self.trans_model]
 
    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        forces = (input_dict["force"]).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action"]).to(self.device)
        epoch = int(input_dict["epoch"].detach().item())

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(forces[:,0])) 
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z_post = sample_gaussian(mu_z, var_z, self.device)
        z_prior = z_post.clone()

        params_list = []
        img_dec_list = []
        frc_dec_list = []
        proprio_dec_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]
            image = images[:,idx+1]
            force = forces[:, idx+1]
            proprio = proprios[:, idx+1]

            img_mu, img_var = gaussian_parameters(self.img_enc(image))
            frc_mu, frc_var = gaussian_parameters(self.frc_enc(force)) 
            proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprio)) 
            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            mu_vect = torch.cat([img_mu, frc_mu, proprio_mu], dim = 2)
            var_vect = torch.cat([img_var, frc_var, proprio_var], dim = 2)
            mu_z, var_z = product_of_experts(mu_vect, var_vect)

            z_post = sample_gaussian(mu_z, var_z, self.device)
            z_prior = sample_gaussian(trans_mu.squeeze(), trans_var.squeeze(), self.device)

            img_dec = 255.0 * torch.sigmoid(self.img_dec(z_post))
            img_dec = F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')
            frc_dec = self.frc_dec(z_post)
            proprio_dec = self.proprio_dec(z_post)

            params = (mu_z.squeeze(), var_z.squeeze(), trans_mu.squeeze(), trans_var.squeeze())

            params_list.append(params)
            img_dec_list.append(img_dec)
            frc_dec_list.append(frc_dec)
            proprio_dec_list.append(proprio_dec)


        return {
            'params': params_list,
            'image_pred': img_dec_list,
            'force_pred': frc_dec_list,
            'proprio_pred': proprio_dec_list,
        }

    def encode(self, input_dict):

        images = (input_dict["image"] / 255.0).to(self.device)
        forces = (input_dict["force"]).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(forces[:,0])) 
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        return {
            'latent_state': mu_z,
        }

    def trans(self, input_dict):
        image = (input_dict["image"] / 255.0).to(self.device)
        force = (input_dict["force"]).to(self.device)
        proprio = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action_sequence"]).to(self.device)

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(forces[:,0])) 
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z_prior = mu_z

        steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]

            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            z_prior = rans_mu.squeeze().unsqueeze(0)

        return {
            'latent_state': z_prior,
        }

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

class Simple_Multimodal_woutforce(Proto_Macromodel):
    def __init__(self, model_name, image_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
        super().__init__()

        self.curriculum = curriculum
        self.device = device
        self.image_size = image_size

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim

        self.proprio_enc = FCN(model_name + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.proprio_dec = FCN(model_name + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.img_enc = CONV2DN(model_name + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(model_name + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.trans_model = FCN(model_name + "_trans_model", self.z_dim + self.action_dim, 2 * self.z_dim, 5, device = self.device)

        self.model_list = [self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.trans_model]
 
    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z_post = sample_gaussian(mu_z, var_z, self.device)
        z_prior = z_post.clone()

        params_list = []
        img_dec_list = []
        proprio_dec_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]
            image = images[:,idx+1]
            proprio = proprios[:, idx+1]

            img_mu, img_var = gaussian_parameters(self.img_enc(image))
            proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprio)) 
            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
            var_vect = torch.cat([img_var, proprio_var], dim = 2)
            mu_z, var_z = product_of_experts(mu_vect, var_vect)

            z_post = sample_gaussian(mu_z, var_z, self.device)
            z_prior = sample_gaussian(trans_mu.squeeze(), trans_var.squeeze(), self.device)

            img_dec = 255.0 * torch.sigmoid(self.img_dec(z_post))
            img_dec = F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')
            proprio_dec = self.proprio_dec(z_post)

            params = (mu_z.squeeze(), var_z.squeeze(), trans_mu.squeeze(), trans_var.squeeze())

            params_list.append(params)
            img_dec_list.append(img_dec)
            proprio_dec_list.append(proprio_dec)


        return {
            'params': params_list,
            'image_pred': img_dec_list,
            'proprio_pred': proprio_dec_list,
        }

    def encode(self, input_dict):

        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        return {
            'latent_state': mu_z,
        }

    def trans(self, input_dict):
        image = (input_dict["image"] / 255.0).to(self.device)
        proprio = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action_sequence"]).to(self.device)

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprios[:,0])) 

        mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z_prior = mu_z

        steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]

            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            z_prior = rans_mu.squeeze().unsqueeze(0)

        return {
            'latent_state': z_prior,
        }

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

class Simple_Multimodal_woutforcewfusion(Proto_Macromodel):
    def __init__(self, model_name, image_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
        super().__init__()

        self.curriculum = curriculum
        self.device = device
        self.image_size = image_size

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim

        self.proprio_enc = FCN(model_name + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.proprio_dec = FCN(model_name + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.img_enc = CONV2DN(model_name + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(model_name + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.fusion_module = FCN(model_name + "_fusion_unit", 4 * self.z_dim, 2 * self.z_dim, 5, device = self.device) 

        self.trans_model = FCN(model_name + "_trans_model", self.z_dim + self.action_dim, 2 * self.z_dim, 5, device = self.device)

        self.model_list = [self.fusion_module, self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.trans_model]
 
    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        img_emb = self.img_enc(images[:,0])
        proprio_emb = self.proprio_enc(proprios[:,0])

        mu_z, var_z = gaussian_parameters(self.fusion_module(torch.cat([img_emb.squeeze(), proprio_emb.squeeze()], dim = 1)))

        z_post = sample_gaussian(mu_z, var_z, self.device).squeeze()
        z_prior = z_post.clone().squeeze()

        # print("Z posterior size: ", z_post.size())
        # print("Z prior size: ", z_prior.size())

        params_list = []
        img_dec_list = []
        proprio_dec_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]
            image = images[:,idx+1]
            proprio = proprios[:, idx+1]

            img_emb = self.img_enc(image)
            proprio_emb = self.proprio_enc(proprio)

            mu_z, var_z = gaussian_parameters(self.fusion_module(torch.cat([img_emb.squeeze(), proprio_emb.squeeze()], dim = 1)))
            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            z_post = sample_gaussian(mu_z, var_z, self.device).squeeze()
            z_prior = sample_gaussian(trans_mu.squeeze(), trans_var.squeeze(), self.device).squeeze()

            # print("Z posterior size: ", z_post.size())
            # print("Z prior size: ", z_prior.size())

            img_dec = 255.0 * torch.sigmoid(self.img_dec(z_post))
            img_dec = F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')
            proprio_dec = self.proprio_dec(z_post)

            params = (mu_z.squeeze(), var_z.squeeze(), trans_mu.squeeze(), trans_var.squeeze())

            params_list.append(params)
            img_dec_list.append(img_dec)
            proprio_dec_list.append(proprio_dec)


        return {
            'params': params_list,
            'image_pred': img_dec_list,
            'proprio_pred': proprio_dec_list,
        }

    def encode(self, input_dict):

        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)

        img_emb = self.img_enc(images[:,0])
        proprio_emb = self.proprio_enc(proprios[:,0])

        mu_z, var_z = gaussian_parameters(self.fusion_module(torch.cat([img_emb.squeeze(), proprio_emb.squeeze()], dim = 1)))

        return {
            'latent_state': mu_z,
        }

    def trans(self, input_dict):
        image = (input_dict["image"] / 255.0).to(self.device)
        proprio = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action_sequence"]).to(self.device)

        img_emb = self.img_enc(images[:,0])
        proprio_emb = self.proprio_enc(proprios[:,0])

        mu_z, var_z = gaussian_parameters(self.fusion_module(torch.cat([img_emb.squeeze(), proprio_emb.squeeze()], dim = 1)))

        z_prior = mu_z

        steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]

            trans_mu, trans_var = gaussian_parameters(self.trans_model(torch.cat([z_prior, action], dim = 1)))

            z_prior = rans_mu.squeeze().unsqueeze(0)

        return {
            'latent_state': z_prior,
        }

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

class EEpos_dynamics(Proto_Macromodel):
    def __init__(self, model_name, proprio_size, action_dim, device = None, curriculum = None):
        super().__init__()

        self.curriculum = curriculum
        self.device = device

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]

        self.proprio_dyn = FCN(model_name + "_proprio_enc", self.proprio_size + self.action_dim, 2 * self.proprio_size, 5, device = self.device)

        self.model_list = [self.proprio_dyn]
 
    def forward(self, input_dict):
        proprio = (input_dict["proprio"]).to(self.device)[:, 0]
        actions = (input_dict["action"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        proprio_dec_list = []

        # print("Proprio size: ", proprio.size())

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]

            proprio_mu, proprio_var = gaussian_parameters(self.proprio_dyn(torch.cat([proprio.squeeze(), action.squeeze()], dim = 1)))

            proprio = sample_gaussian(proprio_mu, proprio_var, self.device).squeeze()

            proprio_dec_list.append(proprio)
        # print("Proprio size: ", proprio.size())

        return {
            'proprio_pred': proprio_dec_list,
        }
    def trans(self, input_dict):
        proprio_pred = (input_dict["proprio"]).to(self.device)
        actions = (input_dict["action"]).to(self.device)
        epoch = input_dict["epoch"]

        proprio_dec_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]

            proprio_mu, proprio_var = gaussian_parameters(self.proprio_dyn(torch.cat([proprio_pred.squeeze(), action.squeeze()], dim = 1)))

            proprio_pred = sample_gaussian(proprio_mu, proprio_var, self.device)

        return {
            'latent_state': proprio_pred,
        }
