import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
from models_utils import *

### incorporate contact into dynamics model
### incorporate force into network

### test a deterministic model without any latent space regularization



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
class Dynamics_DetModel(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, z_dim, action_dim, device = None, curriculum = None):
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
        self.z_dim = z_dim
        self.model_list = []

        self.trans_model = ResNetFCN(folder + "_trans_resnet", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device)
        self.model_list.append(self.trans_model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"]) 

    def forward(self, input_dict):
        z_list = input_dict["z"]
        actions = input_dict["action"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        z_pred = z_list[0]

        z_pred_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)        

        for idx in range(steps - 1):
            action = actions[:,idx]
            z_pred = self.trans_model(torch.cat([z_pred, action], dim = 1), z_pred)

            z_pred_list.append(z_pred)

        return {
            "z_pred": z_pred_list,
        }

    def trans(self, input_dict):
        z = input_dict["z"].to(self.device).squeeze().unsqueeze(0)
        actions = (input_dict["action_sequence"]).to(self.device)

        steps = actions.size(0)

        for idx in range(steps):
            action = actions[idx].unsqueeze(0)
            z = self.trans_model(torch.cat([z, action], dim = 1), z).squeeze().unsqueeze(0)

        return {
            'latent_state': z,
        }

class Relational_Multimodal(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
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
        self.z_dim = z_dim

        self.proprio_enc = FCN(folder + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)

        self.proprio_dec = FCN(folder + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.img_enc = CONV2DN(folder + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(folder + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.fusion_module = FCN(folder + "_fusion_unit", 4 * self.z_dim, self.z_dim, 5, device = self.device) 

        self.trans_model = ResNetFCN(folder + "_trans_resnet", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device)

        self.model_list = [self.fusion_module, self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.trans_model]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_z(self, image, proprio):
        img_emb = self.img_enc(image)
        proprio_emb = self.proprio_enc(proprio)
        return self.fusion_module(torch.cat([img_emb, proprio_emb], dim = 1))

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

    def get_proprio(self, z):
        return self.proprio_dec(z)

    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        z_list = []
        z_perm_list = []
        eepos_list = []
        img_list = []
        prop_list = []

        if self.curriculum is not None:
            steps = images.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = images.size(1)

        for idx in range(steps):
            image = images[:,idx]
            proprio = proprios[:, idx]

            z = self.get_z(image, proprio)
            img = self.get_image(z)
            prop = self.get_proprio(z)

            z_list.append(z)
            eepos_list.append(proprio[:,:3])

            img_list.append(img)
            prop_list.append(prop)


        return {
            'z': z_list,
            'z_dist': (z_list, eepos_list),
            'image_dec': img_list,
            'prop_dec': prop_list,
        }

    def encode(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)

        return {
            'latent_state': self.get_z(images[:,0], proprios[:,0]),
        }

class VAE_Multimodal(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
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
        self.z_dim = z_dim

        self.proprio_enc = FCN(folder + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.proprio_dec = FCN(folder + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.img_enc = CONV2DN(folder + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(folder + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.model_list = [self.img_enc, self.img_dec, self.proprio_enc, self.proprio_dec]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])        

    def get_z(self, image, proprio):

        img_mu, img_var = gaussian_parameters(self.img_enc(image))
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprio)) 

        mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z = sample_gaussian(mu_z.squeeze(), var_z.squeeze(), self.device).squeeze()

        return z, mu_z.squeeze(), var_z.squeeze()

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

    def get_proprio(self, z):
        return self.proprio_dec(z)

    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        params_list = []
        img_list = []
        prop_list = []
        z_list = []

        if self.curriculum is not None:
            steps = images.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = images.size(1)

        for idx in range(steps):
            image = images[:,idx]
            proprio = proprios[:, idx]

            z, z_mu, z_var = self.get_z(image, proprio)
            img = self.get_image(z)
            prop = self.get_proprio(z)

            params_list.append((z_mu, z_var, torch.zeros_like(z_mu), torch.ones_like(z_var)))
            img_list.append(img)
            prop_list.append(prop)
            z_list.append(z_mu)

        return {
            'z': z_list,
            'params': params_list,
            'image_dec': img_list,
            'prop_dec': prop_list,
        }

    def encode(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        z, z_mu, z_var = self.get_z(images[:,0], proprios[:,0])
        return {
            'latent_state': z_mu,
        }

class Selfsupervised_Multimodal(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, proprio_size, z_dim, action_dim, device = None, curriculum = None):
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
        self.z_dim = z_dim

        self.pair_clas = FCN(folder + "_pair_clas", self.z_dim, 1, 3, device = self.device)
        self.contact_clas = FCN(folder + "_contact_clas", self.z_dim + self.action_dim, 1, 3, device = self.device)        

        self.proprio_enc = FCN(folder + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.eepos_dec = FCN(folder + "_eepos_dec", self.z_dim + self.action_dim, 3, 5, device = self.device)

        self.img_enc = CONV2DN(folder + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)

        self.img_dec = DECONV2DN(folder + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.fusion_module = FCN(folder + "_action_fusion_unit", self.z_dim + self.action_dim, self.z_dim, 2, device = self.device) 

        self.model_list = [self.img_enc, self.proprio_enc, self.eepos_dec, self.contact_clas, self.pair_clas, self.fusion_module, self.img_dec]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"]) 

    def get_z(self, image, proprio):

        img_mu, img_var = gaussian_parameters(self.img_enc(image)) 
        proprio_mu, proprio_var = gaussian_parameters(self.proprio_enc(proprio)) 

        mu_vect = torch.cat([img_mu, proprio_mu], dim = 2)
        var_vect = torch.cat([img_var, proprio_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        z = sample_gaussian(mu_z.squeeze(), var_z.squeeze(), self.device).squeeze()

        return z, mu_z.squeeze(), var_z.squeeze()
 
    def get_contact(self, z, action):
        return self.contact_clas(torch.cat([z, action], dim = 1))

    def get_eepos(self, z, action):
        return self.eepos_dec(torch.cat([z, action], dim = 1))

    def get_pairing(self, z):
        return self.pair_clas(z)

    def get_image(self, z, action):
        z_act = self.fusion_module(torch.cat([z, action], dim = 1))
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z_act))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

    def forward(self, input_dict):
        actions = (input_dict["action"]).to(self.device)
        contacts = (input_dict["contact"]).to(self.device)
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        proprios_up = (input_dict["proprio_up"]).to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        params_list = []
        z_list = []
        pair_list = []
        eepos_list = []
        contact_list = []
        img_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            image = images[:,idx]
            proprio = proprios[:, idx]
            proprio_up = proprios_up[:,idx]
            action = actions[:,idx]
            contact = contacts[:,idx + 1]

            z, z_mu, z_var = self.get_z(image, proprio)
            z_up, z_mu_up, z_var_up = self.get_z(image, proprio_up)

            p_logits = self.get_pairing(z).squeeze().unsqueeze(1)
            up_logits = self.get_pairing(z_up).squeeze().unsqueeze(1)

            ones = torch.ones_like(p_logits)
            zeros = torch.zeros_like(up_logits)

            contact_logits = self.get_contact(z, action).squeeze()
            eepos_pred = self.get_eepos(z, action).squeeze()
            img = self.get_image(z, action)

            # print(img.size())

            params_list.append((z_mu, z_var, torch.zeros_like(z_mu), torch.ones_like(z_var)))

            eepos_list.append(eepos_pred)
            contact_list.append((contact_logits.squeeze(), contact.squeeze()))
            img_list.append(img)

            pair_list.append((torch.cat([p_logits, up_logits], dim = 1), torch.cat([ones, zeros], dim = 1)))
            z_list.append(z_mu)

        return {
            'z': z_list,
            'params': params_list,
            'pairing_class': pair_list,
            'eepos_pred': eepos_list,
            'contact_pred': contact_list,
            'image_pred': img_list,
        }

    def encode(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        z, z_mu, z_var = self.get_z(images[:,0], proprios[:,0])
        return {
            'latent_state': z_mu,
        }
