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
class Simple_Multimodal(Proto_Macromodel):
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

    def get_z_pred(self, z, action):
        return self.trans_model(torch.cat([z, action], dim = 1), z)

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

    def get_proprio(self, z):
        return self.proprio_dec(z)

    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        actions = input_dict["action"].to(self.device)
        contacts = input_dict["contact"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        z_list = []
        z_perm_list = []
        eepos_list = []
        z_pred_list = []
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
            action = actions[:,idx]
            contact = contacts[:,idx]

            z = self.get_z(image, proprio)
            z_list.append(z)

            if idx == 0:
                z_pred = z.clone()

            img = self.get_image(z)
            img_list.append(img)
            prop = self.get_proprio(z)
            prop_list.append(prop)

            if idx != steps - 1:
                z_pred = self.get_z_pred(z_pred, action)
                z_pred_list.append(z_pred)

            eepos_list.append(proprio[:,:3])

        return {
            'z': z_list,
            'z_pred': z_pred_list,
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

    def trans(self, input_dict):
        z = input_dict["z"].to(self.device).squeeze().unsqueeze(0)
        actions = (input_dict["action_sequence"]).to(self.device)

        steps = actions.size(0)

        for idx in range(steps):
            action = actions[idx].unsqueeze(0)
            z = self.get_z_pred(z, action).squeeze().unsqueeze(0)

        return {
            'latent_state': z,
        }
