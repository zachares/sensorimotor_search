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

class Contact_Force_Multimodal(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, proprio_size, force_size, z_dim, action_dim, device = None, curriculum = None):
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

        self.proprio_enc = FCN(folder + "_proprio_enc", self.proprio_size, 2 * self.z_dim, 5, device = self.device)
        self.proprio_dec = FCN(folder + "_proprio_dec", self.z_dim, self.proprio_size, 5, device = self.device)

        self.frc_enc = FCN(folder + "_frc_enc", self.force_size, 2 * z_dim, 4, device = self.device)
        self.frc_dec = FCN(folder + "_frc_dec", z_dim, self.force_size, 4, device = self.device)

        self.img_enc = CONV2DN(folder + "_img_enc", image_size, (2 * self.z_dim, 1, 1), False, True, 3, device = self.device)
        self.img_dec = DECONV2DN(folder + "_img_dec", (self.z_dim,  2, 2), (image_size[0], image_size[1] / 4, image_size[2] / 4), False, device = self.device)

        self.fusion_module = FCN(folder + "_fusion_unit", 6 * self.z_dim, self.z_dim, 5, device = self.device) 

        self.trans_model_c = ResNetFCN(folder + "_trans_resnet_c", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device)
        self.trans_model_nc = ResNetFCN(folder + "_trans_resnet_nc", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device) 

        self.contact_clas = FCN(folder + "_contact_clas", self.z_dim, 1, 3, device = self.device)    

        self.model_list = [self.fusion_module, self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.frc_enc, self.frc_dec,\
         self.trans_model_c, self.trans_model_nc, self.contact_clas]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_z(self, image, proprio, force):
        img_emb = self.img_enc(image)
        proprio_emb = self.proprio_enc(proprio)
        frc_emb = self.frc_enc(force)
        return self.fusion_module(torch.cat([img_emb, proprio_emb, frc_emb], dim = 1))

    def get_z_pred(self, z, action):
        # return self.trans_model(torch.cat([z, action], dim = 1), z)
        cont = torch.sigmoid(self.get_contact(z))

        contact = torch.where(cont > 0.5, torch.ones_like(cont), torch.zeros_like(cont))
        non_contact = torch.ones_like(contact) - contact

        return (contact.transpose(0,1) * self.trans_model_c(torch.cat([z, action], dim = 1), z).transpose(0,1) + \
        non_contact.transpose(0,1) * self.trans_model_nc(torch.cat([z, action], dim = 1), z).transpose(0,1)).transpose(0,1)


    def get_contact(self, z):
        return self.contact_clas(z)

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

    def get_proprio(self, z):
        return self.proprio_dec(z)

    def get_force(self, z):
        return self.frc_dec(z)

    def forward(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        forces = (input_dict["force"]).to(self.device)
        actions = input_dict["action"].to(self.device)
        contacts = input_dict["contact"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        z_list = []
        z_pred_list = []

        contact_list = []

        eepos_list = []

        img_list = []
        prop_list = []
        frc_list = []

        if self.curriculum is not None:
            steps = images.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = images.size(1)

        for idx in range(steps):
            image = images[:,idx]
            proprio = proprios[:, idx]
            force = forces[:,idx]
            action = actions[:,idx]
            contact = contacts[:,idx]

            # print(contact)

            z = self.get_z(image, proprio, force)
            z_list.append(z)

            if idx == 0:
                z_pred = z.clone()

            img = self.get_image(z)
            img_list.append(img)

            prop = self.get_proprio(z)
            prop_list.append(prop)

            frc = self.get_force(z)
            frc_list.append(frc)

            cont = self.get_contact(z)
            contact_list.append((cont.squeeze(), contact))

            if idx != steps - 1:
                z_pred = self.get_z_pred(z_pred, action)
                z_pred_list.append(z_pred)

            eepos_list.append(proprio[:,:3])

        return {
            'z': z_list,
            'z_pred': z_pred_list,
            'contact': contact_list,
            # 'z_dist': (z_list, eepos_list),
            'image_dec': img_list,
            'prop_dec': prop_list,
            'frc_dec': frc_list,
        }

    def encode(self, input_dict):
        images = (input_dict["image"] / 255.0).to(self.device)
        proprios = (input_dict["proprio"]).to(self.device)
        forces = (input_dict["force"]).to(self.device)

        return {
            'latent_state': self.get_z(images[:,0], proprios[:,0], forces[:,0]),
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

class PlaNet_Multimodal(Proto_Macromodel):
    def __init__(self, model_name, image_size, force_size, z_dim, action_dim, device = None, curriculum = None):
        super().__init__()

        self.curriculum = curriculum
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
        images = (input_dict["image"] / 255.0).to(self.device)
        forces = (input_dict["force"]).to(self.device)
        actions = (input_dict["action"]).to(self.device)
        epoch = input_dict["epoch"]

        belief_state_post = None
        cell_state_post = None
        belief_state_prior = None
        cell_state_prior = None

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(forces[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        prev_state_post = sample_gaussian(mu_z, var_z, self.device)
        prev_state_prior = prev_state_post.clone()

        params_list = []
        img_dec_list = []
        frc_dec_list = []

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

            belief_state_post, cell_state_post = self.det_state_model(torch.cat([prev_state_post, action], dim = 1), belief_state_post,  cell_state_post)

            belief_state_prior, cell_state_prior = self.det_state_model(torch.cat([prev_state_prior, action], dim = 1), belief_state_prior,  cell_state_prior)

            # print("Belief state size: ", belief_state.size())
            # print("Cell state size: ", cell_state.size())
            
            hid_mu, hid_var = gaussian_parameters(self.hidden_enc(belief_state_post))
            img_mu, img_var = gaussian_parameters(self.img_enc(image))
            frc_mu, frc_var = gaussian_parameters(self.frc_enc(force)) 
            trans_mu, trans_var = gaussian_parameters(self.trans_model(belief_state_prior))

            # print("Image mean size: ", img_mu.size())
            # print("Force mean size: ", frc_mu.size())
            # print("Dynamics mean size: ", trans_mu.size())
            # print("Hidden mean size: ", hid_mu.size())

            mu_vect = torch.cat([hid_mu, img_mu, frc_mu, trans_mu], dim = 2)
            var_vect = torch.cat([hid_var, img_var, frc_var, trans_var], dim = 2)

            mu_z, var_z = product_of_experts(mu_vect, var_vect)

            prev_state_post = sample_gaussian(mu_z, var_z, self.device)
            prev_state_prior = sample_gaussian(trans_mu.squeeze(), trans_var.squeeze(), self.device)

            img_dec = 255.0 * torch.sigmoid(self.img_dec(prev_state_post))
            img_dec = F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')
            frc_dec = self.frc_dec(prev_state_post)

            params = (mu_z, var_z, trans_mu.squeeze(), trans_var.squeeze())

            params_list.append(params)
            img_dec_list.append(img_dec)
            frc_dec_list.append(frc_dec)


        return {
            'params': params_list,
            'image_pred': img_dec_list,
            'force_pred': frc_dec_list,
        }

    def encode(self, input_dict):

        images = (input_dict["image"] / 255.0).to(self.device)
        forces = (input_dict["force"]).to(self.device)

        img_mu, img_var = gaussian_parameters(self.img_enc(images[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(forces[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)   

        return {
            'latent_state': mu_z,
        }

    def trans(self, input_dict):
        image = (input_dict["image"] / 255.0).to(self.device)
        force = (input_dict["force"]).to(self.device)
        actions = (input_dict["action_sequence"]).to(self.device)

        belief_state_post = None
        cell_state_post = None
        belief_state_prior = None
        cell_state_prior = None

        img_mu, img_var = gaussian_parameters(self.img_enc(image[:,0]))
        frc_mu, frc_var = gaussian_parameters(self.frc_enc(force[:,0])) 

        mu_vect = torch.cat([img_mu, frc_mu], dim = 2)
        var_vect = torch.cat([img_var, frc_var], dim = 2)
        mu_z, var_z = product_of_experts(mu_vect, var_vect)

        prev_state_prior = mu_z


        for idx in range(actions.size(1)):
            action = actions[idx].unsqueeze(0)

            # print("Actions size: ", action.size())
            # print("Prev state prior: ", prev_state_prior.size())

            belief_state_prior, cell_state_prior = self.det_state_model(torch.cat([prev_state_prior, action], dim = 1), belief_state_prior,  cell_state_prior)

            trans_mu, trans_var = gaussian_parameters(self.trans_model(belief_state_prior))

            prev_state_prior = trans_mu.squeeze().unsqueeze(0)

            # print("Belief state prior size: ", belief_state_prior.size())
            # print("Cell state prior size: ", cell_state_prior.size())
            # print("trans_mu size: ", trans_mu.size())


        return {
            'latent_state': prev_state_prior,
        }

    def get_image(self, z):
        img_dec = 255.0 * torch.sigmoid(self.img_dec(z))
        return F.interpolate(img_dec, size=(self.image_size[1], self.image_size[2]), mode='bilinear')

class Dynamics_VarModel(Proto_Macromodel):
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

        self.trans_model = ResNetFCN(folder + "_trans_resnet", self.z_dim + self.action_dim, 2*self.z_dim, 3, device = self.device)
        self.model_list.append(self.trans_model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"]) 

    def get_z(self, z, action):
        mu_z, var_z = gaussian_parameters(self.trans_model(torch.cat([z, action], dim = 1), torch.cat([z, z], dim = 1)))

        z = sample_gaussian(mu_z.squeeze(), var_z.squeeze(), self.device).squeeze()

        return z, mu_z.squeeze(), var_z.squeeze()

    def forward(self, input_dict):
        z_list = input_dict["z"]
        params = input_dict["params"]
        actions = input_dict["action"].to(self.device)
        contact = input_dict["contact"]
        epoch =  int(input_dict["epoch"].detach().item())

        param = params[0]
        z_mu_enc, z_var_enc, prior_mu, prior_var = param

        z_pred = z_mu_enc.clone()

        params_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)       

        for idx in range(steps - 1):
            action = actions[:,idx]
            param = params[idx+1]

            z_mu_enc, z_var_enc, prior_mu, prior_var = param
            z_pred, mu_z, var_z = self.get_z(z_pred, action)

            params_list.append((mu_z, var_z, z_mu_enc, z_var_enc))

        return {
            "params": params_list,
        }

    def trans(self, input_dict):
        mu_z = input_dict["z"].to(self.device).squeeze()
        actions = (input_dict["action_sequence"]).to(self.device)

        steps = actions.size(0)

        for idx in range(steps):
            action = actions[idx].squeeze()
            z, mu_z, var_z = self.get_z(mu_z.squeeze().unsqueeze(0), action.squeeze().unsqueeze(0))

        return {
            'latent_state': mu_z,
        }

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

class Contact_Multimodal(Proto_Macromodel):
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

        self.trans_model_c = ResNetFCN(folder + "_trans_resnet_c", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device)
        self.trans_model_nc = ResNetFCN(folder + "_trans_resnet_nc", self.z_dim + self.action_dim, self.z_dim, 3, device = self.device) 

        self.contact_clas = FCN(folder + "_contact_clas", self.z_dim, 1, 3, device = self.device)    

        self.model_list = [self.fusion_module, self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.trans_model_c, self.trans_model_nc, self.contact_clas]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_z(self, image, proprio):
        img_emb = self.img_enc(image)
        proprio_emb = self.proprio_enc(proprio)
        return self.fusion_module(torch.cat([img_emb, proprio_emb], dim = 1))

    def get_z_pred(self, z, action):
        # return self.trans_model(torch.cat([z, action], dim = 1), z)
        cont = torch.sigmoid(self.get_contact(z))

        contact = torch.where(cont > 0.5, torch.ones_like(cont), torch.zeros_like(cont))
        non_contact = torch.ones_like(contact) - contact

        # print("Dynamics Size: ", (contact * self.trans_model_c(torch.cat([z, action], dim = 1), z)).size() )

        return (contact.transpose(0,1) * self.trans_model_c(torch.cat([z, action], dim = 1), z).transpose(0,1) + \
        non_contact.transpose(0,1) * self.trans_model_nc(torch.cat([z, action], dim = 1), z).transpose(0,1)).transpose(0,1)

    def get_contact(self, z):
        return self.contact_clas(z)

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
        z_pred_list = []

        contact_list = []

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

            cont = self.get_contact(z)
            contact_list.append((cont.squeeze(), contact))

            if idx != steps - 1:
                z_pred = self.get_z_pred(z_pred, action)
                z_pred_list.append(z_pred)

            eepos_list.append(proprio[:,:3])

        return {
            'z': z_list,
            'z_pred': z_pred_list,
            'contact': contact_list,
            # 'z_dist': (z_list, eepos_list),
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
