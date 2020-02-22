import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
import time
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

def simple_normalization(vector, dimen = 1):
    return vector / vector.norm(p=2, dim = dimen).unsqueeze(dimen).repeat_interleave(vector.size(dimen), dim = dimen)

def log_normal(x, m, v):
    return -0.5 * ((x - m).pow(2)/ v + torch.log(2 * np.pi * v)).sum(-1)

def weighted_var(x, m, weights):
    return (weights.unsqueeze(2).repeat_interleave(x.size(2), 2) * (x - m.unsqueeze(1).repeat_interleave(x.size(1), 1)).pow(2)).sum(1)

def filter_depth(depth_image):
    depth_image = torch.where( depth_image > 1e-7, depth_image, torch.zeros_like(depth_image))
    return torch.where( depth_image < 2, depth_image, torch.zeros_like(depth_image))

def T_angle(angle):
    TWO_PI = 2 * np.pi
    ones = torch.ones_like(angle)
    zeros = torch.zeros_like(angle)

    case1 = torch.where(angle < -TWO_PI, angle + TWO_PI * ((torch.abs(angle) / TWO_PI).floor() + 1), zeros )
    case2 = torch.where(angle > TWO_PI, angle - TWO_PI * (angle / TWO_PI).floor(), zeros)
    case3 = torch.where(angle > -TWO_PI, ones, zeros) * torch.where(angle < 0, TWO_PI + angle, zeros)
    case4 = torch.where(angle < TWO_PI, ones, zeros) * torch.where(angle > 0, angle, zeros)

    return case1 + case2 + case3 + case4

def cholesky_dec(noise_matrix): #converting noise parameters to cholesky decomposition form
    chol_dec = noise_matrix.tril()
    chol_diag = torch.abs(chol_dec.diag())
    chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] *= 0
    chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] += chol_diag     
    return chol_dec
#######################################
# Defining Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel
class Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, rgbd_size, force_size, proprio_size, joint_size, action_dim, num_options, offset, use_fft = True, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.curriculum = curriculum
        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.pos_size = int(proprio_size[0] / 4)
        self.ang_size = int(proprio_size[0] / 4)
        self.vel_size = int(proprio_size[0] / 2)
        self.joint_size = 2 * joint_size[0]
        self.force_size = force_size
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 8 * 3 * 2
        self.rgbd_size = rgbd_size
        self.rgbd_enc_size = 8 * 4 * 2
        self.use_fft = use_fft

        self.normalization = simple_normalization

        self.oned_state_size = self.ang_size + self.vel_size + self.joint_size
        self.state_size = self.frc_enc_size + self.pos_size + self.oned_state_size + self.contact_size + self.rgbd_enc_size

        if self.use_fft:
            self.frc_enc = CONV2DN(save_folder + "_fft_enc", load_folder + "_fft_enc", (self.force_size[0], 126, 2), (8, 3, 2), False, True, 4, device = self.device) #10 HZ - 27 # 2 HZ - 126
        else:
            self.frc_enc = CONV1DN(save_folder + "_frc_enc", load_folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 4, device = self.device)

        self.rgbd_enc = CONV2DN(save_folder + "_rgbd_enc", load_folder + "_rgbd_enc", self.rgbd_size, (8, 4, 2), False, True, 3, device = self.device)

        self.dyn = ResNetFCN(save_folder + "_dynamics", load_folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options + 1, self.pos_size, 10, device = self.device)

        self.model_list.append(self.dyn)
        self.model_list.append(self.frc_enc)
        self.model_list.append(self.rgbd_enc)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force):
        if self.use_fft:
            # print(force.size())
            fft = torch.rfft(force, 2, normalized=False, onesided=True)
            # print(fft.size())
            # frc_enc = self.frc_enc(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
            frc_enc = self.frc_enc(fft)
        else:
            frc_enc = self.frc_enc(force)

        return frc_enc

    def get_latent(self, state):
        return self.latent_enc(state)

    def forward(self, input_dict):
        actions = input_dict["action"].to(self.device)
        rgbds = input_dict["rgbd"].to(self.device) / 255.0
        proprios = input_dict["proprio"].to(self.device)
        joint_poses = input_dict["joint_pos"].to(self.device)
        joint_vels = input_dict["joint_vel"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        pos_diff_dir_list = []
        pos_diff_mag_list = []

        peg_type = peg_types[:,0]
        hole_type = hole_types[:,0]

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        proprio = proprios[:,0]

        pos_pred = proprio[:,:3]
        
        ang = proprio[:,3:6]
        vel = proprio[:,6:12]
        joint_pos = joint_poses[:,0]
        joint_vel = joint_vels[:,0]

        oned_state = torch.cat([ang, vel, joint_pos, joint_vel], dim = 1)

        force = forces[:,0]
        frc_enc = self.get_frc(force)

        rgbd = rgbds[:,0]
        rgbd_enc = self.rgbd_enc(rgbd)

        contact = contacts[:,0]

        fixed_state = torch.cat([oned_state, frc_enc, rgbd_enc, contact, peg_type, hole_type], dim =1)

        counter = torch.zeros_like(contact)

        for idx in range(steps):
            action = actions[:,idx]

            #### dynamics
            next_state = self.dyn(torch.cat([pos_pred, action, counter, fixed_state], dim = 1))

            pos_diff_pred = next_state[:,:self.pos_size]

            pos_pred = pos_diff_pred + pos_pred

            counter += 1

            if idx >= self.offset:
                pos_diff_dir_list.append(self.normalization(pos_diff_pred))
                pos_diff_mag_list.append(pos_diff_pred.norm(2,dim = 1))

        return {
            "pos_diff_dir": pos_diff_dir_list,
            "pos_diff_mag": pos_diff_mag_list,
        }

class Options_ClassifierTransformer(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, min_steps, use_fft = True, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.proprio_size = proprio_size[0]
        self.pose_size = 3
        self.force_size = force_size
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.state_size = 128
        self.encoded_size = self.state_size - self.frc_enc_size
        self.use_fft = use_fft
        self.min_steps = min_steps

        self.prestate_size = self.proprio_size + self.contact_size + self.action_dim



        self.ensemble_list = []

        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 01
        self.frc_enc01 = CONV2DN(save_folder + "_fft_enc01", load_folder + "_fft_enc01",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc01 = ResNetFCN(save_folder + "_state_enc01", load_folder + "_state_enc01",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec01 = Transformer_Decoder(save_folder + "_options_transdec01", load_folder + "_options_transdec01",\
         self.state_size, 6, device = self.device)

        self.options_class01 = ResNetFCN(save_folder + "_options_class01", load_folder + "_options_class01",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec01 = Transformer_Decoder(save_folder + "_origin_transdec01", load_folder + "_origin_transdec01",\
         self.state_size, 6, device = self.device)

        self.origin_det01 = ResNetFCN(save_folder + "_origin_det01", load_folder + "_origin_det01",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise01 = Params(save_folder + "_origin_noise01", load_folder + "_origin_noise01", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc01) 
        self.model_list.append(self.state_enc01) 
        self.model_list.append(self.options_transdec01) 
        self.model_list.append(self.options_class01) 
        self.model_list.append(self.origin_transdec01) 
        self.model_list.append(self.origin_det01) 
        self.model_list.append(self.origin_noise01) 

        self.ensemble_list.append((self.frc_enc01, self.state_enc01, self.options_transdec01, self.options_class01,\
         self.origin_transdec01, self.origin_det01, self.origin_noise01))

        # Network 02
        self.frc_enc02 = CONV2DN(save_folder + "_fft_enc02", load_folder + "_fft_enc02",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc02 = ResNetFCN(save_folder + "_state_enc02", load_folder + "_state_enc02",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec02 = Transformer_Decoder(save_folder + "_options_transdec02", load_folder + "_options_transdec02",\
         self.state_size, 6, device = self.device)

        self.options_class02 = ResNetFCN(save_folder + "_options_class02", load_folder + "_options_class02",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec02 = Transformer_Decoder(save_folder + "_origin_transdec02", load_folder + "_origin_transdec02",\
         self.state_size, 6, device = self.device)

        self.origin_det02 = ResNetFCN(save_folder + "_origin_det02", load_folder + "_origin_det02",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise02 = Params(save_folder + "_origin_noise02", load_folder + "_origin_noise02", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc02) 
        self.model_list.append(self.state_enc02) 
        self.model_list.append(self.options_transdec02) 
        self.model_list.append(self.options_class02) 
        self.model_list.append(self.origin_transdec02) 
        self.model_list.append(self.origin_det02) 
        self.model_list.append(self.origin_noise02) 

        self.ensemble_list.append((self.frc_enc02, self.state_enc02, self.options_transdec02, self.options_class02,\
         self.origin_transdec02, self.origin_det02, self.origin_noise02))

        # Network 03
        self.frc_enc03 = CONV2DN(save_folder + "_fft_enc03", load_folder + "_fft_enc03",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc03 = ResNetFCN(save_folder + "_state_enc03", load_folder + "_state_enc03",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec03 = Transformer_Decoder(save_folder + "_options_transdec03", load_folder + "_options_transdec03",\
         self.state_size, 6, device = self.device)

        self.options_class03 = ResNetFCN(save_folder + "_options_class03", load_folder + "_options_class03",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec03 = Transformer_Decoder(save_folder + "_origin_transdec03", load_folder + "_origin_transdec03",\
         self.state_size, 6, device = self.device)

        self.origin_det03 = ResNetFCN(save_folder + "_origin_det03", load_folder + "_origin_det03",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise03 = Params(save_folder + "_origin_noise03", load_folder + "_origin_noise03", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc03) 
        self.model_list.append(self.state_enc03) 
        self.model_list.append(self.options_transdec03) 
        self.model_list.append(self.options_class03) 
        self.model_list.append(self.origin_transdec03) 
        self.model_list.append(self.origin_det03) 
        self.model_list.append(self.origin_noise03) 

        self.ensemble_list.append((self.frc_enc03, self.state_enc03, self.options_transdec03, self.options_class03,\
         self.origin_transdec03, self.origin_det03, self.origin_noise03))

        # Network 11
        self.frc_enc11 = CONV2DN(save_folder + "_fft_enc11", load_folder + "_fft_enc11",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc11 = ResNetFCN(save_folder + "_state_enc11", load_folder + "_state_enc11",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec11 = Transformer_Decoder(save_folder + "_options_transdec11", load_folder + "_options_transdec11",\
         self.state_size, 6, device = self.device)

        self.options_class11 = ResNetFCN(save_folder + "_options_class11", load_folder + "_options_class11",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec11 = Transformer_Decoder(save_folder + "_origin_transdec11", load_folder + "_origin_transdec11",\
         self.state_size, 6, device = self.device)

        self.origin_det11 = ResNetFCN(save_folder + "_origin_det11", load_folder + "_origin_det11",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise11 = Params(save_folder + "_origin_noise11", load_folder + "_origin_noise11", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc11) 
        self.model_list.append(self.state_enc11) 
        self.model_list.append(self.options_transdec11) 
        self.model_list.append(self.options_class11) 
        self.model_list.append(self.origin_transdec11) 
        self.model_list.append(self.origin_det11) 
        self.model_list.append(self.origin_noise11) 

        self.ensemble_list.append((self.frc_enc11, self.state_enc11, self.options_transdec11, self.options_class11,\
         self.origin_transdec11, self.origin_det11, self.origin_noise11))

        # Network 12
        self.frc_enc12 = CONV2DN(save_folder + "_fft_enc12", load_folder + "_fft_enc12",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc12 = ResNetFCN(save_folder + "_state_enc12", load_folder + "_state_enc12",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec12 = Transformer_Decoder(save_folder + "_options_transdec12", load_folder + "_options_transdec12",\
         self.state_size, 6, device = self.device)

        self.options_class12 = ResNetFCN(save_folder + "_options_class12", load_folder + "_options_class12",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec12 = Transformer_Decoder(save_folder + "_origin_transdec12", load_folder + "_origin_transdec12",\
         self.state_size, 6, device = self.device)

        self.origin_det12 = ResNetFCN(save_folder + "_origin_det12", load_folder + "_origin_det12",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise12 = Params(save_folder + "_origin_noise12", load_folder + "_origin_noise12", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc12) 
        self.model_list.append(self.state_enc12) 
        self.model_list.append(self.options_transdec12) 
        self.model_list.append(self.options_class12) 
        self.model_list.append(self.origin_transdec12) 
        self.model_list.append(self.origin_det12) 
        self.model_list.append(self.origin_noise12) 

        self.ensemble_list.append((self.frc_enc12, self.state_enc12, self.options_transdec12, self.options_class12,\
         self.origin_transdec12, self.origin_det12, self.origin_noise12))

        # Network 13
        self.frc_enc13 = CONV2DN(save_folder + "_fft_enc13", load_folder + "_fft_enc13",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 136

        self.state_enc13 = ResNetFCN(save_folder + "_state_enc13", load_folder + "_state_enc13",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec13 = Transformer_Decoder(save_folder + "_options_transdec13", load_folder + "_options_transdec13",\
         self.state_size, 6, device = self.device)

        self.options_class13 = ResNetFCN(save_folder + "_options_class13", load_folder + "_options_class13",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec13 = Transformer_Decoder(save_folder + "_origin_transdec13", load_folder + "_origin_transdec13",\
         self.state_size, 6, device = self.device)

        self.origin_det13 = ResNetFCN(save_folder + "_origin_det13", load_folder + "_origin_det13",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise13 = Params(save_folder + "_origin_noise13", load_folder + "_origin_noise13", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc13) 
        self.model_list.append(self.state_enc13) 
        self.model_list.append(self.options_transdec13) 
        self.model_list.append(self.options_class13) 
        self.model_list.append(self.origin_transdec13) 
        self.model_list.append(self.origin_det13) 
        self.model_list.append(self.origin_noise13) 

        self.ensemble_list.append((self.frc_enc13, self.state_enc13, self.options_transdec13, self.options_class13,\
         self.origin_transdec13, self.origin_det13, self.origin_noise13))

        # Network 21
        self.frc_enc21 = CONV2DN(save_folder + "_fft_enc21", load_folder + "_fft_enc21",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 216

        self.state_enc21 = ResNetFCN(save_folder + "_state_enc21", load_folder + "_state_enc21",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec21 = Transformer_Decoder(save_folder + "_options_transdec21", load_folder + "_options_transdec21",\
         self.state_size, 6, device = self.device)

        self.options_class21 = ResNetFCN(save_folder + "_options_class21", load_folder + "_options_class21",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec21 = Transformer_Decoder(save_folder + "_origin_transdec21", load_folder + "_origin_transdec21",\
         self.state_size, 6, device = self.device)

        self.origin_det21 = ResNetFCN(save_folder + "_origin_det21", load_folder + "_origin_det21",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise21 = Params(save_folder + "_origin_noise21", load_folder + "_origin_noise21", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc21) 
        self.model_list.append(self.state_enc21) 
        self.model_list.append(self.options_transdec21) 
        self.model_list.append(self.options_class21) 
        self.model_list.append(self.origin_transdec21) 
        self.model_list.append(self.origin_det21) 
        self.model_list.append(self.origin_noise21) 

        self.ensemble_list.append((self.frc_enc21, self.state_enc21, self.options_transdec21, self.options_class21,\
         self.origin_transdec21, self.origin_det21, self.origin_noise21))

        # Network 22
        self.frc_enc22 = CONV2DN(save_folder + "_fft_enc22", load_folder + "_fft_enc22",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 226

        self.state_enc22 = ResNetFCN(save_folder + "_state_enc22", load_folder + "_state_enc22",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec22 = Transformer_Decoder(save_folder + "_options_transdec22", load_folder + "_options_transdec22",\
         self.state_size, 6, device = self.device)

        self.options_class22 = ResNetFCN(save_folder + "_options_class22", load_folder + "_options_class22",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec22 = Transformer_Decoder(save_folder + "_origin_transdec22", load_folder + "_origin_transdec22",\
         self.state_size, 6, device = self.device)

        self.origin_det22 = ResNetFCN(save_folder + "_origin_det22", load_folder + "_origin_det22",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise22 = Params(save_folder + "_origin_noise22", load_folder + "_origin_noise22", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc22) 
        self.model_list.append(self.state_enc22) 
        self.model_list.append(self.options_transdec22) 
        self.model_list.append(self.options_class22) 
        self.model_list.append(self.origin_transdec22) 
        self.model_list.append(self.origin_det22) 
        self.model_list.append(self.origin_noise22) 

        self.ensemble_list.append((self.frc_enc22, self.state_enc22, self.options_transdec22, self.options_class22,\
         self.origin_transdec22, self.origin_det22, self.origin_noise22))

        # Network 23
        self.frc_enc23 = CONV2DN(save_folder + "_fft_enc23", load_folder + "_fft_enc23",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 236

        self.state_enc23 = ResNetFCN(save_folder + "_state_enc23", load_folder + "_state_enc23",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec23 = Transformer_Decoder(save_folder + "_options_transdec23", load_folder + "_options_transdec23",\
         self.state_size, 6, device = self.device)

        self.options_class23 = ResNetFCN(save_folder + "_options_class23", load_folder + "_options_class23",\
            self.state_size, self.num_options, 3, device = self.device)

        self.origin_transdec23 = Transformer_Decoder(save_folder + "_origin_transdec23", load_folder + "_origin_transdec23",\
         self.state_size, 6, device = self.device)

        self.origin_det23 = ResNetFCN(save_folder + "_origin_det23", load_folder + "_origin_det23",\
            self.state_size, self.pose_size, 3, device = self.device)

        self.origin_noise23 = Params(save_folder + "_origin_noise23", load_folder + "_origin_noise23", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc23) 
        self.model_list.append(self.state_enc23) 
        self.model_list.append(self.options_transdec23) 
        self.model_list.append(self.options_class23) 
        self.model_list.append(self.origin_transdec23) 
        self.model_list.append(self.origin_det23) 
        self.model_list.append(self.origin_noise23) 

        self.ensemble_list.append((self.frc_enc23, self.state_enc23, self.options_transdec23, self.options_class23,\
         self.origin_transdec23, self.origin_det23, self.origin_noise23))

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        fft = torch.rfft(force, 2, normalized=False, onesided=True)
        frc_enc = self.model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        # print("Force size: ", force.size())
        # print(fft.size())
        # frc_enc = self.frc_enc(fft)
        return frc_enc

    def get_data(state, forces, model_tuple, batch_size, sequence_size, padding_masks, lengths):
        frc_enc, state_enc, options_transdec, options_class, origin_transdec, origin_det, origin_noise = model_tuple

        frc_encs_unshaped = self.get_frc(forces, frc_enc)
        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size)).contiguous()

        state_encs_unshaped = self.state_enc(state)
        state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size)).contiguous()

        states = torch.cat([state_encs, frc_encs], dim = 2).transpose(0,1).contiguous()
        final_state = states[lengths,torch.arange(states.size(1)), torch.arange(states.size(2))].unsqueeze(0)

        options_logits = options_class(options_transdec(final_state, states, padding_mask = padding_masks).squeeze(0))
        origin_mean = origin_det(origin_transdec(final_state, states, padding_mask = padding_masks).squeeze(0))

        origin_chol_dec = cholesky_dec(origin_noise).unsqueeze(0).repeat_interleave(batch_size, 0)

        return options_logits, origin_mean, origin_chol_dec

    def get_logits(self, proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks, lengths):
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3))).contiguous()

        prestate = torch.cat([proprio_diffs, contact_diffs, actions], dim = 2)
        prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2))).contiguous()

        options_logits01, origin_mean01, origin_chol_dec01 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[0],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits02, origin_mean02, origin_chol_dec02 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[1],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits03, origin_mean03, origin_chol_dec03 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[2],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits11, origin_mean11, origin_chol_dec11 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[3],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits12, origin_mean12, origin_chol_dec12 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[4],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits13, origin_mean13, origin_chol_dec13 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[5],\
         batch_size, sequence_size, padding_masks, lengths)

        options_logits21, origin_mean21, origin_chol_dec21 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[6],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits22, origin_mean22, origin_chol_dec22 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[7],\
         batch_size, sequence_size, padding_masks, lengths)
        options_logits23, origin_mean23, origin_chol_dec23 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[8],\
         batch_size, sequence_size, padding_masks, lengths)
        
        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)

        om_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)
        om_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)
        om_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)

        oc_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(origin_chol_dec01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_chol_dec01.size(2), dim=2)
        oc_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(origin_chol_dec01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_chol_dec01.size(2), dim=2)
        oc_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(origin_chol_dec01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_chol_dec01.size(2), dim=2)

        options_logits1 = ol_peg0 * options_logits01 + ol_peg1 * options_logits11 + ol_peg2 * options_logits21
        options_logits2 = ol_peg0 * options_logits02 + ol_peg1 * options_logits12 + ol_peg2 * options_logits22
        options_logits3 = ol_peg0 * options_logits03 + ol_peg1 * options_logits13 + ol_peg2 * options_logits23

        origin_mean1 = om_peg0 * origin_mean01 + om_peg1 * origin_mean11 + om_peg2 * origin_mean21
        origin_mean2 = om_peg0 * origin_mean02 + om_peg1 * origin_mean12 + om_peg2 * origin_mean22
        origin_mean3 = om_peg0 * origin_mean03 + om_peg1 * origin_mean13 + om_peg2 * origin_mean23

        origin_chol_dec1 = oc_peg0 * origin_chol_dec01 + oc_peg1 * origin_chol_dec11 + oc_peg2 * origin_chol_dec21
        origin_chol_dec2 = oc_peg0 * origin_chol_dec02 + oc_peg1 * origin_chol_dec12 + oc_peg2 * origin_chol_dec22
        origin_chol_dec3 = oc_peg0 * origin_chol_dec03 + oc_peg1 * origin_chol_dec13 + oc_peg2 * origin_chol_dec23

        return [options_logits1, options_logits2, options_logits3], [(origin_mean1, origin_chol_dec1),\
        (origin_mean2, origin_chol_dec2),(origin_mean3, origin_chol_dec3)]

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)
        lengths = input_dict["length"].to(self.device)
        # prev_time = time.time()
        options_logits, origin_params = self.get_logits(proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks, lengths)
        # print("Took ", time.time() - prev_time, " seconds")

        return {
            'options_class': options_logits,
            'origing_params': origin_params,
            # 'options_class_ss': options_logits_ss,
            # 'contrastive': (output, output_ss),
        }

    def process(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)[:, :-1]
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)


        # print(proprios.size())
        # print(actions.size())
        # print(forces.size())
        # print(contacts.size())
        # print(peg_type.size())

        options_logits = self.get_logits(proprios, contacts, forces, actions, peg_type)   

        # print(options_logits)    

        return options_logits

        # print(forces.size())
        # print(torch.reshape(forces_clipped, (forces_clipped.size(0) * forces_clipped.size(1), forces_clipped.size(2), forces_clipped.size(3))).size())
        # print(frc_encs.size())
        # print(proprio_diffs.size())
        # print(contact_diffs.unsqueeze(2).size())
        # print(actions_clipped.size())
        # print(peg_type.unsqueeze(1).repeat_interleave(forces_clipped.size(1), dim = 1).size())
        # print(torch.zeros_like(proprio_diffs[:,:,:5]).size())