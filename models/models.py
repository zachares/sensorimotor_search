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
        self.force_size = force_size
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.use_fft = use_fft
        self.min_steps = min_steps

        self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + self.num_options + 5

        if self.use_fft:
            self.frc_enc = CONV2DN(save_folder + "_fft_enc", load_folder + "_fft_enc", (self.force_size[0], 126, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126
        else:
            self.frc_enc = CONV1DN(save_folder + "_frc_enc", load_folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.options_class = ResNetFCN(save_folder + "_options_class", load_folder + "_options_class", self.state_size, 3, 3, device = self.device)

        self.options_transenc = Transformer_Encoder(save_folder + "_options_trans_enc", load_folder + "_options_trans_enc", self.state_size, 6, device = self.device)
        self.options_transdec = Transformer_Decoder(save_folder + "_options_trans_dec", load_folder + "_options_trans_dec", self.state_size, 6, device = self.device)

        self.model_list.append(self.options_transenc)
        self.model_list.append(self.options_transdec)
        self.model_list.append(self.options_class)
        self.model_list.append(self.frc_enc)

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

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)

        steps = int(np.random.choice(np.arange(self.min_steps, actions.size(1))))

        proprio_diffs = proprios[:,1:steps + 1] - proprios[:,:steps]
        contact_diffs = contacts[:,1:steps + 1] - contacts[:,:steps]
        forces_clipped = forces[:,1:steps + 1]
        actions_clipped = actions[:,:steps]

        # print(forces.size())
        # print(torch.reshape(forces_clipped, (forces_clipped.size(0) * forces_clipped.size(1), forces_clipped.size(2), forces_clipped.size(3))).size())

        frc_encs = torch.reshape(self.get_frc(torch.reshape(forces_clipped,\
         (forces_clipped.size(0) * forces_clipped.size(1), forces_clipped.size(2), forces_clipped.size(3)))),\
        (forces_clipped.size(0), forces_clipped.size(1), self.frc_enc_size))

        # print(frc_encs.size())
        # print(proprio_diffs.size())
        # print(contact_diffs.unsqueeze(2).size())
        # print(actions_clipped.size())
        # print(peg_type.unsqueeze(1).repeat_interleave(forces_clipped.size(1), dim = 1).size())
        # print(torch.zeros_like(proprio_diffs[:,:,:5]).size())

        states = torch.cat([proprio_diffs, frc_encs, contact_diffs.unsqueeze(2), actions_clipped,\
         peg_type.unsqueeze(1).repeat_interleave(forces_clipped.size(1), dim = 1), torch.zeros_like(proprio_diffs[:,:,:5])], dim = 2)
        
        output = self.options_transdec(states[:,-1].unsqueeze(1), states[:,:-1])

        # print(output.size())

        options_logits = self.options_class(output.squeeze(1))

        # print(options_logits.size())

        return {
            'options_class': options_logits,
        }

    def process(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)

        print(proprios.size())
        print(actions.size())
        print(forces.size())
        print(contacts.size())
        print(peg_type.size())

        steps = int(np.random.choice(np.arange(self.min_steps, actions.size(1))))

        proprio_diffs = proprios[:,1:steps + 1] - proprios[:,:steps]
        contact_diffs = contacts[:,1:steps + 1] - contacts[:,:steps]
        forces_clipped = forces[:,1:steps + 1]
        actions_clipped = actions[:,:steps]

        frc_encs = torch.reshape(self.get_frc(torch.reshape(forces_clipped,\
         (forces_clipped.size(0) * forces_clipped.size(1), forces_clipped.size(2), forces_clipped.size(3)))),\
        (forces_clipped.size(0), forces_clipped.size(1), self.frc_enc_size))

        states = torch.cat([proprio_diffs, frc_encs, contact_diffs.unsqueeze(2), actions_clipped,\
         peg_type.unsqueeze(1).repeat_interleave(forces_clipped.size(1), dim = 1), torch.zeros_like(proprio_diffs[:,:,:5])], dim = 2)
        
        output = self.options_transdec(states[:,-1].unsqueeze(1), self.options_transenc(states[:,:-1]))

        options_logits = self.options_class(output.squeeze(1))

        options_probs = F.softmax(options_logits, dim = 1)

        options_guess = torch.zeros_like(options_probs)

        options_guess[torch.arange(options_guess.size(0)), options_probs.max(1)[1]] = 1

        return options_guess