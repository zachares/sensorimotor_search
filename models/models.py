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
TWO_PI = 2 * np.pi

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

def T_angle(angle):

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
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, joint_size, action_dim, num_options, offset, use_fft = True, device = None, curriculum = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            folder = info_flow[model_name]["model_folder"] + model_name
            self.save_bool = True
        else:
            folder = model_folder + model_name
            self.save_bool = True

        self.curriculum = curriculum
        self.device = device
        self.model_list = []

        self.action_dim = action_dim[0]
        self.pose_size = int(proprio_size[0] / 2)
        self.vel_size = int(proprio_size[0] / 2)
        self.joint_pose_size = joint_size[0]
        self.joint_vel_size = joint_size[0]
        self.force_size = force_size
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 48
        self.np = 20
        self.use_fft = use_fft

        self.normalization = simple_normalization #nn.Softmax(dim=1)
        self.prob_calc = nn.Softmax(dim=1)

        self.state_size = (self.frc_enc_size + self.pose_size + self.vel_size + self.joint_pose_size + self.joint_vel_size + self.contact_size) 

        if self.use_fft:
            self.frc_enc = CONV2DN(folder + "_fft_enc", (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)
        else:
            self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.state_size - self.vel_size - self.joint_vel_size + 1, 8, device = self.device)

        # self.time_step = Params(folder + "_timestep", (1), device = self.device)

        self.model_list.append(self.dyn)
        # self.model_list.append(self.time_step)
        self.model_list.append(self.frc_enc)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force):
        if self.use_fft:
            fft = torch.rfft(force, 2, normalized=False, onesided=True)
            frc_enc = self.frc_enc(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        else:
            frc_enc = self.frc_enc(force)

        return frc_enc

    def forward(self, input_dict):
        actions = input_dict["action"].to(self.device)
        proprios = input_dict["proprio"].to(self.device)
        joint_poses = input_dict["joint_pos"].to(self.device)
        joint_vels = input_dict["joint_vel"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        frc_enc_list = []

        frc_pred_list = []
        pose_pred_list = []
        vel_pred_list = []
        contact_pred_list = []
        joint_pose_pred_list = []
        joint_vel_pred_list = []

        peg_type = peg_types[:,0]
        hole_type = hole_types[:,0]

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        proprio = proprios[:,0]

        pos = proprio[:,:3]
        ang = proprio[:,3:6]
        vel = proprio[:,6:9]
        ang_vel = proprio[:,9:12]

        force = forces[:,0]
        frc_enc = self.get_frc(force)
        contact = contacts[:,0]

        joint_pos = joint_poses[:,0]
        joint_vel = joint_vels[:,0]

        state = torch.cat([proprio, joint_pos, joint_vel, frc_enc, contact], dim =1)

        idxs_list = [self.pose_size,\
          self.pose_size + self.joint_pose_size,\
          self.pose_size + self.joint_pose_size + self.frc_enc_size]

        for idx in range(steps):
            force = forces[:,idx]
            force_next = forces[:,idx+1]
            action = actions[:,idx]

            frc_enc = self.get_frc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))

            frc_enc_next = self.get_frc(force_next)
            if idx == steps - 1:
                frc_enc_list.append(frc_enc_next.unsqueeze(1))

            next_state = self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))

            ee_accel = next_state[:,:3]
            ee_ang_accel = next_state[:,3:6]

            joint_accel = next_state[:,idxs_list[0]:idxs_list[1]]

            frc_pred = next_state[:,idxs_list[1]:idxs_list[2]]

            time_step = next_state[:,-2].unsqueeze(1)
            
            contact_logits = next_state[:,-1]

            pos_pred = time_step.repeat_interleave(pos.size(1), 1).pow(2) * 0.5 * ee_accel + time_step.repeat_interleave(pos.size(1), 1) * vel + pos
            vel_pred = time_step.repeat_interleave(pos.size(1), 1) * ee_accel + vel

            ang_pred = T_angle(time_step.repeat_interleave(ang.size(1), 1).pow(2) * 0.5 * ee_ang_accel + time_step.repeat_interleave(ang.size(1), 1) * ang_vel + ang)
            ang_vel_pred = time_step.repeat_interleave(ang.size(1), 1) * ee_ang_accel + ang_vel

            joint_pos_pred = T_angle(time_step.repeat_interleave(joint_pos.size(1), 1).pow(2) * 0.5 * joint_accel + time_step.repeat_interleave(joint_pos.size(1), 1) * joint_vel + joint_pos)
            joint_vel_pred = time_step.repeat_interleave(joint_pos.size(1), 1) * joint_accel + joint_vel

            contact_probs = torch.sigmoid(contact_logits)
            contact_pred = torch.where(contact_probs > 0.5, torch.ones_like(contact_probs), torch.zeros_like(contact_probs))

            state = torch.cat([pos_pred, ang_pred, vel_pred, ang_vel_pred, joint_pos_pred, joint_vel_pred, frc_pred, contact_pred.unsqueeze(1)], dim = 1)

            pos = pos_pred.clone()
            vel = vel_pred.clone()
            ang = ang_pred.clone()
            ang_vel = ang_vel_pred.clone()

            joint_pos = joint_pos_pred.clone()
            joint_vel = joint_vel_pred.clone()

            if idx >= self.offset:
                pose_pred_list.append(torch.cat([pos_pred, ang_pred], dim = 1))
                vel_pred_list.append(torch.cat([vel_pred, ang_vel_pred], dim =1))
                joint_pose_pred_list.append(joint_pos_pred)
                joint_vel_pred_list.append(joint_vel_pred)
                frc_pred_list.append(frc_pred)
                contact_pred_list.append(contact_logits.squeeze())

        return {
            "pose_pred": pose_pred_list,
            "vel_pred": vel_pred_list,
            "joint_pose_pred": joint_pose_pred_list,
            "joint_vel_pred": joint_vel_pred_list,
            "frc_pred": frc_pred_list,
            "contact_pred": contact_pred_list,
            "frc_enc": torch.cat(frc_enc_list, dim = 1),
        }

class Options_ClassifierLSTM(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, z_dim, num_options, offset, use_fft = True, learn_rep = True, device = None, curriculum = None):
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
        self.frc_enc_size = 48
        self.learn_rep = learn_rep
        self.use_fft = use_fft
        
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

        self.options_lstm = LSTMCell(folder + "_options", self.state_size, self.z_dim, device = self.device)
        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)
        self.options_class = FCN(folder + "_options_class", z_dim, 3, 3, device = self.device)

        self.options_transformer = Transformer(folder + "_options", self.z_dim, 2, 2, self.z_dim, device = self.device)

        if self.use_fft:
            self.frc_enc = CONV2DN(folder + "_fft_enc", (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)
        else:
            self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.model_list.append(self.options_lstm)
        self.model_list.append(self.options_transformer)
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


    def get_frc(self, force):
        if self.use_fft:
            fft = torch.rfft(force, 2, normalized=False, onesided=True)
            frc_enc = self.frc_enc(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        else:
            frc_enc = self.frc_enc(force)

        return frc_enc

    def get_options_class(self, proprio, frc_enc, contact, action, peg_type, hole_type, h = None, c = None, h_list = None, calc_logits = False):
        prestate = torch.cat([proprio, frc_enc, contact.unsqueeze(1), action, peg_type, hole_type], dim = 1)
        state = self.pre_lstm(prestate)

        if h is None or c is None:
            h_pred, c_pred = self.options_lstm(state)
        else:
            h_pred, c_pred = self.options_lstm(state, h, c) 

        if len(h_list) < 2:
            options_logits = self.options_class(h_pred)
        else:
            hidden = self.options_transformer(torch.cat(h_list, dim = 0),\
                h_pred.unsqueeze(0)).view(h_pred.size(0), h_pred.size(1))

            options_logits = self.options_class(hidden)

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
        h_list = []

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

            frc_enc = self.get_frc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))

            if idx == 0:
                options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs, h_list = h_list)
                hole_probs = self.softmax(options_logits)
            # stops gradient after a certain number of steps
            # elif idx != 0 and idx % 8 == 0:
            #     h_clone = h.detach()
            #     c_clone = c.detach()
            #     options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs, h_clone, c_clone)
            #     hole_probs = self.softmax(options_logits)
            else:
                options_logits, h, c = self.get_options_class(proprio, frc_enc, contact, action, peg_type, hole_probs, h = h, c = c, h_list = h_list)
                hole_probs = self.softmax(options_logits)

            h_list.append(h.unsqueeze(0))

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

