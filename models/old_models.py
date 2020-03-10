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
        
class Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, depth_size, force_size, proprio_size, joint_size, action_dim, num_options, offset, use_fft = True, device = None, curriculum = None):
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
        self.frc_enc_size = 8 * 3 * 2
        self.image_size = image_size
        self.img_enc_size = 8 * 4 * 2
        self.dpth_enc_size = 8 * 4 * 2
        self.np = 20
        self.use_fft = use_fft

        self.normalization = simple_normalization
        self.softmax = nn.Softmax(dim=1)

        self.latent_state_size = int(self.pose_size / 2) + self.vel_size + self.joint_pose_size + self.joint_vel_size
        self.state_size = self.frc_enc_size + int(self.pose_size / 2) + + self.latent_state_size + self.contact_size + self.img_enc_size
        self.surr_state_size = self.latent_state_size + self.frc_enc_size + self.contact_size + 2 * self.num_options + self.img_enc_size
        self.idxs = [int(self.pose_size / 2), int(self.pose_size / 2) + self.latent_state_size, int(self.pose_size / 2) + self.latent_state_size + self.frc_enc_size]

        if self.use_fft:
            self.frc_enc = CONV2DN(folder + "_fft_enc", (self.force_size[0], 126, 2), (8, 3, 2), False, True, 4, device = self.device) #10 HZ - 27 # 2 HZ - 126
        else:
            self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.rgbd_enc = CONV2DN(folder + "_img_enc", image_size, (8, 4, 2), False, True, 3, device = self.device)

        # self.latent_enc = ResNetFCN(folder + "_latent_enc", self.latent_state_size, self.latent_state_size, 5, device = self.device)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, int(self.pose_size / 2), 10, device = self.device)

        self.surrogate = ResNetFCN(folder + "_surrogate", self.surr_state_size, int(self.pose_size / 2), 3, device = self.device)

        self.model_list.append(self.dyn)
        self.model_list.append(self.frc_enc)
        self.model_list.append(self.rgbd_enc)
        # self.model_list.append(self.depth_enc)
        # self.model_list.append(self.latent_enc)
        self.model_list.append(self.surrogate)

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
        images = input_dict["image"].to(self.device) / 255.0
        depths = input_dict["depth"].to(self.device).unsqueeze(2) / 255.0
        proprios = input_dict["proprio"].to(self.device)
        joint_poses = input_dict["joint_pos"].to(self.device)
        joint_vels = input_dict["joint_vel"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        frc_enc_diff_list = []
        latent_state_diff_list = []

        frc_enc_diff_est_list = []
        latent_state_diff_est_list = []

        pos_diff_dir_list = []
        pos_diff_mag_list = []
        pos_surr_est_list = []

        contact_pred_list = []

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
        
        ang_pred = proprio[:,3:6]
        vel_pred = proprio[:,6:12]
        joint_pos_pred = joint_poses[:,0]
        joint_vel_pred = joint_vels[:,0]

        latent_state_pred = torch.cat([ang_pred, vel_pred, joint_pos_pred, joint_vel_pred], dim = 1)

        force = forces[:,0]
        frc_enc_pred = self.get_frc(force)

        depth = depths[:,0]
        image = images[:,0]
        # print(torch.cat([image, depth] dim = 1).size())
        rgbd_enc_pred = self.rgbd_enc(torch.cat([image, depth] dim = 1))

        contact = contacts[:,0]

        state = torch.cat([pos_pred, latent_state_pred, frc_enc_pred, rgbd_enc_pred, contact], dim =1)
        surr_state = torch.cat([latent_state_pred, frc_enc_pred, rgbd_enc_pred, contact, peg_type, hole_type], dim = 1)

        for idx in range(steps):
            action = actions[:,idx]

            # ### encoding force
            # force = forces[:,idx]
            # frc_enc = self.get_frc(force)

            # force_next = forces[:,idx+1]
            # frc_enc_next = self.get_frc(force_next)

            # frc_enc_diff = frc_enc_next - frc_enc

            # frc_enc_diff_list.append(frc_enc_diff.unsqueeze(1))

            # ### encoding vel
            # proprio = proprios[:,idx]
            # ang = proprio[:,3:6]
            # vel = proprio[:,6:12]
            # joint_pos = joint_poses[:,idx]
            # joint_vel = joint_vels[:,idx]

            # latent_state = self.get_latent(torch.cat([ang, vel, joint_pos, joint_vel], dim = 1))

            # proprio_next = proprios[:,idx + 1]
            # ang_next = proprio_next[:,3:6]
            # vel_next = proprio_next[:,6:12]
            # joint_pos_next = joint_poses[:,idx + 1]
            # joint_vel_next = joint_vels[:,idx + 1]

            # latent_state_next = self.get_latent(torch.cat([ang_next, vel_next, joint_pos_next, joint_vel_next], dim = 1))

            # latent_state_diff = latent_state_next - latent_state

            # latent_state_diff_list.append(latent_state_diff.unsqueeze(1))

            # #### surrogate model calculatinges
            # pos_sur_est = self.surrogate(surr_state)

            #### dynamics
            next_state = self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))

            pos_diff_pred = next_state[:,:self.idxs[0]]
            # latent_state_diff_pred = next_state[:,self.idxs[0]:self.idxs[1]]
            # frc_enc_diff_pred = next_state[:, self.idxs[1]:self.idxs[2]]
            # rgbd_enc_diff_pred = next_state[:, self.idxs[2]:-1]
            # contact_logits = next_state[:, -1]

            # contact_probs = torch.sigmoid(contact_logits)
            # contact_pred = torch.where(contact_probs > 0.5, torch.ones_like(contact_probs), torch.zeros_like(contact_probs))

            pos_pred = pos_diff_pred + pos_pred
            # latent_state_pred = latent_state_diff_pred + latent_state_pred
            # frc_enc_pred = frc_enc_diff_pred + frc_enc_pred
            # rgbd_enc_pred = rgbd_enc_diff_pred + rgbd_enc_pred

            state = torch.cat([pos_pred, latent_state_pred, frc_enc_pred, rgbd_enc_pred, contact_pred.unsqueeze(1)], dim = 1)
            # surr_state = torch.cat([latent_state_pred, frc_enc_pred, rgbd_enc_pred, contact, peg_type, hole_type], dim = 1)

            if idx >= self.offset:
                pos_diff_dir_list.append(self.normalization(pos_diff_pred))
                pos_diff_mag_list.append(pos_diff_pred.norm(2,dim = 1))

                # frc_enc_diff_est_list.append(frc_enc_diff_pred)
                # latent_state_diff_est_list.append(latent_state_diff_pred)

                # pos_surr_est_list.append(pos_sur_est)

                # contact_pred_list.append(contact_logits.squeeze())

        return {
            "pos_diff_dir": pos_diff_dir_list,
            "pos_diff_mag": pos_diff_mag_list,
            # "contact_pred": contact_pred_list,
            # "latent_state_diff_est": latent_state_diff_est_list,
            # "latent_state_diff": torch.cat(latent_state_diff_list, dim = 1),
            # "frc_enc_diff_est": frc_enc_diff_est_list,
            # "frc_enc_diff": torch.cat(frc_enc_diff_list, dim = 1),
            # "pos_surr_est": pos_surr_est_list,
        }

class DynamicswForce(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, offset, use_fft = True, device = None, curriculum = None):
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
        self.force_size = force_size
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 48
        self.np = 20
        self.use_fft = use_fft

        self.normalization = simple_normalization #nn.Softmax(dim=1)
        self.prob_calc = nn.Softmax(dim=1)

        self.state_size = (self.frc_enc_size + self.pose_size + self.vel_size + self.contact_size) 

        if self.use_fft:
            self.frc_enc = CONV2DN(folder + "_fft_enc", (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)
        else:
            self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        # self.dyn_pose = ResNetFCN(folder + "_dynamics_proprio_pose", self.state_size + self.action_dim + 2 * self.num_options, self.pose_size , 4, device = self.device)
        # self.contact_class = ResNetFCN(folder + "_contact_class", self.pose_size  + 2 * self.num_options, 1, 3, device = self.device) 
        # self.dyn_frc = ResNetFCN(folder + "_dynamics_frc", self.state_size + self.action_dim + self.pose_size + self.vel_size + self.contact_size +\
        #   2 * self.num_options, self.frc_enc_size, 4, device = self.device)

        # self.dyn_vel = ResNetFCN(folder + "_dynamics_proprio_vel", self.state_size + self.action_dim + self.pose_size +\
        #  self.contact_size + 2 * self.num_options, self.vel_size , 4, device = self.device)

        # self.model_list.append(self.dyn_pose)
        # self.model_list.append(self.dyn_vel)
        # self.model_list.append(self.dyn_frc)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.state_size , 8, device = self.device)

        self.model_list.append(self.dyn)
        self.model_list.append(self.frc_enc)
        # self.model_list.append(self.contact_class)

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
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        frc_enc_list = []

        frc_tl_list = []
        vel_tl_list = []
        contact_tl_list = []

        frc_pred_list = []
        pose_pred_list = []
        vel_pred_list = []
        contact_pred_list = []

        peg_type = peg_types[:,0]
        hole_type = hole_types[:,0]

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            proprio = proprios[:,idx]
            proprio_next = proprios[:,idx + 1]
            pose_next = proprio_next[:,:self.pose_size]
            vel_next = proprio_next[:,self.pose_size:]
            force = forces[:,idx]
            force_next = forces[:,idx+1]
            contact = contacts[:,idx]
            contact_next = contacts[:,idx+1]
            action = actions[:,idx]

            frc_enc = self.get_frc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))

            frc_enc_next = self.get_frc(force_next)
            if idx == steps - 1:
                frc_enc_list.append(frc_enc_next.unsqueeze(1))

            state = torch.cat([proprio, frc_enc, contact], dim =1)

            next_state = self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))

            pose_pred = next_state[:,:self.pose_size]
            vel_pred = next_state[:,self.pose_size:(self.pose_size + self.vel_size)]
            frc_pred = next_state[:,(self.pose_size + self.vel_size):(self.pose_size + self.vel_size + self.frc_enc_size)]
            contact_logits = next_state[:,-1]
            # state_next = torch.cat([proprio_next, frc_enc_next, contact_next], dim =1)

            # pose_pred = self.dyn_pose(torch.cat([state, action, peg_type, hole_type], dim = 1))
            # contact_tl = self.contact_class(torch.cat([pose_next, peg_type, hole_type], dim = 1))
            # # frc_enc_tl = self.dyn_frc(torch.cat([state, pose_next, contact_next, action, peg_type, hole_type], dim = 1))
            # # vel_tl = self.dyn_vel(torch.cat([state, pose_next, frc_enc_next, contact_next, action, peg_type, hole_type], dim = 1))
            # vel_tl = self.dyn_vel(torch.cat([state, pose_next, contact_next, action, peg_type, hole_type], dim = 1))
            # frc_enc_tl = self.dyn_frc(torch.cat([state, pose_next, vel_next, contact_next, action, peg_type, hole_type], dim = 1))


            # contact_logits = self.contact_class(torch.cat([pose_pred, peg_type, hole_type], dim = 1))
            # contact_probs = torch.sigmoid(contact_logits)
            # contact_pred = torch.where(contact_probs > 0.5, torch.ones_like(contact_probs), torch.zeros_like(contact_probs))

            # # frc_pred = self.dyn_frc(torch.cat([state, pose_pred, contact_pred, action, peg_type, hole_type], dim = 1))
            # # vel_pred = self.dyn_vel(torch.cat([state, pose_pred, frc_pred, contact_pred, action, peg_type, hole_type], dim = 1))

            # vel_pred = self.dyn_vel(torch.cat([state, pose_pred, contact_pred, action, peg_type, hole_type], dim = 1))
            # frc_pred = self.dyn_frc(torch.cat([state, pose_pred, vel_pred, contact_pred, action, peg_type, hole_type], dim = 1))

            if idx >= self.offset:
                frc_pred_list.append(frc_pred)
                pose_pred_list.append(pose_pred)
                vel_pred_list.append(vel_pred)
                contact_pred_list.append(contact_logits.squeeze())
                # frc_tl_list.append(frc_enc_tl)
                # vel_tl_list.append(vel_tl)
                # contact_tl_list.append(contact_tl.squeeze())

        return {
            "pose_pred": pose_pred_list,
            "vel_pred": vel_pred_list,
            "frc_enc_pred": frc_pred_list,
            "contact_pred": contact_pred_list,
            # "vel_tl": vel_tl_list,
            # "frc_enc_tl": frc_tl_list,
            # "contact_tl": contact_tl_list,
            "frc_enc": torch.cat(frc_enc_list, dim = 1),
        }

class DynamicswForce(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, z_dim, num_options, offset, train_mode = 0, device = None, curriculum = None):
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
        self.proprio_size = proprio_size[0]
        self.z_dim = z_dim
        self.force_size = force_size
        self.offset = offset
        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16
        self.np = 20

        self.train_mode = train_mode # 0 for training models, # 1 for training variance # 2 for evaluating particle filter

        self.normalization = simple_normalization #

        self.state_size = (self.frc_enc_size + self.proprio_size + self.contact_size) 

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + 2 * self.num_options, 1, 3, device = self.device)    

        self.idyn = ResNetFCN(folder + "_inverse_dynamics", 2 * self.state_size + 2 * self.num_options, self.z_dim, 4, device = self.device)
        self.direction_model = FCN(folder + "_direction_est", self.z_dim, self.action_dim, 3, device = self.device)
        self.magnitude_model = FCN(folder + "_magnitude_est", self.z_dim, 1, 3, device = self.device)

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.state_size - self.contact_size, 5, device = self.device)

        self.dyn_noise = Params(folder + "_dynamics_noise", (self.state_size - self.contact_size), device = self.device) 
        self.idyn_noise = Params(folder + "_inv_dynamics_noise", (self.action_dim), device = self.device) 

        self.model_list.append(self.idyn)
        self.model_list.append(self.dyn)
        self.model_list.append(self.direction_model)
        self.model_list.append(self.magnitude_model)

        self.model_list.append(self.frc_enc)
        self.model_list.append(self.contact_class)

        self.model_list.append(self.dyn_noise)
        self.model_list.append(self.idyn_noise)

        self.calc_prob = nn.Softmax(dim=1)

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

        if self.train_mode == 0:
            action_mag_list = []
            action_dir_list = []
            frc_enc_list = []
            contact_list = []
            proprio_pred_list = []
            frc_enc_pred_list = []
        elif self.train_mode == 1:
            action_params_list = []
            frc_enc_params_list = []
            proprio_params_list = []
            frc_enc_list = []
        else:
            frc_pred_list = []
            frc_enc_pred_list = []
            frc_enc_list = []
            prop_pred_list = []
            proprio_pred_list = []

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

            if self.train_mode == 0:
                contact_logits = self.contact_class(torch.cat([proprio, peg_type, hole_type], dim = 1))
                
                frc_enc = self.frc_enc(force)
                frc_enc_list.append(frc_enc.unsqueeze(1))

                frc_enc_next = self.frc_enc(force_next)

                if idx == steps - 1:
                    frc_enc_list.append(frc_enc_next.unsqueeze(1))

                state = torch.cat([proprio, frc_enc, contact], dim =1)
                state_next = torch.cat([proprio_next, frc_enc_next, contact_next], dim = 1)

                action_dir, action_mag = self.get_action_est(state, state_next, peg_type, hole_type)

                proprio_pred, frc_enc_pred = self.get_state_pred(state, action, peg_type, hole_type)

                if idx >= self.offset:
                    action_mag_list.append(action_mag.squeeze())
                    action_dir_list.append(action_dir)
                    contact_list.append(contact_logits.squeeze())
                    proprio_pred_list.append(proprio_pred)
                    frc_enc_pred_list.append(frc_enc_pred)

            elif self.train_mode == 1:

                frc_enc = self.frc_enc(force)
                frc_enc_list.append(frc_enc.clone().detach().unsqueeze(1))
                frc_enc_next = self.frc_enc(force_next)
                if idx == steps - 1:
                    frc_enc_list.append(frc_enc_next.clone().detach().unsqueeze(1))

                state = torch.cat([proprio, frc_enc, contact], dim =1)
                state_next = torch.cat([proprio_next, frc_enc_next, contact_next], dim = 1)

                action_dir, action_mag = self.get_action_est(state, state_next, peg_type, hole_type)

                action_pred = action_dir * action_mag.squeeze().unsqueeze(1).repeat(1, action_dir.size(1))

                proprio_pred, frc_enc_pred = self.get_state_pred(state, action, peg_type, hole_type)

                if idx >= self.offset:
                    action_params_list.append((action_pred.clone().detach(),\
                     torch.abs(self.idyn_noise()).unsqueeze(0).repeat(action_pred.size(0), 1)))
                    proprio_params_list.append((proprio_pred.clone().detach(),\
                     torch.abs(self.dyn_noise())[:self.proprio_size].unsqueeze(0).repeat(proprio_pred.size(0), 1)))
                    frc_enc_params_list.append((frc_enc_pred.clone().detach(),\
                     torch.abs(self.dyn_noise())[self.proprio_size:].unsqueeze(0).repeat(frc_enc_pred.size(0), 1)))

            else:
                frc_enc = self.frc_enc(force)
                frc_enc_list.append(frc_enc.clone().detach().unsqueeze(1))
                frc_enc_next = self.frc_enc(force_next)
                if idx == steps - 1:
                    frc_enc_list.append(frc_enc_next.clone().detach().unsqueeze(1))

                state = torch.cat([proprio, frc_enc, contact], dim =1)

                prop_pred, frc_pred = self.get_state_pred(state, action, peg_type, hole_type)
                contact_logits = self.contact_class(torch.cat([prop_pred, peg_type, hole_type], dim = 1))
                contact_probs = torch.sigmoid(contact_logits)
                contact_pred = torch.where(contact_probs > 0.5, torch.ones_like(contact_probs), torch.zeros_like(contact_probs)).unsqueeze(1).repeat(1,self.np,1)

                next_state_mean = torch.cat([prop_pred, frc_pred], dim=1).unsqueeze(1).repeat(1,self.np,1)
                next_state_var = torch.abs(self.dyn_noise()).unsqueeze(0).unsqueeze(1).repeat(next_state_mean.size(0), next_state_mean.size(1), 1)

                next_state_woutcontact = sample_gaussian(next_state_mean, next_state_var, device = self.device)

                state_next_exp = torch.cat([next_state_woutcontact, contact_pred], dim = 2).view(state.size(0) * self.np, state.size(1))

                state_exp = state.unsqueeze(1).repeat(1,self.np,1).view(state.size(0) * self.np, state.size(1))
                peg_type_exp = peg_type.unsqueeze(1).repeat(1,self.np,1).view(peg_type.size(0) * self.np, peg_type.size(1))
                hole_type_exp = hole_type.unsqueeze(1).repeat(1,self.np,1).view(peg_type.size(0) * self.np, hole_type.size(1))

                action_dir, action_mag = self.get_action_est(state_exp, state_next_exp, peg_type_exp, hole_type_exp)

                action_pred = (action_dir * action_mag.repeat(1, action_dir.size(1))).view(action.size(0), self.np, action.size(1))

                action_mean = action.unsqueeze(1).repeat(1, self.np, 1)

                action_var = torch.abs(self.idyn_noise()).unsqueeze(0).unsqueeze(1).repeat(action_mean.size(0), self.np, 1)

                action_logits = log_normal(action_pred, action_mean, action_var) # size batch_size, num particles

                action_prob = self.calc_prob(action_logits)

                proprio_pred = (state_next_exp.view(state.size(0), self.np, state.size(1))[:,:,:self.proprio_size] *\
                 action_prob.unsqueeze(2).repeat(1,1,self.proprio_size)).sum(1)

                frc_enc_pred = (state_next_exp.view(state.size(0), self.np, state.size(1))[:,:,self.proprio_size:-1] *\
                 action_prob.unsqueeze(2).repeat(1,1,self.frc_enc_size)).sum(1)


                if idx >= self.offset:
                    frc_pred_list.append(frc_pred.clone().detach())
                    frc_enc_pred_list.append(frc_enc_pred.clone().detach())
                    prop_pred_list.append(prop_pred.clone().detach())
                    proprio_pred_list.append(proprio_pred.clone().detach())

        if self.train_mode == 0:
            return {
                "action_mag": action_mag_list,
                "action_dir": action_dir_list,
                "frc_enc": torch.cat(frc_enc_list, dim = 1),
                "contact_class": contact_list,
                "frc_enc_pred": frc_enc_pred_list,
                "proprio_pred": proprio_pred_list,
            }
        elif self.train_mode == 1:
            return {
                "action_params": action_params_list,
                "frc_enc": torch.cat(frc_enc_list, dim = 1),
                "frc_enc_params": frc_enc_params_list,
                "proprio_params": proprio_params_list,
            }
        else:
            return {
                "prop_pred": prop_pred_list,
                "proprio_pred": proprio_pred_list,
                "frc_pred": frc_pred_list,
                "frc_enc_pred": frc_enc_pred_list,
                "frc_enc": torch.cat(frc_enc_list, dim = 1),
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

class Options_UncertaintyQuantifier(Proto_Macromodel):
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
        self.uncertainty_size = 1
        self.frc_enc_size = 16

        self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + 2 * self.num_options + self.uncertainty_size

        self.uncertainty_lstm = LSTMCell(folder + "_uncertainty_lstm", self.state_size, self.z_dim, device = self.device)
        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)
        self.uncertainty_est = FCN(folder + "_uncertainty_est", z_dim, 1, 3, device = self.device)

        self.model_list.append(self.uncertainty_lstm)
        self.model_list.append(self.pre_lstm)
        self.model_list.append(self.uncertainty_est)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_uncertainty(self, proprio, frc_enc, contact, action, peg_type, hole_type, uncertainty, h = None, c = None):
        prestate = torch.cat([proprio, frc_enc, contact.unsqueeze(1), action, peg_type, hole_type, uncertainty], dim = 1)
        state = self.pre_lstm(prestate)

        if h is None or c is None:
            h_pred, c_pred = self.uncertainty_lstm(state)
        else:
            h_pred, c_pred = self.uncertainty_lstm(state, h, c) 

        uncertainty_logits = self.uncertainty_est(h_pred)

        return uncertainty_logits, h_pred, c_pred

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        frc_encs = input_dict["frc_enc"].to(self.device)
        hole_probs = input_dict["hole_probs"].to(self.device)

        # print("Forces size: ", forces.size())
        contacts = input_dict["contact"].to(self.device)
        peg_types = input_dict["peg_type"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        uncertainty_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        uncertainty_probs = torch.ones_like(contacts[:,0]).unsqueeze(1) / 2

        for idx in range(steps):
            action = actions[:,idx]
            proprio = proprios[:,idx]
            frc_enc = frc_encs[:,idx]
            contact = contacts[:,idx]
            peg_type = peg_types[:,idx]
            hole_prob = hole_probs[:, idx]

            if idx == 0:
                uncertainty_logits, h, c = self.get_uncertainty(proprio, frc_enc, contact, action, peg_type, hole_prob, uncertainty_probs)
                uncertainty_probs = torch.sigmoid(uncertainty_logits).squeeze().unsqueeze(1)
            # stops gradient after a certain number of steps
            # elif idx != 0 and idx % 8 == 0:
            #     h_clone = h.detach()
            #     c_clone = c.detach()
            #     uncertainty_logits, h, c = self.get_uncertainty(proprio, frc_enc, contact, action, peg_type, hole_prob, uncertainty_probs, h_clone, c_clone)
            #     uncertainty_probs = torch.sigmoid(uncertainty_logits).squeeze().unsqueeze(1)
            else:
                uncertainty_logits, h, c = self.get_uncertainty(proprio, frc_enc, contact, action, peg_type, hole_prob, uncertainty_probs, h, c)
                uncertainty_probs = torch.sigmoid(uncertainty_logits).squeeze().unsqueeze(1)

            if idx >= self.offset:
                uncertainty_list.append(uncertainty_logits.squeeze())

        return {
            'uncertainty_est': uncertainty_list,
        }
class Particle_Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, z_dim, num_options, offset, heteroscedastic_noise = False, num_particles = 20, device = None, curriculum = None):
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
        self.num_particles = num_particles
        self.num_options = num_options
        self.frc_enc_size = 16
        self.offset = offset

        self.num_options = num_options
        self.contact_size = 1
        self.frc_enc_size = 16
        self.proprio_enc_size = 16
        self.z_dim = z_dim
        self.heteroscedastic_noise = heteroscedastic_noise

        self.state_size = self.proprio_enc_size + self.frc_enc_size

        if self.heteroscedastic_noise:
            self.noise_model = ResNetFCN(folder + "_noise_model", self.state_size + 2 * self.num_options + self.action_dim + self.contact_size,\
             self.state_size, 2, device = self.device)
            self.model_list.append(self.noise_model)
        else:
            self.state_noise = Params(folder + "_state_noise", (self.state_size), device = self.device) 
            self.model_list.append(self.state_noise)

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.proprio_enc = ResNetFCN(folder + "_proprio_enc", self.proprio_size, self.proprio_enc_size, 2, device = self.device)

        self.dynamics = ResNetFCN(folder + "_dymamics_resnet", 2 * self.state_size + 2 * self.num_options + self.action_dim + self.contact_size, self.state_size, 4, device = self.device)
        self.dynamics_flows = PlanarFlow(folder + "_dynamics_flows", self.state_size, 30, device = self.device)

        self.contact_pred = FCN(folder + "_contact_pred", self.state_size + 2 * self.num_options + self.action_dim + self.contact_size, 1, 3, device = self.device)

        self.model_list.append(self.frc_enc)
        self.model_list.append(self.proprio_enc)

        self.model_list.append(self.dynamics)
        self.model_list.append(self.dynamics_flows)

        self.model_list.append(self.contact_pred)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, proprio, force, action, contact, peg_type, hole_type):

        pre_state = torch.cat([proprio, force, contact.unsqueeze(1), action, peg_type, hole_type], dim = 1).unsqueeze(1).repeat(1, self.num_particles, 1)

        contact_logits = self.contact_pred(torch.cat([proprio, force, contact.unsqueeze(1), action, peg_type, hole_type], dim = 1))

        if self.heteroscedastic_noise:
            state_variance = self.noise_model(pre_state.view(pre_state.size(0) * pre_state.size(1), pre_state.size(2))).view(pre_state.size(0), pre_state.size(1), self.state_size)
        else:
            state_variance = self.state_noise.params.unsqueeze(0).unsqueeze(1).repeat(pre_state.size(0), pre_state.size(1), 1)

        state_mean = torch.zeros_like(state_variance)

        noise_samples = sample_gaussian(state_mean, state_variance, self.device)

        state = torch.cat([pre_state, noise_samples], dim = 2).view(pre_state.size(0) * self.num_particles, pre_state.size(2) + noise_samples.size(2))

        pre_particles = self.dynamics(state)#.view(pre_state.size(0), self.num_particles, self.state_size)
        pred_particles = self.dynamics_flows(pre_particles).view(pre_state.size(0), self.num_particles, self.state_size)

        pred_mean = pred_particles.mean(1).squeeze() + torch.cat([proprio, force], dim = 1)
        pred_var = pred_particles.var(1).squeeze()

        return (pred_mean[:, :self.proprio_enc_size], pred_var[:, :self.proprio_enc_size]), \
        (pred_mean[:,self.proprio_enc_size:], pred_var[:,self.proprio_enc_size:]), \
        contact_logits.squeeze()

    def forward(self, input_dict):

        proprios = input_dict["proprio"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)

        contacts = input_dict["contact"].to(self.device)
        epoch =  int(input_dict["epoch"].detach().item())

        frc_enc_list = []
        proprio_enc_list = []
        force_params_list = []
        proprio_params_list = []
        contact_params_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            force = forces[:, idx]
            proprio = proprios[:,idx]
            force_next = forces[:, idx + 1]
            proprio_next = proprios[:, idx + 1]
            action = actions[:,idx]
            contact = contacts[:,idx]
            peg_type = peg_types[:,idx]
            hole_type = hole_types[:,idx]

            frc_enc = self.frc_enc(force)
            frc_enc_list.append(frc_enc.unsqueeze(1))
            proprio_enc = self.proprio_enc(proprio)
            proprio_enc_list.append(proprio_enc.unsqueeze(1))

            if idx == steps - 1:
                frc_enc_next = self.frc_enc(force_next)
                frc_enc_list.append(frc_enc_next.unsqueeze(1))
                proprio_enc_next = self.proprio_enc(proprio_next)
                proprio_enc_list.append(proprio_enc_next.unsqueeze(1))

            proprio_params, force_params, contact_params = self.get_pred(proprio_enc, frc_enc, action, contact, peg_type, hole_type)

            if idx >= self.offset:
                proprio_params_list.append(proprio_params)
                force_params_list.append(force_params)
                contact_params_list.append(contact_params)

        return {
            'proprio_enc_array': torch.cat(proprio_enc_list, dim = 1),
            'frc_enc_array': torch.cat(frc_enc_list, dim = 1),
            'proprio_logprob': proprio_params_list,
            'force_logprob': force_params_list,
            'contact_logprob': contact_params_list,
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

        self.state_size = self.frc_enc_size + self.proprio_size + self.contact_size + self.action_dim + self.num_options + 1

        self.fit_lstm = LSTMCell(folder + "_fit_lstm", self.state_size, self.z_dim, device = self.device) 

        self.pre_lstm = FCN(folder + "_pre_lstm", self.state_size, self.state_size, 3, device = self.device)

        self.fit_class = FCN(folder + "_fit_class", z_dim, 1, 3, device = self.device)  

        self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
         (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.model_list = [self.fit_lstm, self.fit_class, self.frc_enc, self.pre_lstm]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_fit_class(self, proprio, force, contact, action, peg_type, fit_probs, h = None, c = None):
        # print("Force size: ", force.size())
        frc_enc = self.frc_enc(force)

        # print("Proprio size: ", proprio.size())
        # print("Force size: ", frc_enc.size())
        # print("contact size: ", contact.size())
        # print("action size: ", action.size())
        # print("peg type size: ", peg_type.size())
        prestate = torch.cat([proprio, frc_enc, contact.unsqueeze(1), action, peg_type, fit_probs], dim = 1)
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

        fit_probs = torch.ones_like(contacts[:,0]).squeeze().unsqueeze(1) / 2

        # print(fit_probs.size())

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
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type, fit_probs)
                fit_probs = torch.sigmoid(fit_logits).squeeze().unsqueeze(1)

            # stops gradient after a certain number of steps
            elif idx != 0 and idx % 8 == 0:
                h_clone = h.detach()
                c_clone = c.detach()
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type, fit_probs, h_clone, c_clone)
                fit_probs = torch.sigmoid(fit_logits).squeeze().unsqueeze(1)
            else:
                fit_logits, h, c = self.get_fit_class(proprio, force, contact, action, peg_type, fit_probs, h, c)
                fit_probs = torch.sigmoid(fit_logits).squeeze().unsqueeze(1)

            if idx >= self.offset:
                fit_list.append(fit_logits)

        return {
            'fit_class': fit_list,
        }
        
class EEFRC_Prob_Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, joint_size, action_dim, device = None, curriculum = None):
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
        self.force_size = force_size[0]
        self.joint_size = joint_size[0]

        self.ee_nc = ResNetFCN(folder + "_ee_resnet_nc", self.action_dim + self.proprio_size + 2 * self.joint_size, 2 * self.proprio_size, 5, device = self.device)

        self.ee_c = ResNetFCN(folder + "_ee_resnet_c", self.action_dim + self.proprio_size + 2 * self.joint_size + self.force_size, 2 * self.proprio_size, 5, device = self.device)

        self.frc_nc = ResNetFCN(folder + "_frc_resnet_nc", self.action_dim + self.proprio_size + 2 * self.joint_size, 2 * self.force_size, 5, device = self.device) 

        self.frc_c = ResNetFCN(folder + "_frc_resnet_c", self.action_dim + self.proprio_size + 2 * self.joint_size + self.force_size, 2 * self.force_size, 5, device = self.device) 

        self.joint_nc = ResNetFCN(folder + "_joint_resnet_nc", self.action_dim + self.proprio_size + 2 * self.joint_size,  2 * 2 * self.joint_size, 5, device = self.device) 

        self.joint_c = ResNetFCN(folder + "_joint_resnet_c", self.action_dim + self.proprio_size + self.force_size + 2 * self.joint_size, 2 * 2 * self.joint_size, 5, device = self.device) 

        self.contact_class = ResNetFCN(folder + "_contact_class", self.proprio_size + self.force_size, 1, 3, device = self.device)    

        self.model_list = [self.ee_c, self.ee_nc, self.frc_nc, self.frc_c, self.joint_nc, self.joint_c, self.contact_class]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, proprio, joint_proprio, force, action, contact = None, deter_bool=True):

        if contact is None:
            cont = torch.sigmoid(self.contact_class(torch.cat([proprio, force], dim = 1)))
            contact = torch.where(cont > 0.5, torch.ones_like(cont), torch.zeros_like(cont)).squeeze()
        
        proprio_params = self.ee_nc(torch.cat([proprio, joint_proprio, action], dim = 1)) +\
         contact.unsqueeze(1).repeat(1, 2 * proprio.size(1)) * self.ee_c(torch.cat([proprio, joint_proprio, force, action], dim = 1))

        proprio_delta, proprio_var = gaussian_parameters(proprio_params)

        proprio_mu = proprio_delta + proprio

        force_params = self.frc_nc(torch.cat([proprio, joint_proprio, action], dim = 1)) +\
         contact.unsqueeze(1).repeat(1, 2 * force.size(1)) * self.frc_c(torch.cat([proprio, joint_proprio, force, action], dim = 1))

        force_delta, force_var = gaussian_parameters(force_params)

        force_mu = force_delta + force

        joint_proprio_params = self.joint_nc(torch.cat([proprio, joint_proprio, action], dim = 1)) +\
         contact.unsqueeze(1).repeat(1, 2 * joint_proprio.size(1)) * self.joint_c(torch.cat([proprio, joint_proprio, force, action], dim = 1))

        joint_proprio_delta, joint_proprio_var = gaussian_parameters(joint_proprio_params)

        joint_proprio_mu = joint_proprio_delta + joint_proprio

        # print("Checking sizes")
        # print("EE NC size", self.ee_nc(torch.cat([proprio, joint_proprio, action], dim = 1)).size())
        # print("EE C size", self.ee_c(torch.cat([proprio, joint_proprio, force, action], dim = 1)).size())
        # print("FRC NC size", self.frc_nc(torch.cat([proprio, joint_proprio, action], dim = 1)).size())
        # print("FRC C size", self.frc_c(torch.cat([proprio, joint_proprio, force, action], dim = 1)).size())
        # print("Joint NC size", self.joint_nc(torch.cat([proprio, joint_proprio, action], dim = 1)).size() )
        # print("Joint C size", self.joint_c(torch.cat([proprio, joint_proprio, force, action], dim = 1)).size())
        # print("Contact size", self.contact_class(torch.cat([proprio, force], dim = 1)).size())

        if deter_bool:
            return proprio_mu, joint_proprio_mu, force_mu
        else:
            return sample_gaussian(proprio_mu, proprio_var, self.device),\
            sample_gaussian(joint_proprio_mu, joint_proprio_var, self.device),\
            sample_gaussian(force_mu, force_var, self.device)

    def forward(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        
        joint_poses = input_dict["joint_pos"].to(self.device)
        joint_vels = input_dict["joint_vel"].to(self.device)
        joint_proprios = torch.cat([joint_poses, joint_vels], dim = 2)

        actions = input_dict["action"].to(self.device)
        
        forces = input_dict["force"].to(self.device)

        contacts = input_dict["contact"].to(self.device)
        
        epoch =  int(input_dict["epoch"].detach().item())

        prop_list = []
        joint_pose_list = []
        joint_vel_list = []
        force_list = []
        contact_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        force = forces[:,0].clone()
        proprio = proprios[:,0].clone()
        joint_proprio = joint_proprios[:,0].clone()

        for idx in range(steps):
            action = actions[:,idx]
            contact = contacts[:,idx]

            if idx == steps - 1:
                proprio, joint_proprio, force = self.get_pred(proprio, joint_proprio, force, action, contact, False)
            else:
                proprio, joint_proprio, force = self.get_pred(proprio, joint_proprio, force, action, contact)
            
            cont_class = self.contact_class(torch.cat([proprio, force], dim = 1))

            joint_pose_list.append(joint_proprio[:, :self.joint_size])
            joint_vel_list.append(joint_proprio[:,self.joint_size:])
            prop_list.append(proprio)
            force_list.append(force)
            contact_list.append(cont_class)

        return {
            'contact': contact_list,
            'prop_pred': prop_list,
            'joint_pos_pred': joint_pose_list,
            'joint_vel_pred': joint_vel_list,
            'force_pred': force_list,
        }

    def trans(self, input_dict):
        proprio = input_dict["proprio"].to(self.device)
        force = input_dict["force"].to(self.device)
        joint_pos = input_dict["joint_pos"].to(self.device).squeeze()
        joint_vel = input_dict["joint_vel"].to(self.device).squeeze()
        joint_proprio = torch.cat([joint_pos, joint_vel], dim = 0)
        actions = (input_dict["action_sequence"]).to(self.device)

        steps = actions.size(0)

        for idx in range(steps):
            action = actions[idx].unsqueeze(0)
            proprio, joint_proprio, force = self.get_pred(proprio.squeeze().unsqueeze(0),\
            joint_proprio.squeeze().unsqueeze(0), force.squeeze().unsqueeze(0), action)           

        return {
            'proprio': proprio.squeeze().unsqueeze(0),
            'joint_pos': joint_proprio.squeeze().unsqueeze(0)[:, :self.joint_size],
            'joint_vel': joint_proprio.squeeze().unsqueeze(0)[:, self.joint_size:],
            'force': force.squeeze().unsqueeze(0),
        }

class LSTM_Multimodal(Proto_Macromodel):
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

        self.trans_lstm = LSTM(folder + "_trans_net_lstm", z_dim + self.action_dim, z_dim, device = self.device)

        self.trans_module = FCN(folder + "_trans_module", z_dim, z_dim, 3, device = self.device)

        self.model_list = [self.fusion_module, self.proprio_enc, self.proprio_dec, self.img_enc, self.img_dec, self.trans_lstm, self.trans_module]

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_z(self, image, proprio):
        img_emb = self.img_enc(image)
        proprio_emb = self.proprio_enc(proprio)
        return self.fusion_module(torch.cat([img_emb, proprio_emb], dim = 1))

    def get_z_pred(self, z, action, h = None, c = None):
        if h is None or c is None:
            h_pred, c_pred = self.trans_lstm(torch.cat([z, action], dim = 1))
        else:
            h_pred, c_pred = self.trans_lstm(torch.cat([z, action], dim = 1), h, c) 

        z_pred = self.trans_module(h_pred)

        return z_pred, h_pred, c_pred

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
                z_pred, h_pred, c_pred = self.get_z_pred(z_pred, action)
                z_pred_list.append(z_pred)
            elif idx != steps - 1:
                z_pred, h_pred, c_pred = self.get_z_pred(z_pred, action, h_pred, c_pred)
                z_pred_list.append(z_pred)

            img = self.get_image(z)
            img_list.append(img)
            prop = self.get_proprio(z)
            prop_list.append(prop)



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

            if idx == 0:
                z, h, c = self.get_z_pred(z.squeeze().unsqueeze(0), action)
            else:
                z, h, c = self.get_z_pred(z.squeeze().unsqueeze(0), action, h.squeeze().unsqueeze(0), c.squeeze().unsqueeze(0))

        return {
            'latent_state': z.squeeze().unsqueeze(0),
        }
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
class Latent_Dynamics(Proto_Macromodel):
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
        self.frc_enc_size = 8 * 3 * 2
        self.np = 20
        self.use_fft = use_fft

        self.normalization = simple_normalization #nn.Softmax(dim=1)
        self.prob_calc = nn.Softmax(dim=1)
        self.idxs = [12, 24, 52, 100, 101, 102, 103, 104]

        if self.use_fft:
            self.frc_enc = CONV2DN(folder + "_fft_enc", (self.force_size[0], 126, 2), (8, 3, 2), False, True, 1, device = self.device)
        else:
            self.frc_enc = CONV1DN(folder + "_frc_enc", (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device)

        self.ang_enc = ResNetFCN(folder + "_ang_enc", self.pose_size, 2 * self.pose_size, 3, device = self.device)
        self.pos_enc = ResNetFCN(folder + "_pos_enc", self.pose_size, 2 * self.pose_size, 3, device = self.device)
        self.joint_enc = ResNetFCN(folder + "_ang_enc", 2 * self.joint_pose_size, 4 * self.joint_pose_size, 3, device = self.device)

        self.state_size = (self.frc_enc_size + 2 * self.pose_size + 2 * self.pose_size + 4 * self.joint_pose_size + self.contact_size) 

        self.dyn = ResNetFCN(folder + "_dynamics", self.state_size + self.action_dim + 2 * self.num_options, self.state_size + 4, 10, device = self.device)

        self.pos_dec = ResNetFCN(folder + "_pos_dec", self.state, )

        self.model_list.append(self.dyn)
        self.model_list.append(self.frc_enc)
        self.model_list.append(self.ang_enc)
        self.model_list.append(self.pos_enc)
        self.model_list.append(self.joint_enc)

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
        actions = input_dict["action"].to(self.device)
        proprios = input_dict["proprio"].to(self.device)
        joint_poses = input_dict["joint_pos"].to(self.device)
        joint_vels = input_dict["joint_vel"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        peg_types = input_dict["peg_type"].to(self.device)
        hole_types = input_dict["hole_type"].to(self.device)
        contacts = input_dict["contact"].to(self.device).unsqueeze(2)
        epoch =  int(input_dict["epoch"].detach().item())

        pos_enc_diff_m_list = []
        pos_enc_diff_d_list = []
        pos_enc_diff_dir_list = []
        pos_enc_diff_mag_list = []

        ang_enc_diff_m_list = []
        ang_enc_diff_d_list = []
        ang_enc_diff_dir_list = []
        ang_enc_diff_mag_list = []

        joint_enc_diff_m_list = []
        joint_enc_diff_d_list = []
        joint_enc_diff_dir_list = []
        joint_enc_diff_mag_list = []

        frc_enc_diff_m_list = []
        frc_enc_diff_d_list = []
        frc_enc_diff_dir_list = []
        frc_enc_diff_mag_list = []

        contact_pred_list = []

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
        pos_enc_pred = self.pos_enc(torch.cat([pos, vel], dim = 1))
        ang_enc_pred = self.ang_enc(torch.cat([ang, ang_vel], dim =1))

        joint_pos = joint_poses[:,0]
        joint_vel = joint_vels[:,0]

        joint_enc_pred = self.joint_enc(torch.cat([joint_pos, joint_vel], dim = 1))
        
        force = forces[:,0]
        frc_enc_pred = self.get_frc(force)

        contact = contacts[:,0]

        state = torch.cat([pos_enc_pred, ang_enc_pred, joint_enc_pred, frc_enc_pred, contact], dim =1)

        for idx in range(steps):
            action = actions[:,idx]

            ### encoding proprioception
            proprio = proprios[:,idx]
            pos = proprio[:,:3]
            ang = proprio[:,3:6]
            vel = proprio[:,6:9]
            ang_vel = proprio[:,9:12]
            pos_enc = self.pos_enc(torch.cat([pos, vel], dim = 1))
            ang_enc = self.ang_enc(torch.cat([ang, ang_vel], dim =1))

            proprio_next = proprios[:,idx + 1]
            pos_next = proprio_next[:,:3]
            ang_next = proprio_next[:,3:6]
            vel_next = proprio_next[:,6:9]
            ang_vel_next = proprio_next[:,9:12]
            pos_enc_next = self.pos_enc(torch.cat([pos_next, vel_next], dim = 1))
            ang_enc_next = self.ang_enc(torch.cat([ang_next, ang_vel_next], dim =1))

            pos_enc_diff = pos_enc_next - pos_enc
            ang_enc_diff = ang_enc_next - ang_enc

            pos_enc_diff_m_list.append(pos_enc_diff.norm(2,dim=1).unsqueeze(1))
            pos_enc_diff_d_list.append((pos_enc_diff / pos_enc_diff.norm(2,dim=1).unsqueeze(1).repeat_interleave(pos_enc.size(1), 1)).unsqueeze(1))

            ang_enc_diff_m_list.append(ang_enc_diff.norm(2,dim=1).unsqueeze(1))
            ang_enc_diff_d_list.append((ang_enc_diff / ang_enc_diff.norm(2,dim=1).unsqueeze(1).repeat_interleave(ang_enc.size(1), 1)).unsqueeze(1))

            ### encoding joint state
            joint_pos = joint_poses[:,idx]
            joint_vel = joint_vels[:,idx]

            joint_enc = self.joint_enc(torch.cat([joint_pos, joint_vel], dim = 1))

            joint_pos_next = joint_poses[:,idx+1]
            joint_vel_next = joint_vels[:,idx+1]

            joint_enc_next = self.joint_enc(torch.cat([joint_pos_next, joint_vel_next], dim = 1))

            joint_enc_diff = joint_enc_next - joint_enc

            joint_enc_diff_m_list.append(joint_enc_diff.norm(2,dim=1).unsqueeze(1))
            joint_enc_diff_d_list.append((joint_enc_diff / joint_enc_diff.norm(2,dim=1).unsqueeze(1).repeat_interleave(joint_enc.size(1), 1)).unsqueeze(1))

            ### encoding force
            force = forces[:,idx]
            frc_enc = self.get_frc(force)

            force_next = forces[:,idx+1]
            frc_enc_next = self.get_frc(force_next)

            frc_enc_diff = frc_enc_next - frc_enc

            frc_enc_diff_m_list.append(frc_enc_diff.norm(2,dim=1).unsqueeze(1))
            frc_enc_diff_d_list.append((frc_enc_diff / frc_enc_diff.norm(2,dim=1).unsqueeze(1).repeat_interleave(frc_enc.size(1), 1)).unsqueeze(1))

            #### running through dynamics model
            next_state = self.dyn(torch.cat([state, action, peg_type, hole_type], dim = 1))

            pos_enc_diff_est_dir = self.normalization(next_state[:,:self.idxs[0]])
            ang_enc_diff_est_dir = self.normalization(next_state[:,self.idxs[0]:self.idxs[1]])
            joint_enc_diff_est_dir = self.normalization(next_state[:,self.idxs[1]:self.idxs[2]])
            frc_enc_diff_est_dir = self.normalization(next_state[:,self.idxs[2]:self.idxs[3]])

            contact_logits = next_state[:, self.idxs[3]]

            pos_enc_diff_est_mag = next_state[:,self.idxs[4]]
            ang_enc_diff_est_mag = next_state[:,self.idxs[5]]
            joint_enc_diff_est_mag = next_state[:,self.idxs[6]]
            frc_enc_diff_est_mag = next_state[:,self.idxs[7]]

            contact_probs = torch.sigmoid(contact_logits)
            contact_pred = torch.where(contact_probs > 0.5, torch.ones_like(contact_probs), torch.zeros_like(contact_probs))

            pos_enc_pred = pos_enc_diff_est_dir * pos_enc_diff_est_mag.unsqueeze(1).repeat_interleave(pos_enc_diff_est_dir.size(1), 1) + pos_enc_pred
            ang_enc_pred = ang_enc_diff_est_dir * ang_enc_diff_est_mag.unsqueeze(1).repeat_interleave(ang_enc_diff_est_dir.size(1), 1) + ang_enc_pred
            joint_enc_pred = joint_enc_diff_est_dir * joint_enc_diff_est_mag.unsqueeze(1).repeat_interleave(joint_enc_diff_est_dir.size(1), 1) + joint_enc_pred
            frc_enc_pred = frc_enc_diff_est_dir * frc_enc_diff_est_mag.unsqueeze(1).repeat_interleave(frc_enc_diff_est_dir.size(1), 1) + frc_enc_pred

            state = torch.cat([pos_enc_pred, ang_enc_pred, joint_enc_pred, frc_enc_pred, contact_pred.unsqueeze(1)], dim = 1)

            if idx >= self.offset:
                pos_enc_diff_dir_list.append(pos_enc_diff_est_dir)
                pos_enc_diff_mag_list.append(pos_enc_diff_est_mag)
                ang_enc_diff_dir_list.append(ang_enc_diff_est_dir)
                ang_enc_diff_mag_list.append(ang_enc_diff_est_mag)
                joint_enc_diff_dir_list.append(joint_enc_diff_est_dir)
                joint_enc_diff_mag_list.append(joint_enc_diff_est_mag)
                frc_enc_diff_dir_list.append(frc_enc_diff_est_dir)
                frc_enc_diff_mag_list.append(frc_enc_diff_est_mag)

                contact_pred_list.append(contact_logits.squeeze())

        return {
            "pos_enc_diff_m": torch.cat(pos_enc_diff_m_list, dim = 1),
            "pos_enc_diff_d": torch.cat(pos_enc_diff_d_list, dim = 1),
            "ang_enc_diff_m": torch.cat(ang_enc_diff_m_list, dim = 1),
            "ang_enc_diff_d": torch.cat(ang_enc_diff_d_list, dim = 1),
            "joint_enc_diff_m": torch.cat(joint_enc_diff_m_list, dim = 1),
            "joint_enc_diff_d": torch.cat(joint_enc_diff_d_list, dim = 1),
            "frc_enc_diff_m": torch.cat(frc_enc_diff_m_list, dim = 1),
            "frc_enc_diff_d": torch.cat(frc_enc_diff_d_list, dim = 1),
            "pos_enc_diff_mag": pos_enc_diff_mag_list,
            "pos_enc_diff_dir": pos_enc_diff_dir_list,
            "ang_enc_diff_mag": ang_enc_diff_mag_list,
            "ang_enc_diff_dir": ang_enc_diff_dir_list,
            "joint_enc_diff_mag": joint_enc_diff_mag_list,
            "joint_enc_diff_dir": joint_enc_diff_dir_list,
            "frc_enc_diff_mag": frc_enc_diff_mag_list,
            "frc_enc_diff_dir": frc_enc_diff_dir_list,
            "contact_pred": contact_pred_list,
        }