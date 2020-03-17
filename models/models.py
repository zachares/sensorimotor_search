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

def prec_single(noise_matrix): #converting noise parameters to cholesky decomposition form
    chol_dec = noise_matrix.tril()
    # chol_diag = 
    chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] = torch.abs(chol_dec.diag())
    # chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] += chol_diag 
    prec = torch.mm(chol_dec, chol_dec.t())
    # cov = cov / torch.det(cov)      # print(torch.eig(cov))
    return prec
def prec_mult(tril_matrix): 
    tril_matrix[torch.arange(tril_matrix.size(0)), torch.arange(tril_matrix.size(1))] = torch.abs(tril_matrix[torch.arange(tril_matrix.size(0)), torch.arange(tril_matrix.size(1))])
    prec = torch.bmm(tril_matrix.transpose(1,2).transpose(1,0), tril_matrix.transpose(1,2).transpose(1,0).transpose(1,2))
    # cov_det = torch.det(cov).unsqueeze(1).unsqueeze(2).repeat_interleave(tril_matrix.size(0), dim = 1).repeat_interleave(tril_matrix.size(1), dim = 2)
    # cov = cov / cov_det 
    # print(torch.eig(cov))
    return prec

######################################
# Def:ning Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel
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

        self.model_list.append(self.frc_enc01) 
        self.model_list.append(self.state_enc01) 
        self.model_list.append(self.options_transdec01) 
        self.model_list.append(self.options_class01) 

        self.ensemble_list.append((self.frc_enc01, self.state_enc01, self.options_transdec01, self.options_class01)) # self.origin_cov01)) #, 

        # Network 11
        self.frc_enc11 = CONV2DN(save_folder + "_fft_enc11", load_folder + "_fft_enc11",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc11 = ResNetFCN(save_folder + "_state_enc11", load_folder + "_state_enc11",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec11 = Transformer_Decoder(save_folder + "_options_transdec11", load_folder + "_options_transdec11",\
         self.state_size, 6, device = self.device)

        self.options_class11 = ResNetFCN(save_folder + "_options_class11", load_folder + "_options_class11",\
            self.state_size, self.num_options, 3, device = self.device)

        self.model_list.append(self.frc_enc11) 
        self.model_list.append(self.state_enc11) 
        self.model_list.append(self.options_transdec11) 
        self.model_list.append(self.options_class11) 

        self.ensemble_list.append((self.frc_enc11, self.state_enc11, self.options_transdec11, self.options_class11)) #self.origin_cov11)) #, 

        # Network 21
        self.frc_enc21 = CONV2DN(save_folder + "_fft_enc21", load_folder + "_fft_enc21",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc21 = ResNetFCN(save_folder + "_state_enc21", load_folder + "_state_enc21",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.options_transdec21 = Transformer_Decoder(save_folder + "_options_transdec21", load_folder + "_options_transdec21",\
         self.state_size, 6, device = self.device)

        self.options_class21 = ResNetFCN(save_folder + "_options_class21", load_folder + "_options_class21",\
            self.state_size, self.num_options, 3, device = self.device)

        self.model_list.append(self.frc_enc21) 
        self.model_list.append(self.state_enc21) 
        self.model_list.append(self.options_transdec21) 
        self.model_list.append(self.options_class21) 

        self.ensemble_list.append((self.frc_enc21, self.state_enc21, self.options_transdec21, self.options_class21)) # self.origin_cov21)) #, 

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        fft = torch.rfft(force, 2, normalized=False, onesided=True)
        frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        # print("Force size: ", force.size())
        # print(fft.size())
        # frc_enc = self.frc_enc(fft)
        return frc_enc

    def get_data(self, state, forces, model_tuple, batch_size, sequence_size, padding_masks = None, lengths = None):
        frc_enc, state_enc, options_transdec, options_class = model_tuple #origin_cov = model_tuple # 

        frc_encs_unshaped = self.get_frc(forces, frc_enc)
        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))#.contiguous()

        state_encs_unshaped = state_enc(state)
        state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))#.contiguous()

        states = torch.cat([state_encs, frc_encs], dim = 2)#.contiguous()

        # final_state = states[lengths, torch.arange(states.size(1))].unsqueeze(0)

        if padding_masks is None:
            options_logits = options_class(torch.max(options_transdec(states, states), 0)[0])
        else:
            options_logits = options_class(torch.max(options_transdec(states, states, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks), 0)[0])
        # options_logits = options_class(options_transdec(final_state, states, mem_padding_mask = padding_masks).squeeze(0))

        return options_logits

    def get_logits(self, proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks = None, lengths = None):
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

        prestate = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2), actions], dim = 2)
        prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2)))#.contiguous()

        options_logits01 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[0],\
         batch_size, sequence_size, padding_masks, lengths)

        options_logits11 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[1],\
         batch_size, sequence_size, padding_masks, lengths)

        options_logits21 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[2],\
         batch_size, sequence_size, padding_masks, lengths)
        
        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)

        options_logits1 = ol_peg0 * options_logits01 + ol_peg1 * options_logits11 + ol_peg2 * options_logits21

        return options_logits1

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)
        lengths = input_dict["length"].to(self.device).long().squeeze(1)

        options_logits = self.get_logits(proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks, lengths)

        return {
            'options_class': options_logits,
        }

    def process(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        actions = input_dict["action"].to(self.device)[:, :-1]
        peg_type = input_dict["peg_type"].to(self.device)

        proprio_diffs = proprios[:,1:] - proprios[:, :-1]
        contact_diffs = contacts[:,1:] - contacts[:, :-1]
        force_clipped = forces[:,1:]        

        options_logits = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, peg_type)   

        return options_logits

class PosErr_DetectionTransformer(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, min_steps, use_fft = True, cov_type = 0, device = None):
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
        self.extra = 9
        self.noise_mode = "max"

        self.prestate_size = self.proprio_size + self.contact_size + self.action_dim

        self.ensemble_list = []

        self.train_dets = Params(save_folder + "_train_dets", load_folder + "_val_dets", 3, device = self.device)
        self.val_dets = Params(save_folder + "_val_dets", load_folder + "_val_dets", 3, device = self.device)

        self.model_list.append(self.train_dets)
        self.model_list.append(self.val_dets)

        self.val_lr = 0.02
        self.train_lr = 0.001
        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 01
        self.frc_enc01 = CONV2DN(save_folder + "_fft_enc01", load_folder + "_fft_enc01",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc01 = ResNetFCN(save_folder + "_state_enc01", load_folder + "_state_enc01",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.pos_err_transdec01 = Transformer_Decoder(save_folder + "_pos_err_transdec01", load_folder + "_pos_err_transdec01",\
         self.state_size, 6, device = self.device)

        self.pos_err_mean001 = ResNetFCN(save_folder + "_pos_err_mean001", load_folder + "_pos_err_mean001",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet001 = ResNetFCN(save_folder + "_pos_err_prechet001", load_folder + "_pos_err_prechet001",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom001 = Params(save_folder + "_pos_err_prechom001", load_folder + "_pos_err_prechom001", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean101 = ResNetFCN(save_folder + "_pos_err_mean101", load_folder + "_pos_err_mean101",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet101 = ResNetFCN(save_folder + "_pos_err_prechet101", load_folder + "_pos_err_prechet101",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom101 = Params(save_folder + "_pos_err_prechom101", load_folder + "_pos_err_prechom101", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean201 = ResNetFCN(save_folder + "_pos_err_mean201", load_folder + "_pos_err_mean201",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet201 = ResNetFCN(save_folder + "_pos_err_prechet201", load_folder + "_pos_err_prechet201",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom201 = Params(save_folder + "_pos_err_prechom201", load_folder + "_pos_err_prechom201", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc01) 
        self.model_list.append(self.state_enc01) 
        self.model_list.append(self.pos_err_transdec01) 
        self.model_list.append(self.pos_err_mean001)
        self.model_list.append(self.pos_err_prechom001) 
        self.model_list.append(self.pos_err_prechet001) 
        self.model_list.append(self.pos_err_mean101)
        self.model_list.append(self.pos_err_prechom101) 
        self.model_list.append(self.pos_err_prechet101) 
        self.model_list.append(self.pos_err_mean201)
        self.model_list.append(self.pos_err_prechom201) 
        self.model_list.append(self.pos_err_prechet201) 

        self.ensemble_list.append((self.frc_enc01, self.state_enc01, self.pos_err_transdec01,\
         self.pos_err_mean001, self.pos_err_prechom001, self.pos_err_prechet001,\
          self.pos_err_mean101, self.pos_err_prechom101, self.pos_err_prechet101,\
          self.pos_err_mean201, self.pos_err_prechom201, self.pos_err_prechet201))

        #Network 11
        self.frc_enc11 = CONV2DN(save_folder + "_fft_enc11", load_folder + "_fft_enc11",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc11 = ResNetFCN(save_folder + "_state_enc11", load_folder + "_state_enc11",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.pos_err_transdec11 = Transformer_Decoder(save_folder + "_pos_err_transdec11", load_folder + "_pos_err_transdec11",\
         self.state_size, 6, device = self.device)

        self.pos_err_mean011 = ResNetFCN(save_folder + "_pos_err_mean011", load_folder + "_pos_err_mean011",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet011 = ResNetFCN(save_folder + "_pos_err_prechet011", load_folder + "_pos_err_prechet011",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom011 = Params(save_folder + "_pos_err_prechom011", load_folder + "_pos_err_prechom011", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean111 = ResNetFCN(save_folder + "_pos_err_mean111", load_folder + "_pos_err_mean111",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet111 = ResNetFCN(save_folder + "_pos_err_prechet111", load_folder + "_pos_err_prechet111",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom111 = Params(save_folder + "_pos_err_prechom111", load_folder + "_pos_err_prechom111", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean211 = ResNetFCN(save_folder + "_pos_err_mean211", load_folder + "_pos_err_mean211",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet211 = ResNetFCN(save_folder + "_pos_err_prechet211", load_folder + "_pos_err_prechet211",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom211 = Params(save_folder + "_pos_err_prechom211", load_folder + "_pos_err_prechom211", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc11) 
        self.model_list.append(self.state_enc11) 
        self.model_list.append(self.pos_err_transdec11) 
        self.model_list.append(self.pos_err_mean011)
        self.model_list.append(self.pos_err_prechom011) 
        self.model_list.append(self.pos_err_prechet011) 
        self.model_list.append(self.pos_err_mean111)
        self.model_list.append(self.pos_err_prechom111) 
        self.model_list.append(self.pos_err_prechet111) 
        self.model_list.append(self.pos_err_mean211)
        self.model_list.append(self.pos_err_prechom211) 
        self.model_list.append(self.pos_err_prechet211) 

        self.ensemble_list.append((self.frc_enc11, self.state_enc11, self.pos_err_transdec11,\
         self.pos_err_mean011, self.pos_err_prechom011, self.pos_err_prechet011,\
          self.pos_err_mean111, self.pos_err_prechom111, self.pos_err_prechet111,\
          self.pos_err_mean211, self.pos_err_prechom211, self.pos_err_prechet211))

        # Network 21
        self.frc_enc21 = CONV2DN(save_folder + "_fft_enc21", load_folder + "_fft_enc21",\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

        self.state_enc21 = ResNetFCN(save_folder + "_state_enc21", load_folder + "_state_enc21",\
            self.prestate_size, self.encoded_size, 3, device = self.device)

        self.pos_err_transdec21 = Transformer_Decoder(save_folder + "_pos_err_transdec21", load_folder + "_pos_err_transdec21",\
         self.state_size, 6, device = self.device)

        self.pos_err_mean021 = ResNetFCN(save_folder + "_pos_err_mean021", load_folder + "_pos_err_mean021",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet021 = ResNetFCN(save_folder + "_pos_err_prechet021", load_folder + "_pos_err_prechet021",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom021 = Params(save_folder + "_pos_err_prechom021", load_folder + "_pos_err_prechom021", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean121 = ResNetFCN(save_folder + "_pos_err_mean121", load_folder + "_pos_err_mean121",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet121 = ResNetFCN(save_folder + "_pos_err_prechet121", load_folder + "_pos_err_prechet121",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom121 = Params(save_folder + "_pos_err_prechom121", load_folder + "_pos_err_prechom121", (self.pose_size, self.pose_size), device = self.device)

        self.pos_err_mean221 = ResNetFCN(save_folder + "_pos_err_mean221", load_folder + "_pos_err_mean221",\
            self.state_size + self.extra, self.pose_size, 3, device = self.device)

        self.pos_err_prechet221 = ResNetFCN(save_folder + "_pos_err_prechet221", load_folder + "_pos_err_prechet221",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom221 = Params(save_folder + "_pos_err_prechom221", load_folder + "_pos_err_prechom221", (self.pose_size, self.pose_size), device = self.device)

        self.model_list.append(self.frc_enc21) 
        self.model_list.append(self.state_enc21) 
        self.model_list.append(self.pos_err_transdec21) 
        self.model_list.append(self.pos_err_mean021)
        self.model_list.append(self.pos_err_prechom021) 
        self.model_list.append(self.pos_err_prechet021) 
        self.model_list.append(self.pos_err_mean121)
        self.model_list.append(self.pos_err_prechom121) 
        self.model_list.append(self.pos_err_prechet121) 
        self.model_list.append(self.pos_err_mean221)
        self.model_list.append(self.pos_err_prechom221) 
        self.model_list.append(self.pos_err_prechet221) 

        self.ensemble_list.append((self.frc_enc21, self.state_enc21, self.pos_err_transdec21,\
         self.pos_err_mean021, self.pos_err_prechom021, self.pos_err_prechet021,\
          self.pos_err_mean121, self.pos_err_prechom121, self.pos_err_prechet121,\
          self.pos_err_mean221, self.pos_err_prechom221, self.pos_err_prechet221))

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        fft = torch.rfft(force, 2, normalized=False, onesided=True)
        frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        # print("Force size: ", force.size())
        # print(fft.size())
        # frc_enc = self.frc_enc(fft)
        return frc_enc

    def get_data(self, state, pose_vect, forces, hole_type, model_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, state_enc, pos_err_transdec,\
         pos_err_model0, pos_err_prechom0, pos_err_prechet0,\
          pos_err_model1, pos_err_prechom1, pos_err_prechet1,\
           pos_err_model2, pos_err_prechom2, pos_err_prechet2 = model_tuple #pos_err_prec = model_tuple # 

        frc_encs_unshaped = self.get_frc(forces, frc_enc)
        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))#.contiguous()

        state_encs_unshaped = state_enc(state)
        state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))#.contiguous()

        states = torch.cat([state_encs, frc_encs], dim = 2)#.contiguous()

        # print(states.size())
        # print(padding_masks.size())
        print("2")

        if padding_masks is None:
            rep_delta = torch.max(pos_err_transdec(states, states), 1)[0]
        else:
            rep_delta = torch.max(pos_err_transdec(states, states, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks), 1)[0]

        print("2")

        rep = torch.cat([pose_vect, rep_delta], dim = 1)

        print("2")
        pos_err_mean0 = pos_err_model0(rep)
        pos_err_mean1 = pos_err_model1(rep)
        pos_err_mean2 = pos_err_model2(rep)

        print("2")
        rep_prec0 = torch.cat([rep, pos_err_mean0], dim = 1)
        rep_prec1 = torch.cat([rep, pos_err_mean1], dim = 1)
        rep_prec2 = torch.cat([rep, pos_err_mean2], dim = 1)

        pos_err_prechet_vect0 = pos_err_prechet0(rep_prec0).transpose(0,1)
        pos_err_prechet_vect1 = pos_err_prechet1(rep_prec1).transpose(0,1)
        pos_err_prechet_vect2 = pos_err_prechet2(rep_prec2).transpose(0,1)

        tril_indices = torch.tril_indices(row=self.pose_size, col=self.pose_size)

        pos_err_tril0 = torch.zeros((self.pose_size, self.pose_size, pos_err_mean0.size(0))).to(self.device)
        pos_err_tril1 = torch.zeros((self.pose_size, self.pose_size, pos_err_mean0.size(0))).to(self.device)
        pos_err_tril2 = torch.zeros((self.pose_size, self.pose_size, pos_err_mean0.size(0))).to(self.device)

        pos_err_tril0[tril_indices[0], tril_indices[1]] = pos_err_prechet_vect0
        pos_err_tril1[tril_indices[0], tril_indices[1]] = pos_err_prechet_vect1
        pos_err_tril2[tril_indices[0], tril_indices[1]] = pos_err_prechet_vect2

        pos_err_prechet0 = prec_mult(pos_err_tril0)
        pos_err_prechet1 = prec_mult(pos_err_tril1)
        pos_err_prechet2 = prec_mult(pos_err_tril2)

        pos_err_prechom0 = prec_single(pos_err_prechom0()).unsqueeze(0).repeat_interleave(batch_size, 0)
        pos_err_prechom1 = prec_single(pos_err_prechom1()).unsqueeze(0).repeat_interleave(batch_size, 0)
        pos_err_prechom2 = prec_single(pos_err_prechom2()).unsqueeze(0).repeat_interleave(batch_size, 0)

        pos_err_prec0 = (pos_err_prechet0 + pos_err_prechom0)
        pos_err_prec1 = (pos_err_prechet1 + pos_err_prechom1)
        pos_err_prec2 = (pos_err_prechet2 + pos_err_prechom2)

        om_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_mean0.size(1), dim=1)
        om_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_mean0.size(1), dim=1)
        om_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_mean0.size(1), dim=1)
        oc_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_prec0.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec0.size(2), dim=2)
        oc_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_prec0.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec0.size(2), dim=2)
        oc_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_prec0.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec0.size(2), dim=2)

        pos_err_mean = om_hole0 * pos_err_mean0 + om_hole1 * pos_err_mean1 + om_hole2 * pos_err_mean2
        pos_err_prec = oc_hole0 * pos_err_prec0 + oc_hole1 * pos_err_prec1 + oc_hole2 * pos_err_prec2

        if not self.frc_enc21.training:
            if self.noise_mode == 'avg':
                scalar = self.train_dets()[1] / self.val_dets()[1]
            elif self.noise_mode == 'min':
                scalar = self.train_dets()[0] / self.val_dets()[2]
            else:
                scalar = self.train_dets()[2] / self.val_dets()[0]

            # print("Val dets:", self.val_dets())
            # print("Train dets:", self.train_dets())
            # print("scalar: ", scalar)
            pos_err_prec = pos_err_prec * scalar

        return pos_err_mean, pos_err_prec

    def get_logits(self, proprio_diffs, contact_diffs, pose_vect, forces, actions, peg_type, hole_type, padding_masks = None, lengths = None):
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

        prestate = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2), actions], dim = 2)
        prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2)))#.contiguous()
        print("1")
        pos_err_mean01, pos_err_prec01 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[0],\
         batch_size, sequence_size, padding_masks)
        print("1")
        pos_err_mean11, pos_err_prec11 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[1],\
         batch_size, sequence_size, padding_masks)
        print("1")

        pos_err_mean21, pos_err_prec21 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[2],\
         batch_size, sequence_size, padding_masks)
        print("1")

        om_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_mean01.size(1), dim=1)
        om_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_mean01.size(1), dim=1)
        om_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_mean01.size(1), dim=1)

        oc_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_prec01.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec01.size(2), dim=2)
        oc_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_prec01.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec01.size(2), dim=2)
        oc_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_prec01.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec01.size(2), dim=2)

        pos_err_mean1 = om_peg0 * pos_err_mean01 + om_peg1 * pos_err_mean11 + om_peg2 * pos_err_mean21

        pos_err_prec1 = oc_peg0 * pos_err_prec01 + oc_peg1 * pos_err_prec11 + oc_peg2 * pos_err_prec21

        return pos_err_mean1, pos_err_prec1

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)
        pose_vect = input_dict["pose_vect"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)
        lengths = input_dict["length"].to(self.device).long().squeeze(1)
        error = input_dict["pos_err"].to(self.device)

        pos_err_mean, pos_err_prec = self.get_logits(proprio_diffs, contact_diffs, pose_vect, forces, actions,\
         peg_type, hole_type, padding_masks, lengths)

        err_sqr = (error - pos_err_mean).pow(2).sum(1)
        cov_dets = err_sqr

        print(prec_dets.size())

        # if self.frc_enc21.training:
        #     self.train_dets()[0] = (1 - self.train_lr) * self.train_dets()[0] + self.train_lr * cov_dets.min()
        #     self.train_dets()[1] = (1 - self.train_lr) * self.train_dets()[1] + self.train_lr * cov_dets.mean()       
        #     self.train_dets()[2] = (1 - self.train_lr) * self.train_dets()[2] + self.train_lr * cov_dets.max()
        # else:
        #     self.val_dets()[0] = (1 - self.val_lr) * self.val_dets()[0] + self.val_lr * cov_dets.min()
        #     self.val_dets()[1] = (1 - self.val_lr) * self.val_dets()[1] + self.val_lr * cov_dets.mean()       
        #     self.val_dets()[2] = (1 - self.val_lr) * self.val_dets()[2] + self.val_lr * cov_dets.max()

        return {
            'pos_err_params': (pos_err_mean, pos_err_prec),
        }

    def process(self, input_dict):
        proprios = input_dict["proprio"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        actions = input_dict["action"].to(self.device)[:, :-1]
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)
        pose_vect = input_dict["pose_vect"].to(self.device)

        proprio_diffs = proprios[:,1:] - proprios[:, :-1]
        contact_diffs = contacts[:,1:] - contacts[:, :-1]
        force_clipped = forces[:,1:]        

        pos_err_mean, pos_err_prec = self.get_logits(proprio_diffs, contact_diffs, pose_vect, force_clipped, actions, peg_type, hole_type)   

        return pos_err_mean, pos_err_prec

class Options_PredictionResNet(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, pose_size, num_options, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []

        self.pos_size = 3
        self.z_dim = 48
        self.num_options = num_options

        self.ensemble_list = []

        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 001
        self.expand_state001 = FCN(save_folder + "_expand_state001", load_folder + "_expand_state001",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred001 = ResNetFCN(save_folder + "_options_pred001", load_folder + "_options_pred001",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred001)
        self.model_list.append(self.expand_state001) 

        # Network 011
        self.expand_state011 = FCN(save_folder + "_expand_state011", load_folder + "_expand_state011",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred011 = ResNetFCN(save_folder + "_options_pred011", load_folder + "_options_pred011",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred011)
        self.model_list.append(self.expand_state011)

        # Network 021
        self.expand_state021 = FCN(save_folder + "_expand_state021", load_folder + "_expand_state021",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred021 = ResNetFCN(save_folder + "_options_pred021", load_folder + "_options_pred021",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred021)
        self.model_list.append(self.expand_state021) 

        # Network 101
        self.expand_state101 = FCN(save_folder + "_expand_state101", load_folder + "_expand_state101",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred101 = ResNetFCN(save_folder + "_options_pred101", load_folder + "_options_pred101",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred101)
        self.model_list.append(self.expand_state101) 

        # Network 111
        self.expand_state111 = FCN(save_folder + "_expand_state111", load_folder + "_expand_state111",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred111 = ResNetFCN(save_folder + "_options_pred111", load_folder + "_options_pred111",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred111)
        self.model_list.append(self.expand_state111) 

        # Network 121
        self.expand_state121 = FCN(save_folder + "_expand_state121", load_folder + "_expand_state121",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred121 = ResNetFCN(save_folder + "_options_pred121", load_folder + "_options_pred121",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred121)
        self.model_list.append(self.expand_state121) 

        # Network 201
        self.expand_state201 = FCN(save_folder + "_expand_state201", load_folder + "_expand_state201",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred201 = ResNetFCN(save_folder + "_options_pred201", load_folder + "_options_pred201",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred201)
        self.model_list.append(self.expand_state201) 

        # Network 211
        self.expand_state211 = FCN(save_folder + "_expand_state211", load_folder + "_expand_state211",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred211 = ResNetFCN(save_folder + "_options_pred211", load_folder + "_options_pred211",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred211)
        self.model_list.append(self.expand_state211)

        # Network 221
        self.expand_state221 = FCN(save_folder + "_expand_state221", load_folder + "_expand_state221",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.options_pred221 = ResNetFCN(save_folder + "_options_pred221", load_folder + "_options_pred221",\
            self.z_dim, self.num_options, 3, device = self.device)
 
        self.model_list.append(self.options_pred221)
        self.model_list.append(self.expand_state221) 


        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, init_pos, command_pos, peg_type, hole_type):
        state = torch.cat([init_pos, command_pos], dim = 1)

        options_logits001 = self.options_pred001(self.expand_state001(state))
        options_logits011 = self.options_pred011(self.expand_state011(state))
        options_logits021 = self.options_pred021(self.expand_state021(state))

        options_logits101 = self.options_pred101(self.expand_state101(state))
        options_logits111 = self.options_pred111(self.expand_state111(state))
        options_logits121 = self.options_pred121(self.expand_state121(state))

        options_logits201 = self.options_pred201(self.expand_state201(state))
        options_logits211 = self.options_pred211(self.expand_state211(state))
        options_logits221 = self.options_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)

        ol_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)

        options_logits01 = ol_hole0 * options_logits001 + ol_hole1 * options_logits101 + ol_hole2 * options_logits201

        options_logits11 = ol_hole0 * options_logits011 + ol_hole1 * options_logits111 + ol_hole2 * options_logits211

        options_logits21 = ol_hole0 * options_logits021 + ol_hole1 * options_logits121 + ol_hole2 * options_logits221

        options_logits1 = ol_peg0 * options_logits01 + ol_peg1 * options_logits11 + ol_peg2 * options_logits21

        return options_logits1

    def forward(self, input_dict):
        init_pos = input_dict["init_pos"].to(self.device)
        command_pos = input_dict["command_pos"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)

        options_logits = self.get_pred(init_pos, command_pos, peg_type, hole_type)

        return {
            'options_class': options_logits,
        }

    def process(self, state, peg_type):
        options_logits001 = self.options_pred001(self.expand_state001(state)) # cross, cross
        options_logits011 = self.options_pred011(self.expand_state011(state)) # cross, rect
        options_logits021 = self.options_pred021(self.expand_state021(state)) # cross, square

        options_logits101 = self.options_pred101(self.expand_state101(state))
        options_logits111 = self.options_pred111(self.expand_state111(state))
        options_logits121 = self.options_pred121(self.expand_state121(state))

        options_logits201 = self.options_pred201(self.expand_state201(state))
        options_logits211 = self.options_pred211(self.expand_state211(state))
        options_logits221 = self.options_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits001.size(1), dim=1)

        options_logits01 = ol_peg0 * options_logits001 + ol_peg1 * options_logits011 + ol_peg2 * options_logits021 # cross hole logits

        options_logits11 = ol_peg0 * options_logits101 + ol_peg1 * options_logits111 + ol_peg2 * options_logits121 # rect hole logits

        options_logits21 = ol_peg0 * options_logits201 + ol_peg1 * options_logits211 + ol_peg2 * options_logits221 # square hole logits

        return torch.cat([options_logits01.unsqueeze(1), options_logits11.unsqueeze(1), options_logits21.unsqueeze(1)], dim = 1)

class PosErr_PredictionResNet(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, pose_size, num_options, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []

        self.pos_size = 3
        self.z_dim = 48
        self.num_options = num_options

        self.ensemble_list = []

        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 001
        self.expand_state001 = FCN(save_folder + "_expand_state001", load_folder + "_expand_state001",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred001 = ResNetFCN(save_folder + "_pos_err_pred001", load_folder + "_pos_err_pred001",\
            self.z_dim, self.pos_size, 3, device = self.device)

        self.pos_err_prechet001 = ResNetFCN(save_folder + "_pos_err_prechet001", load_folder + "_pos_err_prechet001",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom001 = Params(save_folder + "_pos_err_prechom001", load_folder + "_pos_err_prechom001", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred001)
        self.model_list.append(self.expand_state001)
        self.model_list.append(self.pos_err_prechet001)
        self.model_list.append(self.pos_err_prechom001)

        # Network 011
        self.expand_state011 = FCN(save_folder + "_expand_state011", load_folder + "_expand_state011",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred011 = ResNetFCN(save_folder + "_pos_err_pred011", load_folder + "_pos_err_pred011",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet011 = ResNetFCN(save_folder + "_pos_err_prechet011", load_folder + "_pos_err_prechet011",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom011 = Params(save_folder + "_pos_err_prechom011", load_folder + "_pos_err_prechom011", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred011)
        self.model_list.append(self.expand_state011)
        self.model_list.append(self.pos_err_prechet011)
        self.model_list.append(self.pos_err_prechom011)

        # Network 021
        self.expand_state021 = FCN(save_folder + "_expand_state021", load_folder + "_expand_state021",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred021 = ResNetFCN(save_folder + "_pos_err_pred021", load_folder + "_pos_err_pred021",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet021 = ResNetFCN(save_folder + "_pos_err_prechet021", load_folder + "_pos_err_prechet021",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom021 = Params(save_folder + "_pos_err_prechom021", load_folder + "_pos_err_prechom021", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred021)
        self.model_list.append(self.expand_state021)
        self.model_list.append(self.pos_err_prechet021)
        self.model_list.append(self.pos_err_prechom021) 

        # Network 101
        self.expand_state101 = FCN(save_folder + "_expand_state101", load_folder + "_expand_state101",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred101 = ResNetFCN(save_folder + "_pos_err_pred101", load_folder + "_pos_err_pred101",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet101 = ResNetFCN(save_folder + "_pos_err_prechet101", load_folder + "_pos_err_prechet101",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom101 = Params(save_folder + "_pos_err_prechom101", load_folder + "_pos_err_prechom101", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred101)
        self.model_list.append(self.expand_state101)
        self.model_list.append(self.pos_err_prechet101)
        self.model_list.append(self.pos_err_prechom101) 

        # Network 111
        self.expand_state111 = FCN(save_folder + "_expand_state111", load_folder + "_expand_state111",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred111 = ResNetFCN(save_folder + "_pos_err_pred111", load_folder + "_pos_err_pred111",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet111 = ResNetFCN(save_folder + "_pos_err_prechet111", load_folder + "_pos_err_prechet111",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom111 = Params(save_folder + "_pos_err_prechom111", load_folder + "_pos_err_prechom111", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred111)
        self.model_list.append(self.expand_state111)
        self.model_list.append(self.pos_err_prechet111)
        self.model_list.append(self.pos_err_prechom111) 

        # Network 121
        self.expand_state121 = FCN(save_folder + "_expand_state121", load_folder + "_expand_state121",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred121 = ResNetFCN(save_folder + "_pos_err_pred121", load_folder + "_pos_err_pred121",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet121 = ResNetFCN(save_folder + "_pos_err_prechet121", load_folder + "_pos_err_prechet121",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom121 = Params(save_folder + "_pos_err_prechom121", load_folder + "_pos_err_prechom121", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred121)
        self.model_list.append(self.expand_state121)
        self.model_list.append(self.pos_err_prechet121)
        self.model_list.append(self.pos_err_prechom121) 

        # Network 201
        self.expand_state201 = FCN(save_folder + "_expand_state201", load_folder + "_expand_state201",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred201 = ResNetFCN(save_folder + "_pos_err_pred201", load_folder + "_pos_err_pred201",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet201 = ResNetFCN(save_folder + "_pos_err_prechet201", load_folder + "_pos_err_prechet201",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom201 = Params(save_folder + "_pos_err_prechom201", load_folder + "_pos_err_prechom201", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred201)
        self.model_list.append(self.expand_state201)
        self.model_list.append(self.pos_err_prechet201)
        self.model_list.append(self.pos_err_prechom201) 

        # Network 211
        self.expand_state211 = FCN(save_folder + "_expand_state211", load_folder + "_expand_state211",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred211 = ResNetFCN(save_folder + "_pos_err_pred211", load_folder + "_pos_err_pred211",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet211 = ResNetFCN(save_folder + "_pos_err_prechet211", load_folder + "_pos_err_prechet211",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom211 = Params(save_folder + "_pos_err_prechom211", load_folder + "_pos_err_prechom211", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred211)
        self.model_list.append(self.expand_state211)
        self.model_list.append(self.pos_err_prechet211)
        self.model_list.append(self.pos_err_prechom211)

        # Network 221
        self.expand_state221 = FCN(save_folder + "_expand_state221", load_folder + "_expand_state221",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred221 = ResNetFCN(save_folder + "_pos_err_pred221", load_folder + "_pos_err_pred221",\
            self.z_dim, 1, 3, device = self.device)

        self.pos_err_prechet221 = ResNetFCN(save_folder + "_pos_err_prechet221", load_folder + "_pos_err_prechet221",\
            self.state_size + self.extra + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom221 = Params(save_folder + "_pos_err_prechom221", load_folder + "_pos_err_prechom221", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred221)
        self.model_list.append(self.expand_state221)
        self.model_list.append(self.pos_err_prechet221)
        self.model_list.append(self.pos_err_prechom221) 


        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, init_pos, command_pos, pos_err, peg_type, hole_type):
        state = torch.cat([init_pos, command_pos, pos_err], dim = 1)

        pos_err_logits001 = self.pos_err_pred001(self.expand_state001(state))
        pos_err_logits011 = self.pos_err_pred011(self.expand_state011(state))
        pos_err_logits021 = self.pos_err_pred021(self.expand_state021(state))

        pos_err_logits101 = self.pos_err_pred101(self.expand_state101(state))
        pos_err_logits111 = self.pos_err_pred111(self.expand_state111(state))
        pos_err_logits121 = self.pos_err_pred121(self.expand_state121(state))

        pos_err_logits201 = self.pos_err_pred201(self.expand_state201(state))
        pos_err_logits211 = self.pos_err_pred211(self.expand_state211(state))
        pos_err_logits221 = self.pos_err_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)

        ol_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)

        pos_err_logits01 = ol_hole0 * pos_err_logits001 + ol_hole1 * pos_err_logits101 + ol_hole2 * pos_err_logits201

        pos_err_logits11 = ol_hole0 * pos_err_logits011 + ol_hole1 * pos_err_logits111 + ol_hole2 * pos_err_logits211

        pos_err_logits21 = ol_hole0 * pos_err_logits021 + ol_hole1 * pos_err_logits121 + ol_hole2 * pos_err_logits221

        pos_err_logits1 = ol_peg0 * pos_err_logits01 + ol_peg1 * pos_err_logits11 + ol_peg2 * pos_err_logits21

        return pos_err_logits1

    def forward(self, input_dict):
        init_pos = input_dict["errorinit_pos"].to(self.device)
        command_pos = input_dict["command_pos"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)

        pos_err_mean, pos_err_prec = self.get_pred(init_pos, command_pos, peg_type, hole_type)

        return {
            'pos_err_params': (pos_err_mean, pos_err_prec),
        }

    def process(self, state, peg_type, hole_type):
        pos_err_logits001 = self.pos_err_pred001(self.expand_state001(state)) # cross, cross
        pos_err_logits011 = self.pos_err_pred011(self.expand_state011(state)) # cross, rect
        pos_err_logits021 = self.pos_err_pred021(self.expand_state021(state)) # cross, square

        pos_err_logits101 = self.pos_err_pred101(self.expand_state101(state))
        pos_err_logits111 = self.pos_err_pred111(self.expand_state111(state))
        pos_err_logits121 = self.pos_err_pred121(self.expand_state121(state))

        pos_err_logits201 = self.pos_err_pred201(self.expand_state201(state))
        pos_err_logits211 = self.pos_err_pred211(self.expand_state211(state))
        pos_err_logits221 = self.pos_err_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)

        pos_err_logits01 = ol_peg0 * pos_err_logits001 + ol_peg1 * pos_err_logits011 + ol_peg2 * pos_err_logits021 # cross hole logits

        pos_err_logits11 = ol_peg0 * pos_err_logits101 + ol_peg1 * pos_err_logits111 + ol_peg2 * pos_err_logits121 # rect hole logits

        pos_err_logits21 = ol_peg0 * pos_err_logits201 + ol_peg1 * pos_err_logits211 + ol_peg2 * pos_err_logits221 # square hole logits

        pe_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        pe_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)
        pe_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_logits001.size(1), dim=1)

        pos_err_ratio = pe_hole0 * pos_err_logits01 + pe_hole1 * pos_err_logits11 + pe_hole2 * pos_err_logits21

        return pos_err_ratio.squeeze()
        
# class Origin_DetectionTransformer(Proto_Macromodel):
#     def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, min_steps, use_fft = True, device = None):
#         super().__init__()

#         if info_flow[model_name]["model_folder"] != "":
#             load_folder = info_flow[model_name]["model_folder"] + model_name
#         else:
#             load_folder = model_folder + model_name

#         save_folder = model_folder + model_name 

#         self.device = device
#         self.model_list = []

#         self.action_dim = action_dim[0]
#         self.proprio_size = proprio_size[0]
#         self.pose_size = 3
#         self.force_size = force_size
#         self.num_options = num_options
#         self.contact_size = 1
#         self.frc_enc_size =  8 * 3 * 2
#         self.state_size = 128
#         self.encoded_size = self.state_size - self.frc_enc_size
#         self.use_fft = use_fft
#         self.min_steps = min_steps

#         self.prestate_size = self.proprio_size + self.contact_size + self.action_dim

#         self.ensemble_list = []

#         # first number indicates index in peg vector, second number indicates number in ensemble
#         # Network 01
#         self.frc_enc01 = CONV2DN(save_folder + "_fft_enc01", load_folder + "_fft_enc01",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc01 = ResNetFCN(save_folder + "_state_enc01", load_folder + "_state_enc01",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.origin_transdec01 = Transformer_Decoder(save_folder + "_origin_transdec01", load_folder + "_origin_transdec01",\
#          self.state_size, 6, device = self.device)

#         self.origin_mean01 = ResNetFCN(save_folder + "_origin_mean001", load_folder + "_origin_mean001",\
#             self.state_size, self.pose_size, 3, device = self.device)

#         self.origin_prec01 = Params(save_folder + "_origin_noise001", load_folder + "_origin_noise001", (self.pose_size, self.pose_size), device = self.device)

#         self.model_list.append(self.frc_enc01) 
#         self.model_list.append(self.state_enc01) 
#         self.model_list.append(self.origin_transdec01) 
#         self.model_list.append(self.origin_mean01)
#         self.model_list.append(self.origin_noise01) 

#         self.ensemble_list.append((self.frc_enc01, self.state_enc01, self.origin_transdec01,\
#          self.origin_mean01, self.origin_noise01))

#         #Network 11
#         self.frc_enc11 = CONV2DN(save_folder + "_fft_enc11", load_folder + "_fft_enc11",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc11 = ResNetFCN(save_folder + "_state_enc11", load_folder + "_state_enc11",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.origin_transdec11 = Transformer_Decoder(save_folder + "_origin_transdec11", load_folder + "_origin_transdec11",\
#          self.state_size, 6, device = self.device)

#         self.origin_mean11 = ResNetFCN(save_folder + "_origin_mean11", load_folder + "_origin_mean11",\
#             self.state_size, self.pose_size, 3, device = self.device)

#         self.origin_noise11 = Params(save_folder + "_origin_noise11", load_folder + "_origin_noise11", (self.pose_size, self.pose_size), device = self.device)

#         self.model_list.append(self.frc_enc11) 
#         self.model_list.append(self.state_enc11) 
#         self.model_list.append(self.origin_transdec11) 
#         self.model_list.append(self.origin_mean11)
#         self.model_list.append(self.origin_noise11) 

#         self.ensemble_list.append((self.frc_enc11, self.state_enc11, self.origin_transdec11,\
#          self.origin_mean11, self.origin_noise11))


#         # Network 21
#         self.frc_enc21 = CONV2DN(save_folder + "_fft_enc21", load_folder + "_fft_enc21",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc21 = ResNetFCN(save_folder + "_state_enc21", load_folder + "_state_enc21",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.origin_transdec21 = Transformer_Decoder(save_folder + "_origin_transdec21", load_folder + "_origin_transdec21",\
#          self.state_size, 6, device = self.device)

#         self.origin_mean21 = ResNetFCN(save_folder + "_origin_mean21", load_folder + "_origin_mean21",\
#             self.state_size, self.pose_size, 3, device = self.device)

#         self.origin_noise21 = Params(save_folder + "_origin_noise21", load_folder + "_origin_noise21", (self.pose_size, self.pose_size), device = self.device)

#         self.model_list.append(self.frc_enc21) 
#         self.model_list.append(self.state_enc21) 
#         self.model_list.append(self.origin_transdec21) 
#         self.model_list.append(self.origin_mean21)
#         self.model_list.append(self.origin_noise21) 

#         self.ensemble_list.append((self.frc_enc21, self.state_enc21, self.origin_transdec21,\
#          self.origin_mean21, self.origin_noise21))

#         if info_flow[model_name]["model_folder"] != "":
#             self.load(info_flow[model_name]["epoch_num"])

#     def get_frc(self, force, model):
#         fft = torch.rfft(force, 2, normalized=False, onesided=True)
#         frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
#         # print("Force size: ", force.size())
#         # print(fft.size())
#         # frc_enc = self.frc_enc(fft)
#         return frc_enc

#     def get_data(self, state, forces, model_tuple, batch_size, sequence_size, padding_masks = None):
#         frc_enc, state_enc, origin_transdec, origin_model, origin_noise = model_tuple #origin_cov = model_tuple # 

#         frc_encs_unshaped = self.get_frc(forces, frc_enc)
#         frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))#.contiguous()

#         state_encs_unshaped = state_enc(state)
#         state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))#.contiguous()

#         states = torch.cat([state_encs, frc_encs], dim = 2).transpose(0,1)#.contiguous()

#         if padding_masks is None:
#             rep = torch.max(origin_transdec(states, states), 0)[0]
#             origin_mean = origin_model(rep)
#         else:
#             rep = torch.max(origin_transdec(states, states, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks), 0)[0]
#             origin_mean = origin_model(rep)


#             # origin_cov_vector = origin_cov(rep).transpose(0,1)

#         # tril_indices = torch.tril_indices(row=self.pose_size, col=self.pose_size)
#         # origin_tril = torch.zeros((self.pose_size, self.pose_size, origin_mean.size(0))).to(self.device)
#         # origin_tril[tril_indices[0], tril_indices[1]] += origin_cov_vector
#         # origin_cov = norm_cov(origin_tril)

#         # origin_mean = origin_det(origin_transdec(final_state, states, mem_padding_mask = padding_masks).squeeze(0))

#         origin_cov = norm_cov(origin_noise()).unsqueeze(0).repeat_interleave(batch_size, 0)

#         return origin_mean, origin_cov

#     def get_logits(self, proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks = None, lengths = None):
#         batch_size = proprio_diffs.size(0)
#         sequence_size = proprio_diffs.size(1)

#         forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

#         prestate = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2), actions], dim = 2)
#         prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2)))#.contiguous()

#         origin_mean01, origin_cov01 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[0],\
#          batch_size, sequence_size, padding_masks)

#         origin_mean11, origin_cov11 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[1],\
#          batch_size, sequence_size, padding_masks)

#         origin_mean21, origin_cov21 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[2],\
#          batch_size, sequence_size, padding_masks)

#         om_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)
#         om_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)
#         om_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(origin_mean01.size(1), dim=1)

#         oc_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(origin_cov01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_cov01.size(2), dim=2)
#         oc_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(origin_cov01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_cov01.size(2), dim=2)
#         oc_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(origin_cov01.size(1), dim=1).unsqueeze(2).repeat_interleave(origin_cov01.size(2), dim=2)

#         origin_mean1 = om_peg0 * origin_mean01 + om_peg1 * origin_mean11 + om_peg2 * origin_mean21

#         origin_cov1 = oc_peg0 * origin_cov01 + oc_peg1 * origin_cov11 + oc_peg2 * origin_cov21

#         return origin_mean1, origin_cov1

#     def forward(self, input_dict):
#         proprio_diffs = input_dict["proprio_diff"].to(self.device)
#         actions = input_dict["action"].to(self.device)
#         forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
#         contact_diffs = input_dict["contact_diff"].to(self.device)
#         peg_type = input_dict["peg_type"].to(self.device)
#         hole_type = input_dict["hole_type"].to(self.device)
#         padding_masks = input_dict["padding_mask"].to(self.device)
#         lengths = input_dict["length"].to(self.device).long().squeeze(1)

#         origin_mean, origin_cov = self.get_logits(proprio_diffs, contact_diffs, forces, actions,\
#          peg_type, hole_type, padding_masks, lengths)

#         return {
#             'origin_params': (origin_mean, origin_cov),
#         }

#     def process(self, input_dict):
#         proprios = input_dict["proprio"].to(self.device)
#         forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
#         contacts = input_dict["contact"].to(self.device)
#         actions = input_dict["action"].to(self.device)[:, :-1]
#         peg_type = input_dict["peg_type"].to(self.device)

#         proprio_diffs = proprios[:,1:] - proprios[:, :-1]
#         contact_diffs = contacts[:,1:] - contacts[:, :-1]
#         force_clipped = forces[:,1:]        

#         origin_mean, origin_cov = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, peg_type)   

#         return origin_mean, origin_cov