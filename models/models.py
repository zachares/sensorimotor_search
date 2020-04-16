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

def log_normal(x, m, v):
    return -0.5 * ((x - m).pow(2)/ v + torch.log(2 * np.pi * v)).sum(-1).unsqueeze(-1)
######################################
# Def:ning Custom Macromodels for project
#######################################

#### see LSTM in models_utils for example of how to set up a macromodel
class Options_Sensor(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_dim, num_options, use_fft = True, device = None):
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
        self.use_fft = use_fft

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim
        self.macro_action = 2 * self.pose_size

        self.ensemble_list = []
        self.num_tl = 4
        self.num_cl = 3

        for i in range(self.num_options):
            self.ensemble_list.append((\
                CONV2DN(save_folder + "_fft_enc" + str(i), load_folder + "_fft_enc" + str(i),\
         (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device).to(self.device),\

                Transformer_Decoder(save_folder + "_state_transdec" + str(i), load_folder + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, device = self.device).to(self.device),\

                ResNetFCN(save_folder + "_options_class" + str(i), load_folder + "_options_class" + str(i),\
            self.state_size, self.num_options, self.num_cl, device = self.device).to(self.device)\

                ResNetFCN(save_folder + "_options_conf" + str(i) + str(0), load_folder + "_options_conf" + str(i) + str(0),\
            self.state_size + self.macro_action, self.num_options, self.num_cl, device = self.device).to(self.device)

                ResNetFCN(save_folder + "_options_conf" + str(i) + str(1), load_folder + "_options_conf" + str(i) + str(1),\
            self.state_size + self.macro_action, self.num_options, self.num_cl, device = self.device).to(self.device)

                ResNetFCN(save_folder + "_options_conf" + str(i) + str(2), load_folder + "_options_conf" + str(i) + str(2),\
            self.state_size + self.macro_action, self.num_options, self.num_cl, device = self.device).to(self.device)
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        fft = torch.rfft(force, 2, normalized=False, onesided=True)
        frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        return frc_enc

    def get_enc(self, states, forces, actions, encoder_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, state_transdec = encoder_tuple

        frc_encs_unshaped = self.get_frc(forces, frc_enc)

        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))

        states_t = torch.cat([states, frc_encs, actions], dim = 2).transpose(0,1) 

        if padding_masks is None:
            seq_encs = state_transdec(states_t, states_t).max(0)[0]
        else:
            seq_encs = state_transdec(states_t, states_t, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks).max(0)[0]

        return seq_encs

    def get_data(self, states, forces, actions, macro_action, hole_type, model_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, transformer, options_class, options_conf0, options_conf1, options_conf2 = model_tuple #origin_cov = model_tuple # 
        conf_tuple = (options_conf0, options_conf1, options_conf2)

        seq_enc = self.get_enc(states, forces, actions, (frc_enc, transformer), batch_size, sequence_size, padding_masks = padding_masks)

        options_logits = options_class(seq_enc)

        options_conf = torch.zeros_like(hole_type)

        for i, model in enumerate(conf_tuple):
            hole_bool = hole_type[:,i].unsqueeze(1).repeat_interleave(options_conf.size(1), dim=1)
            options_conf += hole_bool * model(torch.cat([seq_enc, macro_action], dim = 1))           

        return options_logits, options_conf

    def get_logits(self, proprio_diffs, contact_diffs, forces, actions, macro_action, peg_type, hole_type, padding_masks = None):
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

        states = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2)], dim = 2)

        options_logits = torch.zeros_like(peg_type)
        conf_logits = torch.zeros_like(peg_type)

        for i in range(self.num_options):
            peg_bool = peg_type[:,i].unsqueeze(1).repeat_interleave(options_logits.size(1), dim=1)

            options_logits_i, conf_logits_i = self.get_data(states, forces_reshaped, actions, macro_action, hole_type,\
             self.ensemble_list[i], batch_size, sequence_size, padding_masks)

            options_logits += peg_bool * options_logits_i
            conf_logits += peg_bool * conf_logits_i

        return options_logits, conf_logits

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)
        macro_action = input_dict["macro_action"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)

        options_logits, conf_logits = self.get_logits(proprio_diffs, contact_diffs, forces, actions, macro_action, peg_type, hole_type, padding_masks)

        probs = F.softmax(options_logits, dim = 1)

        hole_est = torch.zeros_like(probs)
        hole_est[torch.arange(hole_est.size(0)), probs.max(1)[1]] = 1.0

        return {
            'options_class': options_logits,
            'conf_class': conf_logits
            'hole_est': hole_est,
        }

    def logprobs(self, input_dict, peg_type, hole_type, macro_action):
        proprios = input_dict["proprio"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        actions = input_dict["action"].to(self.device)[:, :-1]

        proprio_diffs = proprios[:,1:] - proprios[:, :-1]
        contact_diffs = contacts[:,1:] - contacts[:, :-1]
        force_clipped = forces[:,1:]

        options_logits, conf_logits = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, macro_action, peg_type, hole_type)

        probs = F.softmax(options_logits, dim = 1)
        conf_logprobs = F.log_softmax(conf_logits, dim = 1)

        conf_logprob = conf_logprobs[torch.arange(conf_logprobs.size(0)), probs.max(1)[1]]

        return conf_logprob

class Options_Predictor(Proto_Macromodel):
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
        self.z_dim = 32
        self.num_options = num_options

        self.ensemble_list = []
        self.nl = 3

        # first number indicates index in peg vector, second number indicates number in ensemble
        self.model_dict = {}
        for i in range(self.num_options):
            for j in range(self.num_options):
                self.model_dict[(i,j)] = (\
                    FCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
                    2 * self.pos_size, self.z_dim, 1, device = self.device).to(self.device),\
                     ResNetFCN(save_folder + "_options_pred" + str(i) + str(j), load_folder + "_options_pred" + str(i) + str(j),\
                    self.z_dim, self.num_options, self.nl, device = self.device).to(self.device),\
                     ResNetFCN(save_folder + "_conf_pred" + str(i) + str(j), load_folder + "_conf_pred" + str(i) + str(j),\
                    self.z_dim, self.num_options, self.nl, device = self.device).to(self.device)\
                     )

        for key in self.model_dict.keys():
            for model in self.model_dict[key]:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, macro_action, peg_type, hole_type):
        options_logits = torch.zeros_like(init_pos[:, :self.num_options])
        conf_logits = torch.zeros_like(init_pos[:, :self.num_options])

        for key in self.model_dict.keys():
            expand_state, options_pred, conf_pred = self.model_dict[key]
            i, j = key #peg, hole
            peg_bool = peg_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim=1)
            hole_bool = hole_type[:,j].unsqueeze(1).repeat_interleave(self.num_options, dim=1)

            estate = expand_state(macro_action)

            options_logits += peg_bool * hole_bool * options_pred(estate)
            conf_logits += peg_bool * hole_bool * conf_pred(estate)

        return options_logits, conf_logits

    def forward(self, input_dict):
        macro_action = input_dict["macro_action"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)

        options_logits, conf_logits = self.get_pred(macro_action, peg_type, hole_type)

        return {
            'options_class': options_logits,
            'conf_class': conf_logits
        }

    def logits(self, macro_action, peg_type, hole_type):
        options_logits = torch.zeros_like(peg_type)

        options_logits = torch.zeros_like(init_pos[:, :self.num_options])
        conf_logits = torch.zeros_like(init_pos[:, :self.num_options])

        for key in self.model_dict.keys():
            expand_state, options_pred, conf_pred = self.model_dict[key]
            i, j = key #peg, hole
            peg_bool = peg_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim=1)
            hole_bool = hole_type[:,j].unsqueeze(1).repeat_interleave(self.num_options, dim=1)

            estate = expand_state(macro_action)

            options_logits += peg_bool * hole_bool * options_pred(estate)
            conf_logits += peg_bool * hole_bool * conf_pred(estate)

        probs = F.softmax(options_logits, dim = 1)
        conf_logprobs = F.log_softmax(conf_logits, dim = 1)

        conf_logprob = conf_logprobs[torch.arange(conf_logprobs.size(0)), probs.max(1)[1]]

        return conf_logprob

class Options_Mat(Proto_Macromodel):
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
        self.z_dim = 32
        self.num_options = num_options

        self.ensemble_list = []
        self.nl = 3
        self.tb = True

        self.conf_mat = Params(save_folder + "_conf_mat", load_folder + "_conf_mat",\
         (self.num_options, self.num_options, self.num_options), device = self.device)\
                 
        self.conf_mat().data[:] = 0

        self.model_list.append(self.conf_mat)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])


    def forward(self, input_dict):
        init_pos = input_dict["init_pos"].to(self.device)
        command_pos = input_dict["command_pos"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)
        hole_est = input_dict["hole_est"].to(self.device)

        tb = self.conf_mat.training

        if tb == self.tb and tb: # continuing to train

            rows = peg_type.max(1)[1]
            cols = hole_type.max(1)[1]
            widths = hole_est.max(1)[1]

            for i in range(rows.size(0)):
                self.conf_mat().data[rows[i], cols[i], widths[i]] += 1

        elif tb != self.tb and tb: # switched to training 
            self.tb = tb
            self.conf_mat().data[:] = 0
            rows = peg_type.max(1)[1]
            cols = hole_type.max(1)[1]
            widths = hole_est.max(1)[1]

            for i in range(rows.size(0)):
                self.conf_mat().data[rows[i], cols[i], widths[i]] += 1


        elif tb != self.tb and not tb: # switched to validation
            self.tb = tb
            for i in range(self.num_options):
                if i == 0:
                    print("Cross")
                elif i == 1:
                    print("Rectangle")
                else:
                    print("Square")

                print(self.conf_mat().data[i])


    def logits(self, action, peg_type, hole_type):
        probs = torch.zeros_like(peg_type)

        for i in range(peg_type.size(0)):
            row = peg_type[i].max(0)[1]
            col = hole_type[i].max(0)[1]

            probs[i] = self.conf_mat().data[row, col] / torch.sum(self.conf_mat().data[row, col])

        return torch.log(probs)

class Insertion_PredictionResNet(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, pose_size, device = None):
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
        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 001
        self.expand_state001 = FCN(save_folder + "_expand_state001", load_folder + "_expand_state001",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.insert_pred001 = ResNetFCN(save_folder + "_insert_pred001", load_folder + "_insert_pred001",\
            self.z_dim, 1, 1, device = self.device)
 
        self.model_list.append(self.insert_pred001)
        self.model_list.append(self.expand_state001) 

        # Network 111
        self.expand_state111 = FCN(save_folder + "_expand_state111", load_folder + "_expand_state111",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.insert_pred111 = ResNetFCN(save_folder + "_insert_pred111", load_folder + "_insert_pred111",\
            self.z_dim, 1, 1, device = self.device)
 
        self.model_list.append(self.insert_pred111)
        self.model_list.append(self.expand_state111) 

        # Network 221
        self.expand_state221 = FCN(save_folder + "_expand_state221", load_folder + "_expand_state221",\
            2 * self.pos_size, self.z_dim, 1, device = self.device)

        self.insert_pred221 = ResNetFCN(save_folder + "_insert_pred221", load_folder + "_insert_pred221",\
            self.z_dim, 1, 1, device = self.device)
 
        self.model_list.append(self.insert_pred221)
        self.model_list.append(self.expand_state221) 


        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, init_pos, command_pos, peg_type, hole_type):
        state = torch.cat([init_pos, command_pos], dim = 1)

        insert_logits001 = self.insert_pred001(self.expand_state001(state))
        insert_logits111 = self.insert_pred111(self.expand_state111(state))
        insert_logits221 = self.insert_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)

        ol_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)

        zero_constant = -600

        insert_logits011, insert_logits021, insert_logits101, insert_logits121,\
        insert_logits201, insert_logits211 = zero_constant, zero_constant, zero_constant, zero_constant, zero_constant, zero_constant
        insert_logits01 = ol_hole0 * insert_logits001 + ol_hole1 * insert_logits011  + ol_hole2 * insert_logits021 

        insert_logits11 = ol_hole0 * insert_logits101 + ol_hole1 * insert_logits111 + ol_hole2 * insert_logits121

        insert_logits21 = ol_hole0 * insert_logits201  + ol_hole1 * insert_logits211  + ol_hole2 * insert_logits221

        insert_logits1 = ol_peg0 * insert_logits01 + ol_peg1 * insert_logits11 + ol_peg2 * insert_logits21

        return insert_logits1

    def forward(self, input_dict):
        init_pos = input_dict["init_pos"].to(self.device)
        command_pos = input_dict["command_pos"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)

        insert_logits = self.get_pred(init_pos, command_pos, peg_type, hole_type)

        return {
            'insert_class': insert_logits,
        }

    def process(self, state, peg_type):
        insert_logits001 = self.insert_pred001(self.expand_state001(state))
        insert_logits111 = self.insert_pred111(self.expand_state111(state))
        insert_logits221 = self.insert_pred221(self.expand_state221(state))

        ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)

        # ol_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        # ol_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)
        # ol_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(insert_logits001.size(1), dim=1)

        zero_constant = -600

        insert_logits011, insert_logits021, insert_logits101, insert_logits121,\
        insert_logits201, insert_logits211 = zero_constant, zero_constant, zero_constant, zero_constant, zero_constant, zero_constant

        insert_logits0_1 = ol_peg0 * insert_logits001 + ol_peg1 * insert_logits101  + ol_peg2 * insert_logits201

        insert_logits1_1 = ol_peg0 * insert_logits011 + ol_peg1 * insert_logits111 + ol_peg2 * insert_logits211

        insert_logits2_1 = ol_peg0 * insert_logits021 + ol_peg1 * insert_logits121  + ol_peg2 * insert_logits221

        return torch.cat([insert_logits0_1, insert_logits1_1, insert_logits2_1], dim = 1)

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

class PosErr_DetectionTransformer(Proto_Macromodel):
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

    def calc_params(self, rep, meanmodel, hetmodel, hom_prec):
        batch_size = rep.size(0)

        mean = meanmodel(rep)
        
        new_rep = torch.cat([rep, mean], dim = 1)
        
        het_vect = hetmodel(new_rep).transpose(0,1)

        tril_indices = torch.tril_indices(row=self.pose_size, col=self.pose_size)

        het_tril = torch.zeros((self.pose_size, self.pose_size, batch_size)).to(self.device)

        het_tril[tril_indices[0], tril_indices[1]] = het_vect

        het_prec = prec_mult(het_tril)

        hom_prec = prec_single(hom_prec()).unsqueeze(0).repeat_interleave(batch_size, 0)  

        prec = het_prec + hom_prec

        return mean, prec

    def get_data(self, state, pose_vect, forces, hole_type, model_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, state_enc, pos_err_transdec,\
         pos_err_model0, pos_err_prechom0, pos_err_prechet0,\
          pos_err_model1, pos_err_prechom1, pos_err_prechet1,\
           pos_err_model2, pos_err_prechom2, pos_err_prechet2 = model_tuple #pos_err_prec = model_tuple # 

        frc_encs_unshaped = self.get_frc(forces, frc_enc)
        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))#.contiguous()

        state_encs_unshaped = state_enc(state)
        state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))#.contiguous()

        states = torch.cat([state_encs, frc_encs], dim = 2).transpose(0,1)#.contiguous()

        if padding_masks is None:
            rep_delta = torch.max(pos_err_transdec(states, states), 0)[0]
        else:
            rep_delta = torch.max(pos_err_transdec(states, states, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks), 0)[0]

        rep = torch.cat([pose_vect, rep_delta], dim = 1)

        pos_err_mean0, pos_err_prec0 = self.calc_params(rep, pos_err_model0, pos_err_prechet0, pos_err_prechom0)
        pos_err_mean1, pos_err_prec1 = self.calc_params(rep, pos_err_model1, pos_err_prechet1, pos_err_prechom1)
        pos_err_mean2, pos_err_prec2 = self.calc_params(rep, pos_err_model2, pos_err_prechet2, pos_err_prechom2)

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

        pos_err_mean01, pos_err_prec01 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[0],\
         batch_size, sequence_size, padding_masks)

        pos_err_mean11, pos_err_prec11 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[1],\
         batch_size, sequence_size, padding_masks)

        pos_err_mean21, pos_err_prec21 = self.get_data(prestate_reshaped, pose_vect, forces_reshaped, hole_type, self.ensemble_list[2],\
         batch_size, sequence_size, padding_masks)

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

        # print(prec_dets.size())

        if self.frc_enc21.training:
            self.train_dets()[0] = (1 - self.train_lr) * self.train_dets()[0] + self.train_lr * cov_dets.min()
            self.train_dets()[1] = (1 - self.train_lr) * self.train_dets()[1] + self.train_lr * cov_dets.mean()       
            self.train_dets()[2] = (1 - self.train_lr) * self.train_dets()[2] + self.train_lr * cov_dets.max()
        else:
            self.val_dets()[0] = (1 - self.val_lr) * self.val_dets()[0] + self.val_lr * cov_dets.min()
            self.val_dets()[1] = (1 - self.val_lr) * self.val_dets()[1] + self.val_lr * cov_dets.mean()       
            self.val_dets()[2] = (1 - self.val_lr) * self.val_dets()[2] + self.val_lr * cov_dets.max()

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

        self.pose_size = 3
        self.z_dim = 48
        self.num_options = num_options

        self.ensemble_list = []

        # first number indicates index in peg vector, second number indicates number in ensemble
        # Network 001
        self.expand_state001 = FCN(save_folder + "_expand_state001", load_folder + "_expand_state001",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred001 = ResNetFCN(save_folder + "_pos_err_pred001", load_folder + "_pos_err_pred001",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet001 = ResNetFCN(save_folder + "_pos_err_prechet001", load_folder + "_pos_err_prechet001",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom001 = Params(save_folder + "_pos_err_prechom001", load_folder + "_pos_err_prechom001", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred001)
        self.model_list.append(self.expand_state001)
        self.model_list.append(self.pos_err_prechet001)
        self.model_list.append(self.pos_err_prechom001)

        # Network 011
        self.expand_state011 = FCN(save_folder + "_expand_state011", load_folder + "_expand_state011",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred011 = ResNetFCN(save_folder + "_pos_err_pred011", load_folder + "_pos_err_pred011",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet011 = ResNetFCN(save_folder + "_pos_err_prechet011", load_folder + "_pos_err_prechet011",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom011 = Params(save_folder + "_pos_err_prechom011", load_folder + "_pos_err_prechom011", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred011)
        self.model_list.append(self.expand_state011)
        self.model_list.append(self.pos_err_prechet011)
        self.model_list.append(self.pos_err_prechom011)

        # Network 021
        self.expand_state021 = FCN(save_folder + "_expand_state021", load_folder + "_expand_state021",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred021 = ResNetFCN(save_folder + "_pos_err_pred021", load_folder + "_pos_err_pred021",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet021 = ResNetFCN(save_folder + "_pos_err_prechet021", load_folder + "_pos_err_prechet021",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom021 = Params(save_folder + "_pos_err_prechom021", load_folder + "_pos_err_prechom021", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred021)
        self.model_list.append(self.expand_state021)
        self.model_list.append(self.pos_err_prechet021)
        self.model_list.append(self.pos_err_prechom021) 

        # Network 101
        self.expand_state101 = FCN(save_folder + "_expand_state101", load_folder + "_expand_state101",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred101 = ResNetFCN(save_folder + "_pos_err_pred101", load_folder + "_pos_err_pred101",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet101 = ResNetFCN(save_folder + "_pos_err_prechet101", load_folder + "_pos_err_prechet101",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom101 = Params(save_folder + "_pos_err_prechom101", load_folder + "_pos_err_prechom101", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred101)
        self.model_list.append(self.expand_state101)
        self.model_list.append(self.pos_err_prechet101)
        self.model_list.append(self.pos_err_prechom101) 

        # Network 111
        self.expand_state111 = FCN(save_folder + "_expand_state111", load_folder + "_expand_state111",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred111 = ResNetFCN(save_folder + "_pos_err_pred111", load_folder + "_pos_err_pred111",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet111 = ResNetFCN(save_folder + "_pos_err_prechet111", load_folder + "_pos_err_prechet111",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom111 = Params(save_folder + "_pos_err_prechom111", load_folder + "_pos_err_prechom111", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred111)
        self.model_list.append(self.expand_state111)
        self.model_list.append(self.pos_err_prechet111)
        self.model_list.append(self.pos_err_prechom111) 

        # Network 121
        self.expand_state121 = FCN(save_folder + "_expand_state121", load_folder + "_expand_state121",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred121 = ResNetFCN(save_folder + "_pos_err_pred121", load_folder + "_pos_err_pred121",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet121 = ResNetFCN(save_folder + "_pos_err_prechet121", load_folder + "_pos_err_prechet121",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom121 = Params(save_folder + "_pos_err_prechom121", load_folder + "_pos_err_prechom121", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred121)
        self.model_list.append(self.expand_state121)
        self.model_list.append(self.pos_err_prechet121)
        self.model_list.append(self.pos_err_prechom121) 

        # Network 201
        self.expand_state201 = FCN(save_folder + "_expand_state201", load_folder + "_expand_state201",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred201 = ResNetFCN(save_folder + "_pos_err_pred201", load_folder + "_pos_err_pred201",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet201 = ResNetFCN(save_folder + "_pos_err_prechet201", load_folder + "_pos_err_prechet201",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom201 = Params(save_folder + "_pos_err_prechom201", load_folder + "_pos_err_prechom201", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred201)
        self.model_list.append(self.expand_state201)
        self.model_list.append(self.pos_err_prechet201)
        self.model_list.append(self.pos_err_prechom201) 

        # Network 211
        self.expand_state211 = FCN(save_folder + "_expand_state211", load_folder + "_expand_state211",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred211 = ResNetFCN(save_folder + "_pos_err_pred211", load_folder + "_pos_err_pred211",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet211 = ResNetFCN(save_folder + "_pos_err_prechet211", load_folder + "_pos_err_prechet211",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom211 = Params(save_folder + "_pos_err_prechom211", load_folder + "_pos_err_prechom211", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred211)
        self.model_list.append(self.expand_state211)
        self.model_list.append(self.pos_err_prechet211)
        self.model_list.append(self.pos_err_prechom211)

        # Network 221
        self.expand_state221 = FCN(save_folder + "_expand_state221", load_folder + "_expand_state221",\
            2 * self.pose_size, self.z_dim, 1, device = self.device)

        self.pos_err_pred221 = ResNetFCN(save_folder + "_pos_err_pred221", load_folder + "_pos_err_pred221",\
            self.z_dim, self.pose_size, 3, device = self.device)

        self.pos_err_prechet221 = ResNetFCN(save_folder + "_pos_err_prechet221", load_folder + "_pos_err_prechet221",\
            self.z_dim + self.pose_size, sum(range(self.pose_size+1)), 4, device = self.device)

        self.pos_err_prechom221 = Params(save_folder + "_pos_err_prechom221", load_folder + "_pos_err_prechom221", (self.pose_size, self.pose_size), device = self.device)
 
        self.model_list.append(self.pos_err_pred221)
        self.model_list.append(self.expand_state221)
        self.model_list.append(self.pos_err_prechet221)
        self.model_list.append(self.pos_err_prechom221) 


        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def calc_params(self, rep, meanmodel, hetmodel, hom_prec):
        batch_size = rep.size(0)

        mean = meanmodel(rep)
        
        new_rep = torch.cat([rep, mean], dim = 1)
        
        het_vect = hetmodel(new_rep).transpose(0,1)

        tril_indices = torch.tril_indices(row=self.pose_size, col=self.pose_size)

        het_tril = torch.zeros((self.pose_size, self.pose_size, batch_size)).to(self.device)

        het_tril[tril_indices[0], tril_indices[1]] = het_vect

        het_prec = prec_mult(het_tril)

        hom_prec = prec_single(hom_prec()).unsqueeze(0).repeat_interleave(batch_size, 0)  

        prec = het_prec + hom_prec

        return mean, prec

    def get_pred(self, init_pos, command_pos, peg_type, hole_type):
        state = torch.cat([init_pos, command_pos], dim = 1)

        pos_err_mean001, pos_err_prec001 = self.calc_params(self.expand_state001(state), self.pos_err_pred001, self.pos_err_prechet001, self.pos_err_prechom001)
        pos_err_mean101, pos_err_prec101 = self.calc_params(self.expand_state101(state), self.pos_err_pred101, self.pos_err_prechet101, self.pos_err_prechom101)
        pos_err_mean201, pos_err_prec201 = self.calc_params(self.expand_state201(state), self.pos_err_pred201, self.pos_err_prechet201, self.pos_err_prechom201)
        pos_err_mean011, pos_err_prec011 = self.calc_params(self.expand_state011(state), self.pos_err_pred011, self.pos_err_prechet011, self.pos_err_prechom011)
        pos_err_mean111, pos_err_prec111 = self.calc_params(self.expand_state111(state), self.pos_err_pred111, self.pos_err_prechet111, self.pos_err_prechom111)
        pos_err_mean211, pos_err_prec211 = self.calc_params(self.expand_state211(state), self.pos_err_pred211, self.pos_err_prechet211, self.pos_err_prechom211)
        pos_err_mean021, pos_err_prec021 = self.calc_params(self.expand_state021(state), self.pos_err_pred021, self.pos_err_prechet021, self.pos_err_prechom021)
        pos_err_mean121, pos_err_prec121 = self.calc_params(self.expand_state121(state), self.pos_err_pred121, self.pos_err_prechet121, self.pos_err_prechom121)
        pos_err_mean221, pos_err_prec221 = self.calc_params(self.expand_state221(state), self.pos_err_pred221, self.pos_err_prechet221, self.pos_err_prechom221)

        om_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        om_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        om_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        oc_hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)
        oc_hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)
        oc_hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)

        om_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        om_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        om_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_mean001.size(1), dim=1)
        oc_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)
        oc_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)
        oc_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(pos_err_prec001.size(1), dim=1).unsqueeze(2).repeat_interleave(pos_err_prec001.size(2), dim=2)

        pos_err_mean01 = om_hole0 * pos_err_mean001 + om_hole1 * pos_err_mean101 + om_hole2 * pos_err_mean201
        pos_err_prec01 = oc_hole0 * pos_err_prec001 + oc_hole1 * pos_err_prec101 + oc_hole2 * pos_err_prec201

        pos_err_mean11 = om_hole0 * pos_err_mean011 + om_hole1 * pos_err_mean111 + om_hole2 * pos_err_mean211
        pos_err_prec11 = oc_hole0 * pos_err_prec011 + oc_hole1 * pos_err_prec111 + oc_hole2 * pos_err_prec211

        pos_err_mean21 = om_hole0 * pos_err_mean021 + om_hole1 * pos_err_mean121 + om_hole2 * pos_err_mean221
        pos_err_prec21 = oc_hole0 * pos_err_prec021 + oc_hole1 * pos_err_prec121 + oc_hole2 * pos_err_prec221

        pos_err_mean1 = om_peg0 * pos_err_mean01 + om_peg1 * pos_err_mean11 + om_peg2 * pos_err_mean21
        pos_err_prec1 = oc_peg0 * pos_err_prec01 + oc_peg1 * pos_err_prec11 + oc_peg2 * pos_err_prec21

        return pos_err_mean1, pos_err_prec1

    def forward(self, input_dict):
        init_pos = input_dict["errorinit_pos"].to(self.device)
        command_pos = input_dict["command_pos"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        hole_type = input_dict["hole_type"].to(self.device)

        pos_err_mean, pos_err_prec = self.get_pred(init_pos, command_pos, peg_type, hole_type)

        return {
            'pos_err_params': (pos_err_mean, pos_err_prec),
        }

    def process(self, init_pos, command_pos, pos_err, peg_type, hole_type):
        return self.get_pred(init_pos, command_pos, peg_type, hole_type)

# class Options_ClassifierTransformer(Proto_Macromodel):
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
#         self.state_size = 64
#         self.action_size = 16
#         self.encoded_size = self.state_size - self.frc_enc_size
#         self.use_fft = use_fft
#         self.min_steps = min_steps
#         self.macro_action_size = 2 * self.pose_size 

#         self.prestate_size = self.proprio_size + self.contact_size

#         self.ensemble_list = []
#         self.num_el = 1
#         self.num_tl = 3
#         self.num_cl = 1
#         self.num_gl = 1
#         self.train_bool = True
#         # first number indicates index in peg vector, second number indicates number in ensemble

#         self.confusion_matrix = Params(save_folder + "_conf_mat", load_folder + "_conf_mat", (self.num_options, self.num_options, self.num_options), device = self.device)
#         self.confusion_matrix().data[:] = 0

#         self.model_list.append(self.confusion_matrix)
#         # Network 0
#         # Encoder model for peg 0
#         self.frc_enc0 = CONV2DN(save_folder + "_fft_enc0", load_folder + "_fft_enc0",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc0 = ResNetFCN(save_folder + "_state_enc0", load_folder + "_state_enc0",\
#             self.prestate_size, self.encoded_size, self.num_el, device = self.device)

#         self.action_enc0 = ResNetFCN(save_folder + "_action_enc0", load_folder + "_action_enc0",\
#             self.pose_size, self.action_size, self.num_el, device = self.device)

#         self.action_enc0a = ResNetFCN(save_folder + "_action_enc0a", load_folder + "_action_enc0a",\
#             self.macro_action_size + self.num_options, self.action_size, self.num_el, device = self.device)

#         self.state_transdec0 = Transformer_Decoder(save_folder + "_state_transdec0", load_folder + "_state_transdec0",\
#          self.state_size, self.num_tl, device = self.device)

#         self.action_transdec0 = Transformer_Decoder(save_folder + "_action_transdec0", load_folder + "_action_transdec0",\
#          self.action_size, self.num_tl, device = self.device)

#         #### Discriminative model for peg 0
#         self.options_class0 = ResNetFCN(save_folder + "_options_class0", load_folder + "_options_class0",\
#             self.state_size + self.action_size , self.num_options, self.num_cl, device = self.device)

#         ### Generative model for peg 0
#         self.options_gmean00 = ResNetFCN(save_folder + "_options_gmean00", load_folder + "_options_gmean00",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar00 = ResNetFCN(save_folder + "_options_gvar00", load_folder + "_options_gvar00",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean10 = ResNetFCN(save_folder + "_options_gmean10", load_folder + "_options_gmean10",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar10 = ResNetFCN(save_folder + "_options_gvar10", load_folder + "_options_gvar10",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean20 = ResNetFCN(save_folder + "_options_gmean20", load_folder + "_options_gmean20",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar20 = ResNetFCN(save_folder + "_options_gvar20", load_folder + "_options_gvar20",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.model_list.append(self.frc_enc0) 
#         self.model_list.append(self.state_enc0)
#         self.model_list.append(self.action_enc0)
#         self.model_list.append(self.action_enc0a)
#         self.model_list.append(self.state_transdec0)
#         self.model_list.append(self.action_transdec0) 

#         self.model_list.append(self.options_class0)

#         self.model_list.append(self.options_gmean00)
#         self.model_list.append(self.options_gvar00)
#         self.model_list.append(self.options_gmean10)
#         self.model_list.append(self.options_gvar10)
#         self.model_list.append(self.options_gmean20)
#         self.model_list.append(self.options_gvar20)

#         self.ensemble_list.append(((self.frc_enc0, self.state_enc0, self.action_enc0, self.action_enc0a, self.state_transdec0, self.action_transdec0), (self.options_class0),\
#             (self.options_gmean00, self.options_gvar00, self.options_gmean10, self.options_gvar10, self.options_gmean20, self.options_gvar20))) 

#         # Network 1
#         # Encoder model for peg 1
#         self.frc_enc1 = CONV2DN(save_folder + "_fft_enc1", load_folder + "_fft_enc1",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc1 = ResNetFCN(save_folder + "_state_enc1", load_folder + "_state_enc1",\
#             self.prestate_size, self.encoded_size, self.num_el, device = self.device)

#         self.action_enc1 = ResNetFCN(save_folder + "_action_enc1", load_folder + "_action_enc1",\
#             self.pose_size, self.action_size, self.num_el, device = self.device)

#         self.action_enc1a = ResNetFCN(save_folder + "_action_enc1a", load_folder + "_action_enc1a",\
#             self.macro_action_size + self.num_options, self.action_size, self.num_el, device = self.device)

#         self.state_transdec1 = Transformer_Decoder(save_folder + "_state_transdec1", load_folder + "_state_transdec1",\
#          self.state_size, self.num_tl, device = self.device)

#         self.action_transdec1 = Transformer_Decoder(save_folder + "_action_transdec1", load_folder + "_action_transdec1",\
#          self.action_size, self.num_tl, device = self.device)

#         #### Discriminative model for peg 1
#         self.options_class1 = ResNetFCN(save_folder + "_options_class1", load_folder + "_options_class1",\
#             self.state_size + self.action_size , self.num_options, self.num_cl, device = self.device)

#         ### Generative model for peg 0
#         self.options_gmean01 = ResNetFCN(save_folder + "_options_gmean01", load_folder + "_options_gmean01",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar01 = ResNetFCN(save_folder + "_options_gvar01", load_folder + "_options_gvar01",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean11 = ResNetFCN(save_folder + "_options_gmean11", load_folder + "_options_gmean11",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar11 = ResNetFCN(save_folder + "_options_gvar11", load_folder + "_options_gvar11",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean21 = ResNetFCN(save_folder + "_options_gmean21", load_folder + "_options_gmean21",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar21 = ResNetFCN(save_folder + "_options_gvar21", load_folder + "_options_gvar21",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.model_list.append(self.frc_enc1) 
#         self.model_list.append(self.state_enc1)
#         self.model_list.append(self.action_enc1)
#         self.model_list.append(self.action_enc1a)
#         self.model_list.append(self.state_transdec1)
#         self.model_list.append(self.action_transdec1) 

#         self.model_list.append(self.options_class1)

#         self.model_list.append(self.options_gmean01)
#         self.model_list.append(self.options_gvar01)
#         self.model_list.append(self.options_gmean11)
#         self.model_list.append(self.options_gvar11)
#         self.model_list.append(self.options_gmean21)
#         self.model_list.append(self.options_gvar21)

#         self.ensemble_list.append(((self.frc_enc1, self.state_enc1, self.action_enc1, self.action_enc1a, self.state_transdec1, self.action_transdec1), (self.options_class1),\
#             (self.options_gmean01, self.options_gvar01, self.options_gmean11, self.options_gvar11, self.options_gmean21, self.options_gvar21)))  

#         # Network 2
#         # Encoder model for peg 2
#         self.frc_enc2 = CONV2DN(save_folder + "_fft_enc2", load_folder + "_fft_enc2",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc2 = ResNetFCN(save_folder + "_state_enc2", load_folder + "_state_enc2",\
#             self.prestate_size, self.encoded_size, self.num_el, device = self.device)

#         self.action_enc2 = ResNetFCN(save_folder + "_action_enc2", load_folder + "_action_enc2",\
#             self.pose_size, self.action_size, self.num_el, device = self.device)

#         self.action_enc2a = ResNetFCN(save_folder + "_action_enc2a", load_folder + "_action_enc2a",\
#             self.macro_action_size + self.num_options, self.action_size, self.num_el, device = self.device)

#         self.state_transdec2 = Transformer_Decoder(save_folder + "_state_transdec2", load_folder + "_state_transdec2",\
#          self.state_size, self.num_tl, device = self.device)

#         self.action_transdec2 = Transformer_Decoder(save_folder + "_action_transdec2", load_folder + "_action_transdec2",\
#          self.action_size, self.num_tl, device = self.device)

#         #### Discriminative model for peg 2
#         self.options_class2 = ResNetFCN(save_folder + "_options_class2", load_folder + "_options_class2",\
#             self.state_size + self.action_size , self.num_options, self.num_cl, device = self.device)

#         ### Generative model for peg 0
#         self.options_gmean02 = ResNetFCN(save_folder + "_options_gmean02", load_folder + "_options_gmean02",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar02 = ResNetFCN(save_folder + "_options_gvar02", load_folder + "_options_gvar02",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean12 = ResNetFCN(save_folder + "_options_gmean12", load_folder + "_options_gmean12",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar12 = ResNetFCN(save_folder + "_options_gvar12", load_folder + "_options_gvar12",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gmean22 = ResNetFCN(save_folder + "_options_gmean22", load_folder + "_options_gmean22",\
#             self.action_size, self.state_size, self.num_gl, device = self.device)

#         self.options_gvar22 = ResNetFCN(save_folder + "_options_gvar22", load_folder + "_options_gvar22",\
#             self.action_size + self.state_size, self.state_size, self.num_gl, device = self.device)

#         self.model_list.append(self.frc_enc2) 
#         self.model_list.append(self.state_enc2)
#         self.model_list.append(self.action_enc2)
#         self.model_list.append(self.action_enc2a)
#         self.model_list.append(self.state_transdec2)
#         self.model_list.append(self.action_transdec2) 

#         self.model_list.append(self.options_class2)

#         self.model_list.append(self.options_gmean02)
#         self.model_list.append(self.options_gvar02)
#         self.model_list.append(self.options_gmean12)
#         self.model_list.append(self.options_gvar12)
#         self.model_list.append(self.options_gmean22)
#         self.model_list.append(self.options_gvar22)

#         self.ensemble_list.append(((self.frc_enc2, self.state_enc2, self.action_enc2, self.action_enc2a, self.state_transdec2, self.action_transdec2), (self.options_class2),\
#             (self.options_gmean02, self.options_gvar02, self.options_gmean12, self.options_gvar12, self.options_gmean22, self.options_gvar22))) 

#         if info_flow[model_name]["model_folder"] != "":
#             self.load(info_flow[model_name]["epoch_num"])

#     def get_frc(self, force, model):
#         fft = torch.rfft(force, 2, normalized=False, onesided=True)
#         frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
#         return frc_enc

#     def get_enc(self, states, forces, actions, macro_action, hole_type, encoder_tuple, batch_size, sequence_size, padding_masks = None):
#         frc_enc, state_enc, action_enc, action_enc1, state_transdec, action_transdec = encoder_tuple

#         frc_encs_unshaped = self.get_frc(forces, frc_enc)

#         frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))

#         state_encs_unshaped = state_enc(states)
#         state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))

#         states_cat = torch.cat([state_encs, frc_encs], dim = 2).transpose(0,1)

#         action_encs_unshaped = action_enc(actions)
#         action_encs = torch.reshape(action_encs_unshaped, (batch_size, sequence_size, self.action_size)).transpose(0,1)

#         # max_len = states_cat.size(0)

#         # mask = (torch.triu(torch.ones(max_len, max_len).to(states_cat.device)) == 1).transpose(0, 1)
#         # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

#         if padding_masks is None:
#             seq_encs = state_transdec(states_cat, states_cat).max(0)[0]
#             act_encs0 = action_transdec(action_encs, action_encs).max(0)[0]
#         else:
#             seq_encs = state_transdec(states_cat, states_cat, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks).max(0)[0]
#             act_encs0 = action_transdec(action_encs, action_encs, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks).max(0)[0]

#         act_encs1 = action_enc1(torch.cat([macro_action, hole_type], dim = 1))

#         sample = torch.tensor(np.random.binomial(1, 0.5, size = act_encs1.size(0))).float().to(seq_encs.device).unsqueeze(1).repeat_interleave(act_encs1.size(1), dim = 1)

#         act_encs = act_encs0 #sample * act_encs0 + (1 - sample) * act_encs1

#         return seq_encs, act_encs, act_encs0, act_encs1

#     def get_params(self, action_enc, hole_type, gen_tuple):
#         gmean0, gvar0, gmean1, gvar1, gmean2, gvar2 = gen_tuple

#         mean0 = gmean0(action_enc)
#         var0 = gvar0(torch.cat([action_enc, mean0], dim = 1)).pow(2) + 0.001

#         mean1 = gmean1(action_enc)
#         var1 = gvar1(torch.cat([action_enc, mean1], dim = 1)).pow(2) + 0.001

#         mean2 = gmean2(action_enc)
#         var2 = gvar2(torch.cat([action_enc, mean2], dim = 1)).pow(2) + 0.001

#         hole0 = hole_type[:,0].unsqueeze(1).repeat_interleave(mean0.size(1), dim=1)
#         hole1 = hole_type[:,1].unsqueeze(1).repeat_interleave(mean0.size(1), dim=1)
#         hole2 = hole_type[:,2].unsqueeze(1).repeat_interleave(mean0.size(1), dim=1)

#         mean = hole0 * mean0 + hole1 * mean1 + hole2 * mean2
#         var = hole0 * var0 + hole1 * var1 + hole2 * var2

#         return (mean, var)

#     def get_data(self, states, forces, actions, macro_action, hole_type, model_tuple, batch_size, sequence_size, padding_masks = None):
#         encoder_tuple, options_class, gen_tuple = model_tuple #origin_cov = model_tuple # 

#         seq_enc, act_encs, act_encs0, act_encs1 = self.get_enc(states, forces, actions, macro_action, hole_type, encoder_tuple, batch_size, sequence_size, padding_masks = padding_masks)

#         options_logits = options_class(torch.cat([seq_enc, act_encs], dim = 1))

#         seq_mean, seq_var = self.get_params(act_encs, hole_type, gen_tuple)

#         return options_logits, seq_enc, seq_mean, seq_var, act_encs0, act_encs1

#     def get_logits(self, proprio_diffs, contact_diffs, forces, actions, macro_action, peg_type, hole_type, padding_masks = None):
#         batch_size = proprio_diffs.size(0)
#         sequence_size = proprio_diffs.size(1)

#         forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

#         prestate = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2)], dim = 2)
#         prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2)))#.contiguous()

#         actions_reshaped = torch.reshape(actions, (actions.size(0) * actions.size(1), actions.size(2)))

#         options_logits0, seq_enc0, seq_mean0, seq_var0, act_encs00, act_encs10 = self.get_data(prestate_reshaped, forces_reshaped,\
#          actions_reshaped, macro_action, hole_type, self.ensemble_list[0], batch_size, sequence_size, padding_masks)

#         options_logits1, seq_enc1, seq_mean1, seq_var1, act_encs01, act_encs11 = self.get_data(prestate_reshaped, forces_reshaped,\
#          actions_reshaped, macro_action, hole_type, self.ensemble_list[1], batch_size, sequence_size, padding_masks)

#         options_logits2, seq_enc2, seq_mean2, seq_var2, act_encs02, act_encs12 = self.get_data(prestate_reshaped, forces_reshaped,\
#          actions_reshaped, macro_action, hole_type, self.ensemble_list[2], batch_size, sequence_size, padding_masks)
        
#         ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits0.size(1), dim=1)
#         ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits0.size(1), dim=1)
#         ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits0.size(1), dim=1)

#         seq_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(seq_enc0.size(1), dim=1)
#         seq_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(seq_enc0.size(1), dim=1)
#         seq_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(seq_enc0.size(1), dim=1)

#         act_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(act_encs00.size(1), dim=1)
#         act_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(act_encs00.size(1), dim=1)
#         act_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(act_encs00.size(1), dim=1)

#         option_logits = ol_peg0 * options_logits0 + ol_peg1 * options_logits1 + ol_peg2 * options_logits2
#         seq_enc = seq_peg0 * seq_enc0 + seq_peg1 * seq_enc1 + seq_peg2 * seq_enc2
#         seq_mean = seq_peg0 * seq_mean0 + seq_peg1 * seq_mean1 + seq_peg2 * seq_mean2
#         seq_var = seq_peg0 * seq_var0 + seq_peg1 * seq_var1 + seq_peg2 * seq_var2
#         act_encs0 = act_peg0 * act_encs00 + act_peg1 * act_encs01 + act_peg2 * act_encs02 
#         act_encs1 = act_peg0 * act_encs10 + act_peg1 * act_encs11 + act_peg2 * act_encs12 

#         return option_logits, seq_enc, seq_mean, seq_var, act_encs0, act_encs1

#     def forward(self, input_dict):
#         proprio_diffs = input_dict["proprio_diff"].to(self.device)
#         actions = input_dict["action"].to(self.device)
#         macro_action = input_dict["macro_action"].to(self.device)
#         forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
#         contact_diffs = input_dict["contact_diff"].to(self.device)
#         peg_type = input_dict["peg_type"].to(self.device)
#         hole_type = input_dict["hole_type"].to(self.device)
#         padding_masks = input_dict["padding_mask"].to(self.device)

#         option_logits, seq_enc, seq_mean, seq_var, act_encs0, act_encs1 = self.get_logits(proprio_diffs, contact_diffs,\
#          forces, actions, macro_action, peg_type, hole_type, padding_masks)

#         if self.confusion_matrix.training != self.train_bool and not self.confusion_matrix.training: # switched to validation
#             self.train_bool = self.confusion_matrix.training
#             self.confusion_matrix().data[:] = 0
#             option_probs = F.softmax(option_logits, dim = 1)

#             for i in range(option_probs.size(0)):
#                 h = peg_type[i].max(0)[1]
#                 row= option_probs[i].max(0)[1]
#                 col = hole_type[i].max(0)[1]
#                 self.confusion_matrix().data[h, row, col] += 1
            
#         elif self.confusion_matrix.training == self.train_bool and not self.confusion_matrix.training: # in validation
#             option_probs = F.softmax(option_logits, dim = 1)

#             for i in range(option_probs.size(0)):
#                 h = peg_type[i].max(0)[1]
#                 row = option_probs[i].max(0)[1]
#                 col = hole_type[i].max(0)[1]
#                 self.confusion_matrix()[h, row, col] += 1

#         elif self.confusion_matrix.training != self.train_bool and self.confusion_matrix.training: #switched to training
#             self.train_bool = self.confusion_matrix.training
#             print("Current Confusion Matrix")
#             mat =self.confusion_matrix() / self.confusion_matrix().sum(2).unsqueeze(2).repeat_interleave(self.num_options, dim = 2)
#             for i in range(self.num_options):
#                 if i == 0:
#                     print("Cross Peg")
#                 elif i == 1:
#                     print("Rect Peg")
#                 else:
#                     print("Square Peg")

#                 print(mat[i].detach().cpu().numpy())

#         return {
#             'options_class': option_logits,
#             'state_enc': seq_enc,
#             'state_params': (seq_mean, seq_var),
#             'act_encs0': act_encs0,
#             'act_encs1': act_encs1,
#         }

#     def process(self, input_dict):
#         proprios = input_dict["proprio"].to(self.device)
#         forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
#         contacts = input_dict["contact"].to(self.device)
#         actions = input_dict["action"].to(self.device)[:, :-1]
#         peg_type = input_dict["peg_type"].to(self.device)
#         # hole_type = input_dict["hole_type"].to(self.device)
#         macro_action = input_dict["macro_action"].to(self.device)


#         proprio_diffs = proprios[:,1:] - proprios[:, :-1]
#         contact_diffs = contacts[:,1:] - contacts[:, :-1]
#         force_clipped = forces[:,1:]

#         logprobs = torch.zeros_like(peg_type)

#         # for i in range(self.num_options):
#         #     hole_type = torch.zeros_like(peg_type)
#         #     hole_type[:,i] = 1.0
#         #     options_logits, seq_enc, seq_mean, seq_var, act_encs0, act_encs1 = self.get_logits(proprio_diffs, contact_diffs,\
#         #      force_clipped, actions, macro_action, peg_type, hole_type)

#         #     logprobs[:,i] = log_normal(seq_enc, seq_mean, seq_var)


#         hole_type = input_dict["hole_type"].to(self.device)
#         options_logits, seq_enc, seq_mean, seq_var, act_encs0, act_encs1 = self.get_logits(proprio_diffs, contact_diffs,\
#          force_clipped, actions, macro_action, peg_type, hole_type)

#         option_probs = F.softmax(options_logits, dim = 1)
#         logprobs = torch.zeros_like(option_probs)

#         for i in range(logprobs.shape[0]):
#             h = peg_type[i].max(0)[1]
#             col = option_probs[i].max(0)[1]
#             counts = self.confusion_matrix()[h,:,col]

#             counts = counts + 5
#             probs = counts / sum(counts)
#             # print(probs)
#             logprobs[i] = torch.log(probs)         

#         print(logprobs)
#         return logprobs
        
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

# class Options_ClassifierTransformer(Proto_Macromodel):
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
#         # Encoder model for peg 1
#         self.frc_enc01 = CONV2DN(save_folder + "_fft_enc01", load_folder + "_fft_enc01",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc01 = ResNetFCN(save_folder + "_state_enc01", load_folder + "_state_enc01",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.options_transdec01 = Transformer_Decoder(save_folder + "_options_transdec01", load_folder + "_options_transdec01",\
#          self.state_size, 6, device = self.device)

#         #### Discriminative model for peg 1
#         self.options_class01 = ResNetFCN(save_folder + "_options_class01", load_folder + "_options_class01",\
#             self.state_size, self.num_options, 3, device = self.device)

#         ### Generative model for peg 1
#         self.options_gmean001 = ResNetFCN(save_folder + "_options_gmean001", load_folder + "_options_gmean001",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.options_gvar001 = ResNetFCN(save_folder + "_options_gvar001", load_folder + "_options_gvar001",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.options_gmean101 = ResNetFCN(save_folder + "_options_gmean101", load_folder + "_options_gmean101",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.options_gvar101 = ResNetFCN(save_folder + "_options_gvar101", load_folder + "_options_gvar101",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.options_gmean201 = ResNetFCN(save_folder + "_options_gmean201", load_folder + "_options_gmean201",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.options_gvar201 = ResNetFCN(save_folder + "_options_gvar201", load_folder + "_options_gvar201",\
#             2 * self.pos_size, self.state_size, 3, device = self.device)

#         self.model_list.append(self.frc_enc01) 
#         self.model_list.append(self.state_enc01) 
#         self.model_list.append(self.options_transdec01) 
#         self.model_list.append(self.options_class01)
#         self.model_list.append(self.options_gen001)
#         self.model_list.append(self.options_gen101)
#         self.model_list.append(self.options_gen201)

#         self.ensemble_list.append(((self.frc_enc01, self.state_enc01, self.options_transdec01), (self.options_class01),\
#             (self.options_gmean001, self.options_gvar001, self.options_gmean101, self.options_gvar101, self.options_gmean201, self.options_gvar201))) 

#         # Network 11
#         self.frc_enc11 = CONV2DN(save_folder + "_fft_enc11", load_folder + "_fft_enc11",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc11 = ResNetFCN(save_folder + "_state_enc11", load_folder + "_state_enc11",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.options_transdec11 = Transformer_Decoder(save_folder + "_options_transdec11", load_folder + "_options_transdec11",\
#          self.state_size, 6, device = self.device)

#         self.options_class11 = ResNetFCN(save_folder + "_options_class11", load_folder + "_options_class11",\
#             self.state_size, self.num_options, 3, device = self.device)

#         self.model_list.append(self.frc_enc11) 
#         self.model_list.append(self.state_enc11) 
#         self.model_list.append(self.options_transdec11) 
#         self.model_list.append(self.options_class11) 

#         self.ensemble_list.append((self.frc_enc11, self.state_enc11, self.options_transdec11, self.options_class11)) #self.origin_cov11)) #, 

#         # Network 21
#         self.frc_enc21 = CONV2DN(save_folder + "_fft_enc21", load_folder + "_fft_enc21",\
#          (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device)  #10 HZ - 27 # 2 HZ - 126

#         self.state_enc21 = ResNetFCN(save_folder + "_state_enc21", load_folder + "_state_enc21",\
#             self.prestate_size, self.encoded_size, 3, device = self.device)

#         self.options_transdec21 = Transformer_Decoder(save_folder + "_options_transdec21", load_folder + "_options_transdec21",\
#          self.state_size, 6, device = self.device)

#         self.options_class21 = ResNetFCN(save_folder + "_options_class21", load_folder + "_options_class21",\
#             self.state_size, self.num_options, 3, device = self.device)

#         self.model_list.append(self.frc_enc21) 
#         self.model_list.append(self.state_enc21) 
#         self.model_list.append(self.options_transdec21) 
#         self.model_list.append(self.options_class21) 

#         self.ensemble_list.append((self.frc_enc21, self.state_enc21, self.options_transdec21, self.options_class21)) # self.origin_cov21)) #, 

#         if info_flow[model_name]["model_folder"] != "":
#             self.load(info_flow[model_name]["epoch_num"])

#     def get_frc(self, force, model):
#         fft = torch.rfft(force, 2, normalized=False, onesided=True)
#         frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
#         # print("Force size: ", force.size())
#         # print(fft.size())
#         # frc_enc = self.frc_enc(fft)
#         return frc_enc

#     def get_data(self, state, forces, model_tuple, batch_size, sequence_size, padding_masks = None, lengths = None):
#         frc_enc, state_enc, options_transdec, options_class = model_tuple #origin_cov = model_tuple # 

#         frc_encs_unshaped = self.get_frc(forces, frc_enc)
#         frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))#.contiguous()

#         state_encs_unshaped = state_enc(state)
#         state_encs = torch.reshape(state_encs_unshaped, (batch_size, sequence_size, self.encoded_size))#.contiguous()

#         states = torch.cat([state_encs, frc_encs], dim = 2).transpose(0,1)#.contiguous()

#         # final_state = states[lengths, torch.arange(states.size(1))].unsqueeze(0)

#         if padding_masks is None:
#             options_logits = options_class(torch.max(options_transdec(states, states), 0)[0])
#         else:
#             options_logits = options_class(torch.max(options_transdec(states, states, mem_padding_mask = padding_masks, tgt_padding_mask = padding_masks), 0)[0])
#         # options_logits = options_class(options_transdec(final_state, states, mem_padding_mask = padding_masks).squeeze(0))

#         return options_logits

#     def get_logits(self, proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks = None, lengths = None):
#         batch_size = proprio_diffs.size(0)
#         sequence_size = proprio_diffs.size(1)

#         forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

#         prestate = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2), actions], dim = 2)
#         prestate_reshaped = torch.reshape(prestate, (prestate.size(0) * prestate.size(1), prestate.size(2)))#.contiguous()

#         options_logits01 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[0],\
#          batch_size, sequence_size, padding_masks, lengths)

#         options_logits11 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[1],\
#          batch_size, sequence_size, padding_masks, lengths)

#         options_logits21 = self.get_data(prestate_reshaped, forces_reshaped, self.ensemble_list[2],\
#          batch_size, sequence_size, padding_masks, lengths)
        
#         ol_peg0 = peg_type[:,0].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
#         ol_peg1 = peg_type[:,1].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)
#         ol_peg2 = peg_type[:,2].unsqueeze(1).repeat_interleave(options_logits01.size(1), dim=1)

#         options_logits1 = ol_peg0 * options_logits01 + ol_peg1 * options_logits11 + ol_peg2 * options_logits21

#         return options_logits1

#     def forward(self, input_dict):
#         proprio_diffs = input_dict["proprio_diff"].to(self.device)
#         actions = input_dict["action"].to(self.device)
#         forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
#         contact_diffs = input_dict["contact_diff"].to(self.device)
#         peg_type = input_dict["peg_type"].to(self.device)
#         padding_masks = input_dict["padding_mask"].to(self.device)
#         lengths = input_dict["length"].to(self.device).long().squeeze(1)

#         options_logits = self.get_logits(proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks, lengths)

#         return {
#             'options_class': options_logits,
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

#         options_logits = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, peg_type)   

#         return options_logits