import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
import time
import yaml
import itertools
import random
from collections import OrderedDict

import sys

import project_utils as pu
import models_modules as mm
import multinomial as multinomial

def get_ref_model_dict():
    return {
        'History_Encoder': History_Encoder,
        'Variational_History_Encoder': Variational_History_Encoder,
        'Selfsupervised_History_Encoder': Selfsupervised_History_Encoder,
        'Unsupervised_History_Encoder': Unsupervised_History_Encoder,
        'StatePosSensor_wUncertainty': StatePosSensor_wUncertainty,
        'StatePosSensor_wConstantUncertainty' : StatePosSensor_wConstantUncertainty,
        'History_Encoder_Baseline': History_Encoder_Baseline,
        'StatePos_DualFilter': StatePos_DualFilter,
        'StatePos_ComboFilter': StatePos_ComboFilter,
    }

######################################
# Defining Custom Macromodels for project
#######################################

#### All models must have three input arguments: model_name, init_args, device
def sample_gaussian(m, v, device):
    
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z

def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

class History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_objects']
        self.num_states = init_args['num_objects']
        self.num_obs = 2

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + 2 * self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, obs_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(torch.cat([states_T, shape_embed(input_dict['state_idx'].long())], dim = 1))

        obs_logits = obs_classifier(states_T)

        return pos_ests, obs_logits, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
        }      

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def pos_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            return 0.01 * pos_ests.squeeze().cpu().numpy(), None

    def type_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), None

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['state_idx'] = input_dict['hole_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0) 
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class Variational_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed, pos_estimator, obs_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_params = params_estimator(seq_output)

        seq_mean, seq_var  = gaussian_parameters(seq_params)

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        obs_logits = obs_classifier(states_T)

        return pos_ests, obs_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, obs_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, obs_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class Selfsupervised_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_force_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 6, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_contact_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pairing_class" + str(i),\
            self.state_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed,\
         pos_predictor, force_predictor, contact_predictor, pairing_classifier = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_mean, seq_var  = gaussian_parameters(params_estimator(seq_output))

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        action_states_T = torch.cat([seq_enc, tool_embed, input_dict['final_action']], dim = 1)

        pos_preds = pos_predictor(action_states_T)

        force_preds = force_predictor(action_states_T)

        contact_preds = contact_predictor(action_states_T)

        paired_logits = pairing_classifier(seq_enc)

        if 'force_unpaired_reshaped' in input_dict.keys():
            frc_encs_unpaired_reshaped = self.flatten(frc_enc(input_dict["force_unpaired_reshaped"]))

            frc_unpaired_encs = torch.reshape(frc_encs_unpaired_reshaped,\
             (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

            states_unpaired_t = torch.cat([frc_unpaired_encs, input_dict['sensor_inputs_unpaired']], dim = 2).transpose(0,1)

            if "padding_mask" in input_dict.keys():
                seq_unpaired_output = seq_processor(states_unpairedno_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
            else:
                seq_unpaired_output = seq_processor(states_unpaired_t).max(0)[0]

            seq_unpaired_mean, seq_unpaired_var  = gaussian_parameters(params_estimator(seq_unpaired_output))

            seq_unpaired_enc = sample_gaussian(seq_unpaired_mean, seq_unpaired_var, self.device)

            unpaired_logits = pairing_classifier(seq_unpaired_enc)

        else:
            unpaired_logits = None

        return pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits,\
        enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

        paired_idxs = torch.cat([
            torch.ones_like(paired_logits[:,0]).long(), torch.zeros_like(unpaired_logits[:,0]).long()
            ], dim = 0)

        paired_combined_logits = torch.cat([paired_logits, unpaired_logits], dim = 0)
        
        return {
            'pos_pred': pos_preds,
            'force_pred': force_preds,
            'contact_pred': contact_preds,
            'contact_inputs': multinomial.logits2inputs(contact_preds),
            'paired_class': paired_combined_logits,
            'paired_inputs': multinomial.logits2inputs(paired_combined_logits),
            'paired_idx': paired_idxs,
            'unpaired_inputs': multinomial.logits2inputs(paired_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits,\
            enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool
    
    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
         input_dict['proprio'][:,1:] - input_dict['proprio'][:,:-1], torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)

        if 'force_hi_freq_unpaired' in input_dict.keys():
            input_dict['force_unpaired'] = input_dict['force_hi_freq_unpaired'].transpose(2,3)

            input_dict['force_unpaired_reshaped'] = torch.reshape(input_dict["force_unpaired"],\
         (input_dict["force_unpaired"].size(0) * input_dict["force_unpaired"].size(1), \
         input_dict["force_unpaired"].size(2), input_dict["force_unpaired"].size(3)))

            input_dict['proprio_unpaired_diff'] = torch.where(input_dict['proprio_unpaired'][:,1:] != 0,\
             input_dict['proprio_unpaired'][:,1:] - input_dict['proprio_unpaired'][:,:-1], torch.zeros_like(input_dict['proprio_unpaired'][:,1:]))            
            
            input_dict['sensor_inputs_unpaired'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_unpaired_diff']], dim = 2)

    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        input_dict['final_action'] = input_dict['action'][:,-1].repeat_interleave(T,0)

class Unsupervised_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_objects']
        self.num_states = init_args['num_objects']
        self.num_obs = init_args['num_objects']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_force_est" + str(i),\
            self.state_size + self.tool_dim, 6, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_contact_class" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed,\
         pos_estimator, force_estimator, contact_classifier = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_params = params_estimator(seq_output)

        seq_mean, seq_var  = gaussian_parameters(seq_params)

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        force_ests = force_estimator(states_T)

        contact_logits = contact_classifier(states_T)

        return pos_ests, force_ests, contact_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, force_ests, contact_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'force_est': force_ests,
            'contact_class': contact_logits,
            'contact_inputs': multinomial.logits2inputs(contact_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, force_ests, contact_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
         input_dict['proprio'][:,1:] - input_dict['proprio'][:,:-1], torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class StatePosSensor_wUncertainty(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est_dyn_noise" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_likelihood" + str(i),\
            self.state_size + self.tool_dim, self.num_obs * self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, dyn_noise_estimator, obs_noise_estimator,\
         obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests_obs = pos_estimator(states_T)
        
        pos_ests_dyn_noise = dyn_noise_estimator(states_T).pow(2) + 1e-2 #+ pos_ests_obs.pow(2)

        pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2) + 1e-2

        obs_logits = obs_classifier(states_T)

        # likelihood network
        obs_state_logits = torch.reshape(obs_likelihood(states_T), (input_dict['batch_size'], self.num_obs, self.num_states))
        obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

        state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

        return pos_ests_obs, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, state_logprobs, obs_state_logprobs, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

        prior_noise = input_dict['rel_pos_prior_var'] #+ pos_ests_dyn_noise

        y = pos_ests_mean - input_dict['rel_pos_prior_mean']
        S = prior_noise + pos_ests_obs_noise
        K = prior_noise / S

        pos_post = input_dict['rel_pos_prior_mean'] + K * y
        pos_post_var = (1 - K) * prior_noise

        state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

        # print(state_logits_unc[:10])
        # print(input_dict['done_mask'][:10])
        # print(state_logits[:10])
        # print(input_dict['done_mask'][:,0].mean())

        # print((1 / pos_post_var)[:10])
        # if torch.isnan(1 / pos_post_var).any():
        #     print("Problem")

        return {
            'pos_est': pos_post,
            'pos_est_params': (pos_post, 1 / pos_post_var),
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'state_logits': state_logits,
            'state_inputs': multinomial.logits2inputs(state_logits),
            'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def pos_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return 0.01 * pos_ests_mean.squeeze().cpu().numpy(), 0.0001 * pos_ests_obs_noise.squeeze().cpu().numpy()

    def type_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_mean,  pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), obs_state_logprobs.squeeze().cpu().numpy()
              
    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class StatePosSensor_wConstantUncertainty(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_objects']
        self.num_states = init_args['num_objects']
        self.num_obs = 2
        self.num_objects = init_args['num_objects']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_pos_est_obs_noise" + str(i),\
                    self.num_tools, 2, device= self.device).to(self.device),\

            #     mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
            # self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            # uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_obs_likelihood" + str(i),\
                    self.num_tools, self.num_obs * self.num_states, device= self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, obs_noise_estimator,\
         obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests_obs = pos_estimator(states_T)

        # pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2) + 1e-2
        pos_ests_obs_noise = obs_noise_estimator(input_dict['tool_idx'].long()).pow(2) + 1e-2

        # used to constrain the solution during test time based on prior knowledge
        # if 'reference_pos' in input_dict.keys():
        #     distance_norm = (pos_ests_obs - input_dict['reference_pos']).norm(p=2,dim=1).unsqueeze(1).repeat_interleave(2, dim=1)

        #     pos_ests_obs = torch.where(distance_norm > 2, (2 / distance_norm) * (pos_ests_obs - input_dict['reference_pos']) +\
        #      input_dict['reference_pos'], pos_ests_obs)

        # pos_ests_obs = pos_estimator(torch.cat([states_T, shape_embed(input_dict['state_idx'].long())], dim = 1))
        # pos_ests_obs_state_noise = torch.reshape(obs_noise_estimator(input_dict['tool_idx'].long()).pow(2) + 1e-2,\
        #  (input_dict['batch_size'], self.num_states, 2))

        # pos_ests_obs_noise = pos_ests_obs_state_noise[torch.arange(input_dict['batch_size']), input_dict['state_idx']]

        obs_logits = obs_classifier(states_T)

        # likelihood network
        obs_state_logits = torch.reshape(obs_likelihood(input_dict['tool_idx'].long()), (input_dict['batch_size'], self.num_obs, self.num_states))
        obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

        state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

        return pos_ests_obs, pos_ests_obs_noise, obs_logits, state_logprobs, obs_state_logprobs, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

        prior_noise = input_dict['rel_pos_prior_var']

        y = pos_ests_mean - input_dict['rel_pos_prior_mean']
        S = prior_noise + pos_ests_obs_noise
        K = prior_noise / S

        pos_post = input_dict['rel_pos_prior_mean'] + K * y
        pos_post_var = (1 - K) * prior_noise

        state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

        # print(state_logits_unc[:10])
        # print(input_dict['done_mask'][:10])
        # print(state_logits[:10])
        # print(input_dict['done_mask'][:,0].mean())

        # print((1 / pos_post_var)[:10])
        # if torch.isnan(1 / pos_post_var).any():
        #     print("Problem")

        return {
            'pos_est': pos_post,
            'pos_est_params': (pos_post, 1 / pos_post_var),
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'state_logits': state_logits,
            'state_inputs': multinomial.logits2inputs(state_logits),
            'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def pos_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)
            # input_dict['state_idx'] = input_dict['tool_idx']

            pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return 0.01 * pos_ests_mean.squeeze().cpu().numpy(), 0.0001 * pos_ests_obs_noise.squeeze().cpu().numpy()

    def type_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)
            # input_dict['state_idx'] = input_dict['tool_idx']

            pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), obs_state_logprobs.squeeze().cpu().numpy()

    def get_loglikelihood_model(self):
        with torch.no_grad():
            self.eval()

            ll_embedding_layer = self.ensemble_list[0][6]
            idxs = torch.arange(self.num_tools).to(self.device).long()
            loglikelihood_model = F.log_softmax(torch.reshape(ll_embedding_layer(idxs), (self.num_tools, self.num_obs, self.num_states)), dim = 1)

            # print(loglikelihood_model.size())

            return loglikelihood_model.cpu().numpy()

    def process_inputs(self, input_dict):
        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['state_idx'] = input_dict['hole_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

        if 'rel_pos_init' in input_dict.keys():
            input_dict['reference_pos'] = 100 * input_dict['rel_pos_init'].unsqueeze(0).repeat_interleave(T, 0)

class StatePos_DualFilter(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6
        self.state_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_tool_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_type_classifier" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_type_likelihood" + str(i),\
                    self.num_tools, self.num_states * self.num_states, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "particle_likelihood" + str(i),\
            self.state_size + self.tool_dim + 2, 1, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple):
        frc_enc, seq_processor, tool_embed, type_classifier, type_likelihood, particle_likelihood = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = tool_embed(input_dict['tool_idx'].long()) 

        obs_enc = torch.cat([seq_enc, tool_embed], dim = 1)

        # Histogram Filter Steps
        obs_logits = type_classifier(obs_enc)

        # likelihood network # batch x obs x states
        obs_state_logits = torch.reshape(type_likelihood(input_dict['tool_idx'].long()), (input_dict['batch_size'], self.num_states, self.num_states))
        obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

        # batch size x states
        state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

        ###########################################################################################
        # Particle Filter Steps
        obs_enc_prior = obs_enc.unsqueeze(1).repeat_interleave(input_dict['pos_particles'].size(1), dim = 1)

        #calculating likelihood weights
        obs_plus_particles = torch.cat([obs_enc_prior, input_dict['pos_particles']], dim = 2)

        obs_weights = torch.clamp(torch.sigmoid(particle_likelihood(obs_plus_particles)), 0.004, 1).squeeze(2)

        return obs_enc, obs_state_logprobs, state_logprobs, obs_logits, obs_weights

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        enc, obs_state_logprobs, obs_logprobs, obs_logits, obs_weights = self.get_outputs(input_dict, self.ensemble_list[0])

        state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

        pos_probs = F.softmax(torch.log(input_dict['weights_particles']) + torch.log(obs_weights), dim = 1)

        pos_est = (input_dict['pos_particles'] * pos_probs.unsqueeze(2).repeat_interleave(2, dim =  2)).sum(1)

        return {
            'pos_est': pos_est,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'state_logits': state_logits,
            'state_inputs': multinomial.logits2inputs(state_logits),
            'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def type_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            enc, obs_state_logprobs, obs_logprobs, obs_logits, obs_weights = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), obs_state_logprobs.squeeze().cpu().numpy(),

    def update_weights(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            enc, obs_state_logprobs, obs_logprobs, obs_logits, obs_weights = self.get_outputs(input_dict, self.ensemble_list[0])

            pos_probs = F.softmax(torch.log(input_dict['weights_particles']) + torch.log(obs_weights), dim = 1)

            pos_est = (input_dict['pos_particles'] * pos_probs.unsqueeze(2).repeat_interleave(2, dim =  2)).sum(1)

            return pos_probs.squeeze().cpu().numpy(), None, 0.01 * pos_est.squeeze().cpu().numpy()

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        input_dict['pos_particles'] = input_dict['pos_particles'].unsqueeze(0).repeat_interleave(T,0)
        input_dict['weights_particles'] = input_dict['weights_particles'].unsqueeze(0).repeat_interleave(T,0)
        input_dict['type_particles_idx'] = input_dict['type_particles_idx'].unsqueeze(0).repeat_interleave(T,0)

class StatePos_ComboFilter(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6
        self.state_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False
        self.discretization = 20

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_tool_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.Embedding(model_name + "_type_embed" + str(i),\
                         self.num_tools, self.state_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "particle_likelihood" + str(i),\
            self.state_size + self.tool_dim + 2 + self.state_dim, 1, 2 * self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple):
        frc_enc, seq_processor, tool_embed, type_embed, particle_likelihood = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = tool_embed(input_dict['tool_idx'].long()) 

        obs_enc = torch.cat([seq_enc, tool_embed], dim = 1)
        ###########################################################################################
        # Particle Filter Steps
        obs_enc_prior = obs_enc.unsqueeze(1).repeat_interleave(input_dict['type_particles_idx'].size(1), dim = 1)

        #calculating likelihood weights
        obs_plus_particles = torch.cat([obs_enc_prior, type_embed(input_dict['type_particles_idx'].long()),\
         input_dict['pos_particles']], dim = 2)

        obs_weights = torch.clamp(torch.sigmoid(particle_likelihood(obs_plus_particles)), 0.004, 1).squeeze(2)

        return obs_enc, obs_weights

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        enc, obs_weights = self.get_outputs(input_dict, self.ensemble_list[0])

        # batch size x particles
        state_pos_probs = F.softmax(torch.log(obs_weights) + torch.log(input_dict['weights_particles']), dim = 1)

        # batch size x state size x particles
        type_particles_idx = input_dict['type_particles_idx'].unsqueeze(2).repeat_interleave(self.num_states, 2)
        idxs = torch.arange(self.num_states).unsqueeze(0).unsqueeze(1).repeat(input_dict['batch_size'], obs_weights.size(1), 1).to(self.device)
        mask = torch.where(type_particles_idx == idxs, torch.ones_like(idxs), torch.zeros_like(idxs))

        state_pos_probs_exp = state_pos_probs.unsqueeze(2).repeat_interleave(self.num_states, 2)

        state_probs = (mask * state_pos_probs_exp).sum(1) # doesn't always sum to 1, but it is always within 1e-6 of 1

        assert torch.where(torch.abs(state_probs.sum(1) - 1) < 1e-4, torch.zeros_like(state_probs.sum(1)),\
            torch.ones_like(state_probs.sum(1))).sum() == 0

        state_logits = torch.log(state_probs)

        state_probs_weights = (mask * state_probs.unsqueeze(1).repeat_interleave(input_dict['weights_particles'].size(1), dim = 1)).sum(-1)

        pos_logprobs = torch.log(state_pos_probs) - torch.log(state_probs_weights) + torch.log(torch.ones_like(state_probs_weights) / self.num_states)

        assert torch.where(torch.abs(torch.exp(pos_logprobs).sum(1) - 1) < 1e-4, torch.zeros_like(torch.exp(pos_logprobs).sum(1)),\
            torch.ones_like(torch.exp(pos_logprobs).sum(1))).sum() == 0

        pos_est = (input_dict['pos_particles'] * torch.exp(pos_logprobs).unsqueeze(2).repeat_interleave(2, dim =  2)).sum(1)

        return {
            'pos_est': pos_est,
            'state_logits': state_logits,
            'state_inputs': multinomial.logits2inputs(state_logits),
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def update_weights(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            enc, obs_weights = self.get_outputs(input_dict, self.ensemble_list[0])

            # batch size x particles
            state_pos_probs = F.softmax(torch.log(obs_weights) + torch.log(input_dict['weights_particles']), dim = 1)

            # batch size x state size x particles
            type_particles_idx = input_dict['type_particles_idx'].unsqueeze(2).repeat_interleave(self.num_states, 2)
            idxs = torch.arange(self.num_states).unsqueeze(0).unsqueeze(1).repeat(input_dict['batch_size'], obs_weights.size(1), 1).to(self.device)
            mask = torch.where(type_particles_idx == idxs, torch.ones_like(idxs), torch.zeros_like(idxs))

            state_pos_probs_exp = state_pos_probs.unsqueeze(2).repeat_interleave(self.num_states, 2)

            state_probs = (mask * state_pos_probs_exp).sum(1) # doesn't always sum to 1, but it is always within 1e-6 of 1

            assert torch.where(torch.abs(state_probs.sum(1) - 1) < 1e-4, torch.zeros_like(state_probs.sum(1)),\
                torch.ones_like(state_probs.sum(1))).sum() == 0

            state_logits = torch.log(state_probs)

            state_probs_weights = (mask * state_probs.unsqueeze(1).repeat_interleave(input_dict['weights_particles'].size(1), dim = 1)).sum(-1)

            pos_logprobs = torch.log(state_pos_probs) - torch.log(state_probs_weights) + torch.log(torch.ones_like(state_probs_weights) / self.num_states)

            assert torch.where(torch.abs(torch.exp(pos_logprobs).sum(1) - 1) < 1e-4, torch.zeros_like(torch.exp(pos_logprobs).sum(1)),\
                torch.ones_like(torch.exp(pos_logprobs).sum(1))).sum() == 0

            pos_est = (input_dict['pos_particles'] * torch.exp(pos_logprobs).unsqueeze(2).repeat_interleave(2, dim =  2)).sum(1)

            return state_pos_probs.squeeze().cpu().numpy(), state_probs.squeeze().cpu().numpy(), 0.01 * pos_est.squeeze().cpu().numpy()

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        input_dict['pos_particles'] = input_dict['pos_particles'].unsqueeze(0).repeat_interleave(T,0)
        input_dict['weights_particles'] = input_dict['weights_particles'].unsqueeze(0).repeat_interleave(T,0)
        input_dict['type_particles_idx'] = input_dict['type_particles_idx'].unsqueeze(0).repeat_interleave(T,0)

class History_Encoder_Baseline(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))


        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, obs_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        obs_logits = obs_classifier(states_T)

        return pos_ests, obs_logits, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_obs, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

# class PosSensor_wUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = False

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est_dyn_noise" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, pos_estimator, dyn_noise_estimator, obs_noise_estimator = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         pos_ests_obs = pos_estimator(states_T)
        
#         pos_ests_dyn_noise = dyn_noise_estimator(states_T).pow(2) + 1e-2 #+ pos_ests_obs.pow(2)

#         pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2) + 1e-2


#         return pos_ests_obs, pos_ests_dyn_noise, pos_ests_obs_noise, states_T

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#         prior_noise = input_dict['rel_pos_prior_var'] + pos_ests_dyn_noise

#         y = pos_ests_mean - input_dict['rel_pos_prior_mean']
#         S = prior_noise + pos_ests_obs_noise
#         K = prior_noise / S

#         pos_post = input_dict['rel_pos_prior_mean'] + K * y
#         pos_post_var = (1 - K) * prior_noise

#         # print(state_logits_unc[:10])
#         # print(input_dict['done_mask'][:10])
#         # print(state_logits[:10])
#         # print(input_dict['done_mask'][:,0].mean())

#         # print((1 / pos_post_var)[:10])
#         # if torch.isnan(1 / pos_post_var).any():
#         #     print("Problem")

#         return {
#             'pos_est': pos_post,
#             'pos_est_params': (pos_post, 1 / pos_post_var),
#         }

#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

#             return 0.01 * pos_ests_mean[0], 0.0001 * pos_ests_dyn_noise[0], 0.0001 * pos_ests_obs_noise[0]

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2].unsqueeze(0)\
#         .repeat_interleave(input_dict['rel_proprio'].size(1),0)\
#          - input_dict['rel_proprio'][0,:,:2]

#         input_dict['rel_pos_shift'] = input_dict['pos_change'][-1].unsqueeze(0).repeat_interleave(T, 0)

# class StateSensor_wUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = False

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_class" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_likelihood" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs * self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         obs_logits = obs_classifier(states_T)

#         # likelihood network
#         obs_state_logits = torch.reshape(obs_likelihood(states_T), (input_dict['batch_size'], self.num_obs, self.num_states))
#         obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

#         state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

#         return obs_logits, state_logprobs

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         obs_logits, obs_logprobs = self.get_outputs(input_dict, self.ensemble_list[0])

#         state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

#         return {
#             'obs_logits': obs_logits,
#             'obs_inputs': multinomial.logits2inputs(obs_logits),
#             'state_logits': state_logits,
#             'state_inputs': multinomial.logits2inputs(state_logits),
#             'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
#         }


#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             obs_logits, obs_logprobs = self.get_outputs(input_dict, self.ensemble_list[0])

#             obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

#             return int(obs_idx.item()), obs_logprobs[0].cpu().numpy(), None, None

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

# class StateSensor_wConstantUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = False

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_class" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.Embedding(model_name + "_obs_likelihood" + str(i),\
#                          self.num_tools, self.num_states * self.num_obs, device= self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         obs_logits = obs_classifier(states_T)

#         # likelihood network
#         obs_state_logits = torch.reshape(obs_likelihood(input_dict['tool_idx'].long()), (input_dict['batch_size'], self.num_obs, self.num_states))
#         obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

#         state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

#         return obs_logits, state_logprobs

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         obs_logits, obs_logprobs = self.get_outputs(input_dict, self.ensemble_list[0])

#         state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

#         return {
#             'obs_logits': obs_logits,
#             'obs_inputs': multinomial.logits2inputs(obs_logits),
#             'state_logits': state_logits,
#             'state_inputs': multinomial.logits2inputs(state_logits),
#             'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
#         }


#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             obs_logits, obs_logprobs = self.get_outputs(input_dict, self.ensemble_list[0])

#             obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

#             return int(obs_idx.item()), obs_logprobs[0].cpu().numpy(), None, None

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

# class History_Encoder_wUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = False

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_class" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_likelihood" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs * self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, pos_estimator, obs_noise_estimator,\
#          obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         pos_ests_obs = pos_estimator(states_T)
        
#         pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2)

#         obs_logits = obs_classifier(states_T)

#         # likelihood network
#         obs_state_logits = torch.reshape(obs_likelihood(states_T), (input_dict['batch_size'], self.num_obs, self.num_states))
#         obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

#         state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

#         return pos_ests_obs, pos_ests_obs_noise, obs_logits, state_logprobs, states_T

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#         prior_noise = input_dict['rel_pos_prior_var']

#         y = pos_ests_mean - input_dict['rel_pos_prior_mean']
#         S = prior_noise + pos_ests_obs_noise
#         K = prior_noise / S

#         pos_post = input_dict['rel_pos_prior_mean'] + K * y
#         pos_post_var = (1 - K) * prior_noise

#         state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

#         # print(state_logits_unc[:10])
#         # print(input_dict['done_mask'][:10])
#         # print(state_logits[:10])
#         # print(input_dict['done_mask'][:,0].mean())

#         # print((1 / pos_post_var)[:10])
#         # if torch.isnan(1 / pos_post_var).any():
#         #     print("Problem")

#         return {
#             'pos_est': pos_post,
#             # 'pos_est_params': (pos_post, 1 / pos_post_var),
#             'obs_logits': obs_logits,
#             'obs_inputs': multinomial.logits2inputs(obs_logits),
#             'state_logits': state_logits,
#             'state_inputs': multinomial.logits2inputs(state_logits),
#             'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
#         }

#     def get_encoding(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             ### need self uc to be off
#             # prev_time = time.time()          
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             return enc

#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

#             return int(obs_idx.item()), obs_logprobs[0], 0.01 * pos_ests_mean[0], 0.0001 * pos_ests_obs_noise[0]

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2].unsqueeze(0)\
#         .repeat_interleave(input_dict['rel_proprio'].size(1),0)\
#          - input_dict['rel_proprio'][0,:,:2]

#         input_dict['rel_pos_shift'] = input_dict['pos_change'][-1].unsqueeze(0).repeat_interleave(T, 0)

# class History_Encoder_wConstantUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = False

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
#             self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_class" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.Embedding(model_name + "_obs_likelihood" + str(i),\
#                          self.num_tools, self.num_states * self.num_obs, device= self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, pos_estimator, obs_noise_estimator,\
#          obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         pos_ests_obs = pos_estimator(states_T)
        
#         pos_ests_var = obs_noise_estimator(states_T).pow(2)

#         obs_logits = obs_classifier(states_T)

#         # likelihood network
#         obs_state_logits = torch.reshape(obs_likelihood(input_dict['tool_idx'].long()), (input_dict['batch_size'], self.num_obs, self.num_states))
#         obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

#         state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

#         return pos_ests_obs, pos_ests_var, obs_logits, state_logprobs, states_T

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#         prior_noise = input_dict['rel_pos_prior_var']

#         y = pos_ests_mean - input_dict['rel_pos_prior_mean']
#         S = prior_noise + pos_ests_obs_noise
#         K = prior_noise / S

#         pos_post = input_dict['rel_pos_prior_mean'] + K * y
#         pos_post_var = (1 - K) * prior_noise

#         state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

#         # print(state_logits_unc[:10])
#         # print(input_dict['done_mask'][:10])
#         # print(state_logits[:10])
#         # print(input_dict['done_mask'][:,0].mean())

#         return {
#             'pos_est': pos_post,
#             'obs_logits': obs_logits,
#             'obs_inputs': multinomial.logits2inputs(obs_logits),
#             'state_logits': state_logits,
#             'state_inputs': multinomial.logits2inputs(state_logits),
#             'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
#         }

#     def get_encoding(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             ### need self uc to be off
#             # prev_time = time.time()          
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_obs, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             return enc

#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_obs, pos_ests_obs_noise, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

#             return int(obs_idx.item()), obs_logprobs[0], 0.01 * pos_ests_obs[0], 0.0001 * pos_ests_obs_noise[0]

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2].unsqueeze(0)\
#         .repeat_interleave(input_dict['rel_proprio'].size(1),0)\
#          - input_dict['rel_proprio'][0,:,:2]

#         input_dict['rel_pos_shift'] = input_dict['pos_change'][-1].unsqueeze(0).repeat_interleave(T, 0)

# class History_Encoder_wEstUncertainty(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']
#         self.num_obs = 2 #init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']
#         self.proprio_size = init_args['proprio_size']
#         self.min_length = init_args['min_length']
#         self.contact_size = 1

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.tool_dim = 6

#         self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

#         self.num_ensembles = 1

#         self.num_tl = 4
#         self.num_cl = 3
#         self.flatten = nn.Flatten()
#         self.uc = True

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
#                 mm.Embedding(model_name + "_shape_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_pos_est" + str(i),\
#             self.state_size + self.tool_dim, 2, 2 * self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_class" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_obs_likelihood" + str(i),\
#             self.state_size + self.tool_dim, self.num_obs * self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple, final_idx = None):
#         frc_enc, seq_processor, shape_embed, pos_model,\
#          obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs = torch.reshape(frc_encs_reshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

#         states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
#         else:
#             seq_enc = seq_processor(states_t).max(0)[0]

#         tool_embed = shape_embed(input_dict['tool_idx'].long()) 

#         states_T = torch.cat([seq_enc, tool_embed], dim = 1)

#         pos_samples = 0.01 * torch.cat([\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1),\
#         pos_model(states_T).unsqueeze(1)\
#         ], dim = 1)
#         pos_ests_mean = torch.mean(pos_samples, dim = 1)
#         pos_ests_var = torch.var(pos_samples, unbiased=True, dim = 1)

#         # obs_logits_samples = torch.cat([\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     obs_classifier(states_T).unsqueeze(1),\
#         #     ], dim = 1)

#         # obs_logits_mean = torch.mean(obs_logits_samples, dim = 1)
#         # obs_logits_var = torch.var(obs_logits_samples, unbiased = True, dim =1)

#         # obs_logits = sample_gaussian(obs_logits_mean, obs_logits_var, self.device)
        
#         obs_logits = obs_classifier(states_T)

#         # likelihood network
#         obs_state_logits = torch.reshape(obs_likelihood(states_T), (input_dict['batch_size'], self.num_obs, self.num_states))
#         obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

#         state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

#         return pos_ests_mean, pos_ests_var, obs_logits, state_logprobs, states_T

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)
#         self.set_uc(True)
#         pos_ests_mean, pos_ests_var, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#         y = pos_ests_mean - input_dict['rel_pos_shift']
#         S = input_dict['rel_pos_shift_var'] + pos_ests_var
#         K = input_dict['rel_pos_shift_var'] / S
#         pos_ests = input_dict['rel_pos_shift'] + K * y

#         state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

#         return {
#             'pos_est': pos_ests,
#             'obs_logits': obs_logits,
#             'obs_inputs': multinomial.logits2inputs(obs_logits),
#             'state_logits': state_logits,
#             'state_inputs': multinomial.logits2inputs(state_logits),
#             'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
#         }      

#     def get_encoding(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             ### need self uc to be off
#             # prev_time = time.time()          
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(False)

#             pos_ests_mean, pos_ests_var, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             return enc

#     def set_uc(self, uc_bool):
#         for model in self.ensemble_list[0]:
#             if hasattr(model, 'set_uc'):
#                 model.set_uc(uc_bool)
#             elif hasattr(model, 'uc'):
#                 model.uc = uc_bool

#     def get_obs(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()
#             ### Method 1 #### without uncertainty estimation
#             T = 1
#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)
#             self.set_uc(True)

#             pos_ests_mean, pos_ests_var, obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             # obs_logits, obs_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             # pos_estimator = self.ensemble_list[0][-3]

#             # pos_samples = 0.01 * torch.cat([\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1),\
#             # pos_estimator(enc).unsqueeze(1)\
#             # ], dim = 1)

#             # pos_ests_mean = torch.mean(pos_samples, dim = 1)
#             # pos_ests_var = torch.var(pos_samples, unbiased=True, dim = 1)
                    
#             obs_idx = F.softmax(obs_logits[0], dim = 0).max(0)[1]

#             return int(obs_idx.item()), obs_logprobs[0], pos_ests_mean[0], pos_ests_var[0]

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
#          input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

#         input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

# ### Method 2 ### with uncertainty estimation
# T = 100
# self.test_time_process_inputs(input_dict, T)
# self.process_inputs(input_dict)
# self.set_uc(True)

# pos_ests_obs, pos_ests_var, obs_logits, enc  = self.get_outputs(input_dict, self.ensemble_list[0])

# obs_samples = torch.zeros((input_dict["batch_size"], self.num_observations)).float().to(self.device)
# obs_samples[torch.arange(input_dict['batch_size']), obs_logits.max(1)[1]] = 1.0
# obs_probs = obs_samples.sum(0) / input_dict['batch_size']

# pos_est = pos_ests.mean(0)
# pos_est_var = pos_ests.var(0)
# obs_idx = obs_probs.max(0)[1]

# return obs_idx, pos_est, pos_est_var

# class Options_Sensor(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_options = init_args['num_options']

#         self.action_dim = init_args['action_dim']
#         self.proprio_size = init_args['proprio_size'] + 8
#         self.force_size = init_args['force_size']

#         self.tool_dim = init_args['tool_dim']

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(init_args['force_mean']).to(self.device).float()
#         self.force_std = torch.from_numpy(init_args['force_std']).to(self.device).float()

#         self.contact_size = 1
#         self.frc_enc_size =  8 * 3 * 2
#         self.num_ensembles = 3

#         self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim

#         self.num_tl = 4
#         self.num_cl = 2
#         self.flatten = nn.Flatten()
#         self.uc = True

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 mm.Embedding(model_name + "_tool_embed" + str(i),\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device),\

#                 mm.CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                mm.ResNetFCN(model_name + "_frc_enc_context" + str(i),\
#             self.frc_enc_size + self.tool_dim, self.frc_enc_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 mm.Transformer_Comparer(model_name + "_state_transdec" + str(i),\
#          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

#                 mm.ResNetFCN(model_name + "_options_class" + str(i),\
#             self.state_size, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device)
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_logits(self, input_dict, model_tuple):
#         tool_embed, frc_enc, frc_enc_context, state_transdec, states_class = model_tuple #origin_cov = model_tuple # 

#         ##### Calculating Series Encoding
#         # Step 1. frc encoding
#         # print(input_dict["forces_reshaped"].size())
#         # print(input_dict["forces_reshaped"].max())
#         # print(input_dict["forces_reshaped"].min())
#         # print(torch.isnan(input_dict["forces_reshaped"]).sum())
#         tool_embeds = tool_embed(input_dict["tool_idx"].long()).unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

#         tool_embeds_unshaped = torch.reshape(tool_embeds, (input_dict['batch_size']*input_dict['sequence_size'], tool_embeds.size(2)))

#         frc_encs_unshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
#         frc_encs_context_unshaped = frc_enc_context(torch.cat([frc_encs_unshaped, tool_embeds_unshaped]))
#         frc_encs = torch.reshape(frc_encs_context_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_enc_size))

#         # img_encs_unshaped = self.flatten(img_enc(input_dict["rgbd_reshaped"]))
#         # img_encs = torch.reshape(img_encs_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.img_enc_size))
#         # Step 2. full state sequence encoding
#         states_t = torch.cat([input_dict["states"], frc_encs, input_dict["action"]], dim = 2).transpose(0,1)

#         if "padding_mask" not in input_dict.keys():
#             seq_encs = state_transdec(states_t).max(0)[0]
#         else:
#             seq_encs = state_transdec(states_t, padding_mask = input_dict["padding_mask_extended"]).max(0)[0]

#         #### Calculating Options Logits
#         # print(seq_encs.size(), tool_embeds.size(), policy_embeds.size(), input_dict["tool_idx"].size())
#         states_logits = states_class(seq_encs)

#         options_logits = torch.zeros((input_dict['batch_size'], 2))

#         options_logits[:,0] += (states_logits * input_dict['tool_vector']).sum(1)

#         options_logits[:,1] += (states_logits * (1 - input_dict['tool_vector'])).sum(1)

#         return options_logits

#     def getall_logits(self, input_dict):

#         ol_list = []

#         for i in range(self.num_ensembles):
#             if "padding_mask" in input_dict.keys():
#                 input_dict["padding_mask_extended"] = self.get_input_dropout(input_dict["padding_mask"])

#             ol = self.get_logits(input_dict, self.ensemble_list[i])

#             ol_list.append(ol)

#         return ol_list #, seq_encs ol_list #

#     def get_input_dropout(self, padding_masks):
#         # print("Padding mask\n", padding_masks[0])
#         input_dropout = F.dropout(torch.ones(padding_masks.size()).float().to(self.device), p = (1 - (self.dropout_prob / 1))).bool()

#         padding_masks_extended = torch.where(padding_masks == False, input_dropout, padding_masks)

#         # print("New Padding mask\n", padding_masks_extended[0])

#         return  padding_masks_extended

#     def get_uncertainty_quant(self, input_dict):
#         with torch.no_grad():
#             ol_uncertainty_list = []
#             T = 60

#             for i in range(T):
#                 ol_list_sample = self.getall_logits(input_dict)

#                 for i in range(self.num_ensembles):
#                     ol_list_sample[i] = ol_list_sample[i].unsqueeze(0)

#                 ol_uncertainty_list += ol_list_sample

#             uncertainty_logits = torch.cat(ol_uncertainty_list, dim = 0)
#             # print(uncertainty_logits.size())

#             # for i in range(uncertainty_logits.size(0)):
#             #     for j in range(uncertainty_logits.size(1)):
#             #         print(F.softmax(uncertainty_logits[i,j], dim = 0))
#             uncertainty_votes = uncertainty_logits.max(2)[1]

#             uncertainty = torch.zeros((input_dict["batch_size"], self.num_options)).float().to(self.device)

#             for i in range(self.num_options):
#                 i_votes = torch.where(uncertainty_votes == i, torch.ones_like(uncertainty_votes), torch.zeros_like(uncertainty_votes)).sum(0)
#                 uncertainty[:, i] = i_votes

#             # print(uncertainty)
#             uncertainty = uncertainty / uncertainty.sum(1).unsqueeze(1).repeat_interleave(self.num_options, dim = 1)

#             return uncertainty

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         ol_list = self.getall_logits(input_dict)

#         inputs_list =[]

#         for i in range(self.num_ensembles):
#             inputs_list.append(multinomial.logits2inputs(ol_list[i]).unsqueeze(0))
#             ol_list[i] = ol_list[i].unsqueeze(0)

#         # uncertainty = self.get_uncertainty_quant(input_dict)

#         # print(uncertainty.size())

#         # uncertainty_list = [probs2inputs(uncertainty).unsqueeze(0)]
#         # for i in range(self.num_tools):
#         #     for j in range(self.num_states):
#         #         tb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
#         #         sb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
#         # #         print("Uncertainty:\n", tb * sb * uncertainty)
#         # print("Uncertainty: ", uncertainty)

#         return {
#             'options_class': torch.cat(ol_list, dim = 0),
#             'options_inputs': torch.cat(inputs_list, dim = 0),
#             # 'uncertainty_inputs': probs2inputs(uncertainty),
#         }

#     def probs(self, input_dict): 
#         with torch.no_grad():
#             self.eval()

#             self.test_time_process_inputs(input_dict)
#             self.process_inputs(input_dict)

#             ol_list = self.getall_logits(input_dict)

#             # for i in range(self.num_ensembles):
#             #     print(F.softmax(ol_list[i], dim = 1))

#             probs = F.softmax(random.choice(ol_list), dim = 1)

#             return probs.max(1)[1]

#     def process_inputs(self, input_dict):
#         input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
#          - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
#         self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3))) 

#         input_dict["states"] = torch.cat([input_dict["rel_proprio_diff"], input_dict["contact_diff"]], dim = 2)
        
#         input_dict["batch_size"] = input_dict["rel_proprio_diff"].size(0)
#         input_dict["sequence_size"] = input_dict["rel_proprio_diff"].size(1)

         
#     def test_time_process_inputs(self, input_dict):

#         # print(input_dict.keys())

#         input_dict["rel_proprio_diff"] = input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1]

#         # print(input_dict['rel_proprio_diff'].size())
#         input_dict["contact_diff"] = input_dict["contact"][:,1:] - input_dict["contact"][:, :-1]
#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
#         input_dict["action"] = input_dict["action"][:, :-1]
#         input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1]

#         if len(input_dict['tool_idx'].size()) == 0:
#             input_dict['tool_idx'] = input_dict['tool_idx'].unsqueeze(0)

# class Options_LikelihoodNet(mm.Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_options = init_args['num_options']
#         self.num_states = init_args['num_states']
#         self.num_policies = init_args['num_policies']

#         self.tool_dim = init_args['tool_dim']
#         self.state_dim = self.tool_dim
#         self.policy_dim = init_args['policy_dim']
#         self.dropout_prob = 0.1 # init_args['dropout_prob']

#         self.nl = 3

#         self.tool_embed = mm.Embedding(model_name + "_tool_embed",\
#                          self.num_tools, self.tool_dim, device= self.device).to(self.device)

#         self.model_list.append(self.tool_embed)

#         self.state_embed = mm.Embedding(model_name + "_state_embed",\
#                          self.num_states, self.state_dim, device= self.device).to(self.device)

#         self.model_list.append(self.state_embed)

#         self.policy_embed = mm.Embedding(model_name + "_policy_embed",\
#          self.num_policies, self.policy_dim, device= self.device).to(self.device)
        
#         self.model_list.append(self.policy_embed)

#         self.likelihood_net = mm.ResNetFCN(model_name + "_likelihood_net", self.tool_dim + self.state_dim + self.policy_dim,\
#          self.num_options, self.nl, dropout = True, dropout_prob = self.dropout_prob, uc = False, device = self.device).to(self.device)\

#         self.model_list.append(self.likelihood_net)

#     def forward(self, input_dict):
#         tool_embed = self.state_embed(input_dict["tool_idx"].long())
#         state_embed = self.state_embed(input_dict["state_idx"].long())
#         pol_embed = self.policy_embed(input_dict["pol_idx"].long())

#         likelihood_logits = self.likelihood_net(torch.cat([tool_embed, state_embed, policy_embed], dim =1))

#         likelihood_inputs = multinomial.logits2inputs(likelihood_logits)

#         # print(likelihood_inputs.size())

#         return {
#             'likelihood_inputs': likelihood_inputs,
#         }

#     def logprobs(self, input_dict, obs_idx, with_margin):
#         self.eval()

#         tool_embed = self.state_embed(input_dict["tool_idx"].long())
#         state_embed = self.state_embed(input_dict["state_idx"].long())
#         pol_embed = self.policy_embed(input_dict["pol_idx"].long())

#         likelihood_logits = self.likelihood_net(torch.cat([tool_embed, state_embed, policy_embed], dim =1))

#         if with_margin:
#             uninfo_constant = 0.2            
#             likelihood_logprobs = F.log_softmax(torch.log(F.softmax(likelihood_logits, dim = 1) + uninfo_constant), dim = 1)
#         else:
#             likelihood_logprobs = F.log_softmax(likelihood_logits, dim=1)

#         # print(likelihood_logprobs)
#         # print(obs_idx)

#         # print(likelihood_logprobs.size())

#         likelihood_logprob = likelihood_logprobs[torch.arange(likelihood_logprobs.size(0)), obs_idx]


                #     \
        #  - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
        # self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
        #  .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)

        # input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        # #     \
        # #  - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
        # # self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
        # #  .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
        # input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
        #  (input_dict["force"].size(0) * input_dict["force"].size(1), \
        #  input_dict["force"].size(2), input_dict["force"].size(3)))

        # input_dict["batch_size"] = input_dict["force"].size(0)
        # input_dict["sequence_size"] = input_dict["force"].size(1)

        # # print(input_dict["batch_size"])
        # # print(input_dict["sequence_size"])

        # tool_vectors = input_dict['tool_vector'].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

        # input_dict['context_reshaped'] = torch.reshape(torch.cat([tool_vectors, input_dict['action']], dim = 2),\
        #  (input_dict['batch_size'] * input_dict['sequence_size'], self.num_tools + self.action_size))

        # xyz_poses = input_dict['rel_proprio'][:,:,:3]

        # xyz_relative = torch.where(xyz_poses[:,1:] != 0, input_dict['contact'].repeat_interleave(3, dim = 2) * (xyz_poses[:,1:]\
        #  - xyz_poses[:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)), torch.zeros_like(xyz_poses[:,1:]))

        # d = xyz_relative.norm(p=2, dim=2)

        # input_dict['distance'] = torch.cat([xyz_relative, d.unsqueeze(2)], dim = 2)
# class History_Encoder_CNN(Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.force_size = init_args['force_size']

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.frc_context_enc_size = 48

#         self.num_ensembles = 1

#         self.num_cl = 2
#         self.flatten = nn.Flatten()
#         self.uc = False
#         self.image_size = 48
#         self.length_scale = 0.03
#         self.tolerance = 0.003

#         self.num_pixels = int(self.tolerance / (self.length_scale * 2 / self.image_size))

#         row_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(1).repeat_interleave(self.image_size, dim = 1).unsqueeze(2).to(self.device).float()
#         col_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(0).repeat_interleave(self.image_size, dim = 0).unsqueeze(2).to(self.device).float()

#         self.reference_image = (2 * self.length_scale / self.image_size) * torch.cat([row_idx, col_idx], dim = 2) + self.length_scale / self.image_size 

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = False, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_frc_context" + str(i),\
#              self.frc_enc_size + self.action_size + self.num_tools, self.frc_context_enc_size, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 CONV2DN(model_name + "_spatial_process" + str(i),\
#                     (self.frc_context_enc_size, self.image_size, self.image_size),\
#                     (128, 1, 1), nonlinear=False, batchnorm=True, dropout=False, dropout_prob= self.dropout_prob,\
#                     uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_pos_est" + str(i),\
#             128 + self.num_tools, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_state_class" + str(i),\
#             128 + self.num_tools, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device)
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_outputs(self, input_dict, model_tuple):
#         frc_enc, frc_enc_context, spatial_processor, pos_estimator, state_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding       
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs_wcontext_unshaped = frc_enc_context(torch.cat([frc_encs_reshaped, input_dict['context_reshaped']], dim = 1))

#         spatial_features = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, self.frc_context_enc_size).to(self.device).float()
        
#         spatial_features[input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]]\
#          += frc_encs_wcontext_unshaped[input_dict['frc_enc_idxs']]

#         spatial_enc = self.flatten(spatial_processor(spatial_features.transpose(2,3).transpose(1,2)))

#         state_logits = state_classifier(torch.cat([spatial_enc, input_dict['tool_vector']], dim = 1))

#         pos_ests = 0.01 * pos_estimator(torch.cat([spatial_enc, input_dict['tool_vector']], dim = 1))

#         return pos_ests, state_logits, spatial_enc

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests, state_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#         return {
#             'pos_est': pos_ests,
#             'state_logits': state_logits,
#             'state_inputs': logits2inputs(state_logits),
#         }

#     def get_params(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             prev_time = time.time()
            
#             T = 1

#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)

#             pos_ests, state_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

#             return enc, pos_ests + input_dict['pos_change'], F.softmax(state_logits, dim = 1).max(1)[1], input_dict['sequence_size']

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
#          - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
#         self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         # print(input_dict["batch_size"])
#         # print(input_dict["sequence_size"])

#         tool_vectors = input_dict['tool_vector'].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

#         input_dict['context_reshaped'] = torch.reshape(torch.cat([tool_vectors, input_dict['action']], dim = 2),\
#          (input_dict['batch_size'] * input_dict['sequence_size'], self.num_tools + self.action_size))

#         ##### creating_image
#         xy_poses = input_dict['rel_proprio'][:,:,:2]
#         xyz_poses = input_dict['rel_proprio'][:,1:,:3]

#         xy_poses_diff = torch.where(xy_poses[:,1:] != 0, input_dict['contact'].repeat_interleave(2, dim = 2) * (xy_poses[:,1:]\
#          - xy_poses[:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)), torch.zeros_like(xy_poses[:,1:]))

#         batch_idxs = torch.arange(input_dict['batch_size']).to(self.device).float().unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1).unsqueeze(2)
#         x_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,0] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)
#         y_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,1] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)

#         full_pixel_idxs = torch.cat([batch_idxs, x_idxs.unsqueeze(2), y_idxs.unsqueeze(2)], dim = 2)

#         xyz_poses_reshaped = torch.reshape(xyz_poses, (input_dict['batch_size'] * input_dict['sequence_size'], 3))

#         input_dict['pixel_idxs'] = torch.reshape(full_pixel_idxs, (input_dict['batch_size'] * input_dict['sequence_size'], 3)).long()

#         input_dict['frc_enc_idxs'] = torch.arange(input_dict['pixel_idxs'].size(0)).to(self.device).long()

#         input_dict['pixel_idxs'] = input_dict['pixel_idxs'][abs(xyz_poses_reshaped).sum(1) != 0]
#         input_dict['frc_enc_idxs'] = input_dict['frc_enc_idxs'][abs(xyz_poses_reshaped).sum(1) != 0]

#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_vector'] = input_dict['peg_vector'].unsqueeze(0).repeat_interleave(T, 0)
#         input_dict['contact'] = input_dict['contact'].repeat_interleave(T,0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2] - input_dict['rel_proprio'][0,0,:2]




        # x_labels =torch.clamp(self.image_size / 2 + torch.floor(-xy_poses[:,0,0] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)
        # y_labels = torch.clamp(self.image_size / 2 + torch.floor(-xy_poses[:,0,1] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)

        # input_dict['label_idxs'] = torch.cat([torch.arange(input_dict['batch_size']).to(self.device).float().unsqueeze(1),\
        #  x_labels.unsqueeze(1), y_labels.unsqueeze(1)], dim = 1).long()

        # spatial_filters = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, self.frc_context_enc_size).to(self.device).float()
        # spatial_filters[input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]]\
        #  += torch.cat([frc_encs_wcontext_unshaped], dim = 1)

        # spatial_comparison = F.conv2d(object_maps(), spatial_filters.transpose(3,2).transpose(2,1), stride=1,\
        #  padding = int(self.image_size/2)).transpose(0,1)[torch.arange(input_dict['batch_size']), input_dict['state_idx']]

        # flattened_logprobs = F.log_softmax(torch.reshape(spatial_comparison, (input_dict['batch_size'], self.image_size * self.image_size)), dim=1)
        
        # spatial_logprobs = torch.reshape(flattened_logprobs, (input_dict['batch_size'], self.image_size, self.image_size))

        # max_logprobs = torch.zeros((input_dict['batch_size'])).to(self.device).float()

        # for i in range(-self.num_pixels, self.num_pixels + 1):
        #     for j in range(-self.num_pixels, self.num_pixels + 1):
        #         height_idxs = torch.clamp(input_dict['label_idxs'][:,1] + i ,0, self.image_size-1).long()
        #         width_idxs = torch.clamp(input_dict['label_idxs'][:,2] + j ,0,self.image_size-1).long()
        #         max_logprobs += spatial_logprobs[input_dict['label_idxs'][:,0], height_idxs , width_idxs ]

        # with torch.no_grad():
        #     pos_correct = torch.zeros((input_dict['batch_size'], 2)).to(self.device).float()

        #     pos_correct = -self.reference_image[input_dict['label_idxs'][:,1], input_dict['label_idxs'][:,2]]

        #     max_logprob = spatial_logprobs.max(2)[0].max(1)[0].unsqueeze(1).unsqueeze(2).repeat((1,self.image_size, self.image_size))

        #     max_mask = torch.where(max_logprob == spatial_logprobs, torch.ones_like(spatial_logprobs), torch.zeros_like(spatial_logprobs))

        #     pos_est = -(max_mask.unsqueeze(3).repeat_interleave(2,dim=3) *\
        #      self.reference_image.unsqueeze(0).repeat_interleave(input_dict['batch_size'], dim = 0)).sum(1).sum(1)

# class History_Encoder(Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.distance_size = init_args['distance_size']
#         self.force_size = init_args['force_size']
#         self.virtual_force_size = init_args['virtual_force_size']

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.frc_context_enc_size = 42

#         self.state_size = self.frc_context_enc_size + self.virtual_force_size + self.distance_size

#         self.num_ensembles = 1

#         self.num_tl = 1
#         self.num_cl = 2
#         self.flatten = nn.Flatten()
#         self.uc = True
#         self.image_size = 32
#         self.length_scale = 0.03

#         # row_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(1).repeat_interleave(self.image_size, dim = 1).unsqueeze(2).to(self.device).float()
#         # col_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(0).repeat_interleave(self.image_size, dim = 0).unsqueeze(2).to(self.device).float()

#         # self.reference_image = (2 * self.length_scale / self.image_size) * torch.cat([row_idx, col_idx], dim = 2) + self.length_scale / self.image_size 

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_frc_context" + str(i),\
#              self.frc_enc_size + self.action_size + self.num_tools, self.frc_context_enc_size, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, nhead = 7, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

#                 CONV2DN(model_name + "_spatial_process" + str(i),\
#                     (self.frc_context_enc_size + self.virtual_force_size, self.image_size, self.image_size),\
#                     (32, 1, 1), nonlinear=False, batchnorm=True, dropout=True, dropout_prob= self.dropout_prob,\
#                     uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_pos_est" + str(i),\
#             2 * self.state_size + 32, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device)

#                 ResNetFCN(model_name + "_state_class" + str(i),\
#              self.state_size, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 FCN(model_name + "_state_embeds" + str(i), self.num_states, self.state_size,\
#                  1, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_pos_est" + str(i),\
#             2 * self.state_size + 32, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device)
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_logits(self, input_dict, model_tuple):
#         frc_enc, frc_enc_context, seq_processor, spatial_processor, state_class, state_embed, pos_est = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding       
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs_wcontext_unshaped = frc_enc_context(torch.cat([frc_encs_reshaped, input_dict['context_reshaped']], dim = 1))

#         frc_encs_wcontext = torch.reshape(frc_encs_wcontext_unshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_context_enc_size))

#         states = torch.cat([frc_encs_wcontext, input_dict['virtual_force'], input_dict['distance']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states, padding_mask = input_dict["padding_mask"]).mean(0)
#         else:
#             seq_enc = seq_processor(states).mean(0)

#         spatial_feature_map = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, self.frc_context_enc_size + self.virtual_force_size).to(self.device).float()
#         spatial_feature_map[input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]]\
#          += torch.cat([frc_encs_wcontext_unshaped, input_dict['virtual_force_reshaped']], dim = 1)

#         spatial_enc = self.flatten(spatial_processor(spatial_feature_map.transpose(3,2).transpose(2,1)))

#         state_logits = state_class(seq_enc)

#         state_embeds = state_embed(F.softmax(state_logits, dim = 1))

#         pos_ests = 0.01 * pos_est(torch.cat([seq_enc, state_embeds, spatial_enc], dim = 1))

#         return pos_ests, state_logits 

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests, state_logits = self.get_logits(input_dict, self.ensemble_list[0])

#         return {
#             'state_logits': state_logits,
#             'pos_ests': pos_ests,
#             'state_inputs': logits2inputs(state_logits),
#         }

#     def get_params(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             prev_time = time.time()
            
#             T = 100

#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)

#             pos_ests, state_logits = self.get_logits(input_dict, self.ensemble_list[0])

#             pos_mean = pos_ests.mean(0) + input_dict['pos_change']
#             pos_std = pos_ests.std(0)

#             state_probs = F.softmax(state_logits, dim = 1)

#             state_samples = torch.zeros_like(state_probs)

#             state_samples[torch.arange(T), state_probs.max(1)[1]] = 1.0

#             state_est = state_samples.mean(0)

#             # print(pos_mean, pos_std, state_est)

#             # print("Sampling time: ", time.time() - prev_time)

#             return torch.cat([pos_mean, pos_std, state_est], dim = 0), pos_mean, state_est.max(0)[1], input_dict['sequence_size']

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
#          - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
#         self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['virtual_force_reshaped'] = torch.reshape(input_dict['virtual_force'],\
#             (input_dict['batch_size'] * input_dict['sequence_size'], self.virtual_force_size))

#         # print(input_dict["batch_size"])
#         # print(input_dict["sequence_size"])

#         tool_vectors = input_dict['tool_vector'].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

#         input_dict['context_reshaped'] = torch.reshape(torch.cat([tool_vectors, input_dict['action']], dim = 2),\
#          (input_dict['batch_size'] * input_dict['sequence_size'], self.num_tools + self.action_size))

#         ##### creating_image
#         xy_poses = input_dict['rel_proprio'][:,:,:2]

#         xy_poses_diff = torch.where(xy_poses[:,1:] != 0, xy_poses[:,1:]\
#          - xy_poses[:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1), torch.zeros_like(xy_poses[:,1:]))

#         batch_idxs = torch.arange(input_dict['batch_size']).to(self.device).float().unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1).unsqueeze(2)
#         x_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,0] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)
#         y_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,1] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)

#         full_pixel_idxs = torch.cat([batch_idxs, x_idxs.unsqueeze(2), y_idxs.unsqueeze(2)], dim = 2)

#         input_dict['pixel_idxs'] = torch.reshape(full_pixel_idxs, (input_dict['batch_size'] * input_dict['sequence_size'], 3)).long()

#         xyz_poses = input_dict['rel_proprio'][:,:,:3]
#         xyz_poses_diff = xyz_poses[:,1:] - xyz_poses[:,:-1]

#         xyz_cumsum = torch.cumsum(xyz_poses_diff, dim = 1)
#         d = xyz_cumsum.norm(p=2, dim=2)

#         input_dict['distance'] = torch.cat([xyz_cumsum, d.unsqueeze(2)], dim = 2)

#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["virtual_force"] = input_dict['virtual_force'][:,1:].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_vector'] = input_dict['peg_vector'].unsqueeze(0).repeat_interleave(T, 0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2] - input_dict['rel_proprio'][0,0,:2]class History_Encoder(Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.distance_size = init_args['distance_size']
#         self.force_size = init_args['force_size']
#         self.virtual_force_size = init_args['virtual_force_size']

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.frc_context_enc_size = 42

#         self.state_size = self.frc_context_enc_size + self.virtual_force_size + self.distance_size

#         self.num_ensembles = 1

#         self.num_tl = 1
#         self.num_cl = 2
#         self.flatten = nn.Flatten()
#         self.uc = True
#         self.image_size = 32
#         self.length_scale = 0.03

#         # row_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(1).repeat_interleave(self.image_size, dim = 1).unsqueeze(2).to(self.device).float()
#         # col_idx = (torch.arange(self.image_size) - self.image_size / 2).unsqueeze(0).repeat_interleave(self.image_size, dim = 0).unsqueeze(2).to(self.device).float()

#         # self.reference_image = (2 * self.length_scale / self.image_size) * torch.cat([row_idx, col_idx], dim = 2) + self.length_scale / self.image_size 

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_frc_context" + str(i),\
#              self.frc_enc_size + self.action_size + self.num_tools, self.frc_context_enc_size, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 Transformer_Comparer(model_name + "_sequential_process" + str(i),\
#           self.state_size, self.num_tl, nhead = 7, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

#                 CONV2DN(model_name + "_spatial_process" + str(i),\
#                     (self.frc_context_enc_size + self.virtual_force_size, self.image_size, self.image_size),\
#                     (32, 1, 1), nonlinear=False, batchnorm=True, dropout=True, dropout_prob= self.dropout_prob,\
#                     uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_state_class" + str(i),\
#              self.state_size, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 FCN(model_name + "_state_embeds" + str(i), self.num_states, self.state_size,\
#                  1, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_pos_est" + str(i),\
#             2 * self.state_size + 32, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device)
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_logits(self, input_dict, model_tuple):
#         frc_enc, frc_enc_context, seq_processor, spatial_processor, state_class, state_embed, pos_est = model_tuple #origin_cov = model_tuple # spatial_processor,

#         ##### Calculating Series Encoding       
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

#         frc_encs_wcontext_unshaped = frc_enc_context(torch.cat([frc_encs_reshaped, input_dict['context_reshaped']], dim = 1))

#         frc_encs_wcontext = torch.reshape(frc_encs_wcontext_unshaped,\
#          (input_dict['batch_size'], input_dict['sequence_size'], self.frc_context_enc_size))

#         states = torch.cat([frc_encs_wcontext, input_dict['virtual_force'], input_dict['distance']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_enc = seq_processor(states, padding_mask = input_dict["padding_mask"]).mean(0)
#         else:
#             seq_enc = seq_processor(states).mean(0)

#         spatial_feature_map = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, self.frc_context_enc_size + self.virtual_force_size).to(self.device).float()
#         spatial_feature_map[input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]]\
#          += torch.cat([frc_encs_wcontext_unshaped, input_dict['virtual_force_reshaped']], dim = 1)

#         spatial_enc = self.flatten(spatial_processor(spatial_feature_map.transpose(3,2).transpose(2,1)))

#         state_logits = state_class(seq_enc)

#         state_embeds = state_embed(F.softmax(state_logits, dim = 1))

#         pos_ests = 0.01 * pos_est(torch.cat([seq_enc, state_embeds, spatial_enc], dim = 1))

#         return pos_ests, state_logits 

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         pos_ests, state_logits = self.get_logits(input_dict, self.ensemble_list[0])

#         return {
#             'state_logits': state_logits,
#             'pos_ests': pos_ests,
#             'state_inputs': logits2inputs(state_logits),
#         }

#     def get_params(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             prev_time = time.time()
            
#             T = 100

#             self.test_time_process_inputs(input_dict, T)
#             self.process_inputs(input_dict)

#             pos_ests, state_logits = self.get_logits(input_dict, self.ensemble_list[0])

#             pos_mean = pos_ests.mean(0) + input_dict['pos_change']
#             pos_std = pos_ests.std(0)

#             state_probs = F.softmax(state_logits, dim = 1)

#             state_samples = torch.zeros_like(state_probs)

#             state_samples[torch.arange(T), state_probs.max(1)[1]] = 1.0

#             state_est = state_samples.mean(0)

#             # print(pos_mean, pos_std, state_est)

#             # print("Sampling time: ", time.time() - prev_time)

#             return torch.cat([pos_mean, pos_std, state_est], dim = 0), pos_mean, state_est.max(0)[1], input_dict['sequence_size']

#     def process_inputs(self, input_dict):

#         ##### feature processing
#         input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
#          - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
#         self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))

#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         input_dict['virtual_force_reshaped'] = torch.reshape(input_dict['virtual_force'],\
#             (input_dict['batch_size'] * input_dict['sequence_size'], self.virtual_force_size))

#         # print(input_dict["batch_size"])
#         # print(input_dict["sequence_size"])

#         tool_vectors = input_dict['tool_vector'].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

#         input_dict['context_reshaped'] = torch.reshape(torch.cat([tool_vectors, input_dict['action']], dim = 2),\
#          (input_dict['batch_size'] * input_dict['sequence_size'], self.num_tools + self.action_size))

#         ##### creating_image
#         xy_poses = input_dict['rel_proprio'][:,:,:2]

#         xy_poses_diff = torch.where(xy_poses[:,1:] != 0, xy_poses[:,1:]\
#          - xy_poses[:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1), torch.zeros_like(xy_poses[:,1:]))

#         batch_idxs = torch.arange(input_dict['batch_size']).to(self.device).float().unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1).unsqueeze(2)
#         x_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,0] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)
#         y_idxs = torch.clamp(self.image_size / 2 + torch.floor(xy_poses_diff[:,:,1] / (2 * self.length_scale / self.image_size)), 0, self.image_size - 1)

#         full_pixel_idxs = torch.cat([batch_idxs, x_idxs.unsqueeze(2), y_idxs.unsqueeze(2)], dim = 2)

#         input_dict['pixel_idxs'] = torch.reshape(full_pixel_idxs, (input_dict['batch_size'] * input_dict['sequence_size'], 3)).long()

#         xyz_poses = input_dict['rel_proprio'][:,:,:3]
#         xyz_poses_diff = xyz_poses[:,1:] - xyz_poses[:,:-1]

#         xyz_cumsum = torch.cumsum(xyz_poses_diff, dim = 1)
#         d = xyz_cumsum.norm(p=2, dim=2)

#         input_dict['distance'] = torch.cat([xyz_cumsum, d.unsqueeze(2)], dim = 2)

#     def test_time_process_inputs(self, input_dict, T):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
#         input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
#         input_dict["virtual_force"] = input_dict['virtual_force'][:,1:].repeat_interleave(T, 0)
#         input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
#         input_dict['tool_vector'] = input_dict['peg_vector'].unsqueeze(0).repeat_interleave(T, 0)

#         input_dict['pos_change'] = input_dict['rel_proprio'][0,-1,:2] - input_dict['rel_proprio'][0,0,:2]

        # print(likelihood_logprob.size())

        # print(likelihood_logprob)

        # print("Peg Type: ", peg_type)
        # print("Option_Type: ", option_type)
        # print("Macro action: ", macro_action)
        # print("Obs Idx: ", obs_idx)
        # print("likelihood probs: ", F.softmax(likelihood_logits, dim = 1))
        # print("likelihood probs less: ", torch.exp(likelihood_logprobs))
        # print("likelihood logprobs: ", likelihood_logprobs)
        # print("likelihood logprob: ", likelihood_logprob)

        # return likelihood_logprob


        # if s_size < 3:
        #     xy_2pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_4pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        # elif s_size < 5:
        #     xy_2pos = torch.cat([\
        #         torch.zeros((b_size, 1, 2)).to(self.device).float(),
        #         xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
        #     xy_4pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        # elif s_size < 9:
        #     xy_2pos = torch.cat([\
        #         torch.zeros((b_size, 1, 2)).to(self.device).float(),
        #         xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
        #     xy_4pos = torch.cat([\
        #         torch.zeros((b_size, 3, 2)).to(self.device).float(),
        #         xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

        #     xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        #     xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        # elif s_size < 17:
        #     xy_2pos = torch.cat([\
        #         torch.zeros((b_size, 1, 2)).to(self.device).float(),
        #         xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
        #     xy_4pos = torch.cat([\
        #         torch.zeros((b_size, 3, 2)).to(self.device).float(),
        #         xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

        #     xy_8pos = torch.cat([\
        #         torch.zeros((b_size, 7, 2)).to(self.device).float(),
        #         xy_pos[:,8:] - xy_pos[:,:-8]], dim = 1)

        #     xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        # else:
        #     xy_2pos = torch.cat([\
        #         torch.zeros((b_size, 1, 2)).to(self.device).float(),
        #         xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
        #     xy_4pos = torch.cat([\
        #         torch.zeros((b_size, 3, 2)).to(self.device).float(),
        #         xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

        #     xy_8pos = torch.cat([\
        #         torch.zeros((b_size, 7, 2)).to(self.device).float(),
        #         xy_pos[:,8:] - xy_pos[:,:-8]], dim = 1)

        #     xy_16pos = torch.cat([\
        #         torch.zeros((b_size, 15, 2)).to(self.device).float(),
        #         xy_pos[:,16:] - xy_pos[:,:-16]], dim = 1)


        # input_dict["rel_proprio_diff"] = torch.cat([input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1],\
        #     xy_2pos, xy_4pos, xy_8pos, xy_16pos], dim = 2)

        # roll_list = [[1],[2],[3],[4],[5,6,7,8],[9,10,11,12],\
        # [13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35],[36,37,38,39,40],\
        # [41,42,43,44,45],[46,47,48,49,50],[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]

        # for i in range(2, self.frc_enc_size // interval):
        #     roll_index = i - 2

        #     roll = random.choice(roll_list[roll_index])

        #     frc_features = frc_encs[:,:,i * interval : (i+1) * interval]
        #     frc_diffs = frc_features - torch.roll(frc_features, roll, dims = 1)
        #     distance_diffs = distances - torch.roll(distances, roll, dims = 1)

        #     frc_diffs[:,:roll] = 0.0

        #     states[:,:,i * interval : (i+1) * interval] = frc_diffs * distance_diffs

        # xyz_poses = input_dict['rel_proprio'][:,:,:3]
        # xyz_poses_diff = (xyz_poses[:,1:] - xyz_poses[:,:-1])

        # xyz_cumsum = torch.cumsum(xyz_poses_diff, dim = 1)
        # d = xyz_cumsum.norm(p=2, dim=2)

        # input_dict['distances'] = torch.cat([xyz_cumsum, d.unsqueeze(2)], dim = 2)

        # input_dict["num_steps"] = input_dict["distances"].size(1)

        # b_size = input_dict['distances'].size(0)
        # s_size = input_dict['distances'].size(1)

        # input_dict['s_size'] = s_size

        # input_dict["contact"] = torch.cat([\
        #     input_dict["contact"][:,1:],\
        #     torch.zeros((b_size, 101 - s_size, input_dict['contact'].size(2))).to(self.device).float()], dim = 1)

        # input_dict["force_hi_freq"] = torch.cat([\
        #     input_dict["force_hi_freq"][:,1:],\
        #     torch.zeros((b_size, 101 - s_size, input_dict['force_hi_freq'].size(2), input_dict['force_hi_freq'].size(3))).to(self.device).float()], dim = 1)

        # input_dict["action"] = torch.cat([\
        #     input_dict["action"][:, :-1],
        #     torch.zeros((b_size, 101- s_size, input_dict['action'].size(2))).to(self.device).float()], dim = 1)

        # input_dict['distances'] = torch.cat([\
        #     input_dict['distances'],
        #     torch.zeros((b_size, 101-s_size,  input_dict['distances'].size(2))).to(self.device).float()], dim = 1)

        # input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].unsqueeze(0).repeat_interleave(b_size, 0)

        # input_dict['padding_mask'] = torch.cat([\
        #     torch.zeros((b_size, s_size)),\
        #      torch.ones((b_size, 101 - s_size))], dim = 1).to(self.device).bool()


        # lb = 73
        # ub = 77

        # print("Mask", input_dict['image_mask'][0,lb:ub,lb:ub])

        # print("Estimate: ", pos_ests_image[0,lb:ub,lb:ub,0])
        # print("Estimate: ",pos_ests_image[0,lb:ub,lb:ub,1])

        # print("Label: ", input_dict['xy_pos_image'][0,lb:ub,lb:ub,0])
        # print("Label: ", input_dict['xy_pos_image'][0,lb:ub,lb:ub,1])

        # test_image = (self.reference_image + input_dict['rel_proprio'][0,0,:2].unsqueeze(0).unsqueeze(1).repeat_interleave(self.image_size, dim = 0)\
        #     .repeat_interleave(self.image_size, dim = 1)) * input_dict['image_mask'][0].unsqueeze(2).repeat_interleave(2, dim = 2)

        # print("Test: ", test_image[lb:ub,lb:ub,0])
        # print("Test: ", test_image[lb:ub,lb:ub,1])

        # print("rel pos", input_dict['rel_proprio'][0,0,:2])      

        # test_normed = (test - input_dict['rel_proprio'][:,0,:2].unsqueeze(1).unsqueeze(2).repeat_interleave(self.image_size,dim=1).repeat_interleave(self.image_size,dim=2)).norm(p=2,dim=2)

        # test_mean = test_normed.sum(1).sum(1) / input_dict['image_mask'].sum(1).sum(1)

        # print("Error: ", test_mean.mean())

        # pos_ests_image = 0.01 * processed_features[:,:,:,:2] *  input_dict['image_mask'].unsqueeze(3).repeat_interleave(2, dim = 3)

        # state_logits_image = processed_features[:,:,:,2:] * input_dict['image_mask'].unsqueeze(3).repeat_interleave(self.num_states, dim = 3)


        # input_dict['xy_pos_image'] = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, 2).to(self.device).float()
        # input_dict['xy_pos_image'][input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]]\
        #  = torch.reshape(xy_poses[:,1:], (input_dict['batch_size'] * input_dict['sequence_size'], xy_poses.size(2)))


        # input_dict['image_mask'] =  torch.zeros(input_dict['batch_size'], self.image_size, self.image_size).to(self.device).float()

        # # print(input_dict['image_mask'].size())

        # input_dict['image_mask'][input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1], input_dict['pixel_idxs'][:,2]] = 1.0

        # # print(input_dict['image_mask'].size())

        # state_idx_reshaped = torch.reshape(input_dict['state_idx'].unsqueeze(1)\
        #     .repeat_interleave(input_dict['sequence_size'],dim=1), (input_dict['batch_size']*input_dict['sequence_size'],1)).long()

        # input_dict['state_image'] = torch.zeros(input_dict['batch_size'], self.image_size, self.image_size, 3).to(self.device).float()

        # # print(input_dict['state_image'].size())

        # input_dict['state_image'][input_dict['pixel_idxs'][:,0], input_dict['pixel_idxs'][:,1],\
        #  input_dict['pixel_idxs'][:,2], state_idx_reshaped[:,0]] = 1.0

        # # print(input_dict['state_image'].size())
        
        # # print(input_dict['image_mask']\
        # # .unsqueeze(3).repeat_interleave(self.num_states, dim = 3).size())

        # input_dict['state_label'] = input_dict['state_image'][input_dict['image_mask']\
        # .unsqueeze(3).repeat_interleave(self.num_states, dim = 3).nonzero(as_tuple=True)]

        # # print(input_dict['state_image'].size())

        # # print(input_dict['xy_pos_image'].size())


# class History_Encoder(Proto_Macromodel):
#     def __init__(self, model_name, init_args, device = None):
#         super().__init__()

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = init_args['num_tools']
#         self.num_states = init_args['num_states']

#         self.action_size = init_args['action_size']
#         self.distance_size = init_args['distance_size']
#         self.force_size = init_args['force_size']
#         self.virtual_force_size = init_args['virtual_force_size']

#         self.dropout_prob = init_args['dropout_prob']

#         self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
#         self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

#         self.frc_enc_size = 48
#         self.frc_context_enc_size = 57

#         self.state_size = self.frc_context_enc_size + self.distance_size + self.virtual_force_size

#         self.num_ensembles = 3

#         self.num_tl = 1
#         self.num_cl = 2
#         self.flatten = nn.Flatten()
#         self.uc = True

#         for i in range(self.num_ensembles):
#             self.ensemble_list.append((\
#                 CONV1DN(model_name + "_frc_enc" + str(i),\
#                  (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
#                   nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
#                    uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_frc_context" + str(i),\
#              self.frc_enc_size + self.action_size + self.num_tools, self.frc_context_enc_size, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 Transformer_Comparer(model_name + "_state_transdec" + str(i),\
#           self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_state_class" + str(i),\
#              self.state_size, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\

#                 FCN(model_name + "_state_embeds" + str(i), self.num_states, self.state_size,\
#                  1, device = self.device).to(self.device),\

#                 ResNetFCN(model_name + "_pos_est" + str(i),\
#             2 *  self.state_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
#             uc = self.uc, device = self.device).to(self.device),\
#             ))

#         for model_tuple in self.ensemble_list:
#             for model in model_tuple:
#                 self.model_list.append(model)

#     def get_logits(self, input_dict, model_tuple):
#         frc_enc, frc_enc_context, state_transdec, state_class, state_embeds, pos_estimator = model_tuple #origin_cov = model_tuple # 

#         ##### Calculating Series Encoding       
#         frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
#         frc_encs_wcontext_unshaped = frc_enc_context(torch.cat([frc_encs_reshaped, input_dict['context_reshaped']], dim = 1))

#         frc_encs = torch.reshape(frc_encs_wcontext_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_context_enc_size))

#         states = torch.cat([frc_encs, input_dict['distance'], input_dict['virtual_force']], dim = 2).transpose(0,1)

#         if "padding_mask" in input_dict.keys():
#             seq_encs = state_transdec(states, padding_mask = input_dict["padding_mask"]).mean(0)
#         else:
#             seq_encs = state_transdec(states).mean(0)

#         # seq_encs[:,:self.frc_enc_size] += frc_encs.max(1)[0]

#         #### Calculating State Logits
#         state_logits = state_class(seq_encs)

#         hole_embeds = state_embeds(F.softmax(state_logits, dim = 1))

#         pos_ests = 0.02 * torch.tanh(pos_estimator(torch.cat([seq_encs, hole_embeds], dim = 1)))

#         return state_logits, pos_ests

#     def getall_logits(self, input_dict):

#         ol_list = []
#         pe_list = []

#         for i in range(self.num_ensembles):
#             ol, pe = self.get_logits(input_dict, self.ensemble_list[i])

#             ol_list.append(ol)
#             pe_list.append(pe)

#         return ol_list, pe_list #, seq_encs ol_list #

#     def forward(self, input_dict):
#         self.process_inputs(input_dict)

#         ol_list, pe_list = self.getall_logits(input_dict)

#         inputs_list =[]

#         for i in range(self.num_ensembles):
#             inputs_list.append(logits2inputs(ol_list[i]).unsqueeze(0))
#             ol_list[i] = ol_list[i].unsqueeze(0)
#             pe_list[i] = pe_list[i].unsqueeze(0)

#         return {
#             'options_class': torch.cat(ol_list, dim = 0),
#             'rel_pos_est': torch.cat(pe_list, dim = 0),
#             'options_inputs': torch.cat(inputs_list, dim = 0),
#         }

#     def pos_unest(self, input_dict):
#         with torch.no_grad():
#             self.eval()
#             # prev_time = time.time()

#             self.test_time_process_inputs(input_dict)
#             self.process_inputs(input_dict)

#             T = 2
#             pe_list = []

#             for i in range(T):
#                 _, pe_list_samples = self.getall_logits(input_dict)

#                 pe_list += pe_list_samples

#             pos_ests = torch.cat(pe_list, dim = 0)

#             pos_est_mean = torch.mean(pos_ests, dim = 0)
#             pos_est_var = torch.std(pos_ests, dim = 0)

#             # print("Time Elapsed", time.time() - prev_time)
#             return torch.cat([pos_est_mean, pos_est_var], dim = 0), pos_est_mean, input_dict["num_steps"]

#     def process_inputs(self, input_dict):
#         input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
#          - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
#         self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
#          .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
#         input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
#          (input_dict["force"].size(0) * input_dict["force"].size(1), \
#          input_dict["force"].size(2), input_dict["force"].size(3)))
        
#         input_dict["batch_size"] = input_dict["force"].size(0)
#         input_dict["sequence_size"] = input_dict["force"].size(1)

#         xyz_poses = input_dict['rel_proprio'][:,:,:3]
#         xyz_poses_diff = xyz_poses[:,1:] - xyz_poses[:,:-1]

#         xyz_cumsum = torch.cumsum(xyz_poses_diff, dim = 1)
#         d = xyz_cumsum.norm(p=2, dim=2)

#         input_dict['distance'] = torch.cat([xyz_cumsum, d.unsqueeze(2)], dim = 2)

#         tool_vectors = input_dict['tool_vector'].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

#         input_dict['context_reshaped'] = torch.reshape(torch.cat([tool_vectors, input_dict['action']], dim = 2),\
#          (input_dict['batch_size'] * input_dict['sequence_size'], self.num_tools + self.action_size))
         
#     def test_time_process_inputs(self, input_dict):

#         input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
#         input_dict["action"] = input_dict["action"][:, :-1]
#         input_dict['tool_vector'] = input_dict['peg_vector'].unsqueeze(0).repeat_interleave(input_dict["force_hi_freq"].size(0), 0)
