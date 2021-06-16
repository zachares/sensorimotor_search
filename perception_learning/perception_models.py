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

# commented out models in old experimental models
def get_ref_model_dict():
    return {
        'History_Encoder': History_Encoder,
        # 'Variational_History_Encoder': Variational_History_Encoder,
        # 'Selfsupervised_History_Encoder': Selfsupervised_History_Encoder,
        # 'Unsupervised_History_Encoder': Unsupervised_History_Encoder,
        # 'StatePosSensor_wUncertainty': StatePosSensor_wUncertainty,
        'StatePosSensor_wConstantUncertainty' : StatePosSensor_wConstantUncertainty,
        # 'History_Encoder_Baseline': History_Encoder_Baseline,
        # 'Voting_Policy' : Voting_Policy,
    }

######################################
# Defining Custom Macromodels for project
#######################################

#### All models must have three input arguments: model_name, init_args, device
# def sample_gaussian(m, v, device):
    
#     epsilon = Normal(0, 1).sample(m.size())
#     z = m + torch.sqrt(v) * epsilon.to(device)

#     return z

# def gaussian_parameters(h, dim=-1):

#     m, h = torch.split(h, h.size(dim) // 2, dim=dim)
#     v = F.softplus(h) + 1e-8
#     return m, v

class History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__(model_name)

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
        self.tool_dim = 64

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False
        self.pos_input = self.state_size + self.tool_dim + init_args['residual_dims']

        if init_args['residual_dims'] > 0:
            residual_string = '_residual'
        else:
            residual_string = ''

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

                mm.ResNetFCN(model_name + "_pos_est" + residual_string + str(i),\
            self.pos_input, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
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

        # print(frc_encs_reshaped[:100,:10])

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t.transpose(0,1), padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t.transpose(0,1)).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        if self.pos_input == (self.state_size + self.tool_dim):
            pos_ests = input_dict['final_pos'] - pos_estimator(states_T)
        else:
            pos_ests = input_dict['final_pos'] - pos_estimator(torch.cat([states_T, input_dict['rel_pos_estimate']], dim = 1))\
             + input_dict['rel_pos_estimate']

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

            return pos_ests.squeeze().cpu().numpy(), None

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
        
        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['idxs_flat'] = np.arange(input_dict['batch_size'] * input_dict['sequence_size'])

        input_dict['idxs_batch'], input_dict['idxs_sequence'] = np.unravel_index(input_dict['idxs_flat'], (input_dict['batch_size'], input_dict['sequence_size']))

        input_dict["force_reshaped"] = torch.zeros((input_dict['batch_size']*input_dict['sequence_size'],\
         input_dict['force'].size(2), input_dict['force'].size(3))).float().to(self.device)

        input_dict["force_reshaped"][input_dict['idxs_flat']] = input_dict["force"][input_dict['idxs_batch'], input_dict['idxs_sequence']]

        # input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
        #  (input_dict["force"].size(0) * input_dict["force"].size(1), \
        #  input_dict["force"].size(2), input_dict["force"].size(3)))

        # input_dict["batch_size"] = input_dict["force"].size(0)
        # input_dict["sequence_size"] = input_dict["force"].size(1)

        if 'rel_proprio' in input_dict.keys():
            key = 'rel_proprio'
        else:
            key = 'proprio'
            
        input_dict['proprio_diff'] = torch.where(input_dict[key][:,1:] != 0,\
         input_dict[key][:,1:] - input_dict[key][:,:-1], torch.zeros_like(input_dict[key][:,1:]))

        # input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
        #  input_dict['proprio'][:,1:] - input_dict['proprio'][:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], 1)\
        #  , torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        # input_dict["rel_proprio"] = input_dict['proprio']
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['state_idx'] = input_dict['hole_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

        input_dict['rel_pos_estimate'] = 100 * (input_dict['proprio'][:,-1,:2] - input_dict['proprio'][:,0,:2]).repeat_interleave(T,0)
        input_dict['final_pos'] = 100 * input_dict['proprio'][:,-1,:2]

class StatePosSensor_wConstantUncertainty(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__(model_name)

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

        self.frc_enc_size = 48
        self.tool_dim = 64

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False
        self.pos_input = self.state_size + self.tool_dim + init_args['residual_dims']

        if init_args['residual_dims'] > 0:
            residual_string = '_residual'
        else:
            residual_string = ''

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

                mm.ResNetFCN(model_name + "_pos_est" + residual_string + str(i),\
            self.pos_input, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_pos_est_obs_noise" + str(i),\
                    self.num_tools, 2, device= self.device).to(self.device),\

            #     mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
            # self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            # uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(0),\
            2 * self.state_size, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
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

        frc_encs = torch.zeros((input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size)).float().to(self.device)
        frc_encs[input_dict['idxs_batch'], input_dict['idxs_sequence']] = frc_encs_reshaped[input_dict['idxs_flat']]

        # frc_encs = torch.reshape(frc_encs_reshaped,\
        #  (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2)

        if 'trans_comparer' in seq_processor.model_name: # Transformer Padded
            if "padding_mask" in input_dict.keys():
                # step_mask = F.dropout(torch.ones((input_dict['batch_size'],\
                #  input_dict['sequence_size'])).to(self.device).float(), p=0.5).unsqueeze(2).repeat_interleave(states_t.size(2), 2)
                
                # seq_enc = seq_processor(states_t.transpose(0,1), padding_mask = input_dict["padding_mask"]).max(0)[0]

                seq_enc = seq_processor(states_t.transpose(0,1), padding_mask = input_dict["padding_mask"]).max(0)[0]
            else:
                seq_enc = seq_processor(states_t.transpose(0,1)).max(0)[0]
        else:
            raise Exception('Unsupported Encoder Type')
            
        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        if self.pos_input == (self.state_size + self.tool_dim):
            pos_ests_obs = input_dict['final_pos'] - pos_estimator(states_T)
        else:
            pos_ests_obs_residual = pos_estimator(torch.cat([states_T, input_dict['rel_pos_estimate']], dim = 1))

            pos_ests_obs = input_dict['final_pos'] - (input_dict['rel_pos_estimate'] + pos_ests_obs_residual)

        # pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2) + 1e-2

        # small constant added to avoid numerical issues during training
        pos_ests_obs_noise = obs_noise_estimator(input_dict['tool_idx'].long()).pow(2) + 1e-2

        #new_states_T = torch.cat([seq_enc, shape_embed(input_dict['new_tool_idx'])], dim = 1)

        if 'new_tool_idx' in input_dict.keys():
            new_states_T = torch.cat([seq_enc, shape_embed(input_dict['new_tool_idx'].long())], dim = 1)
            old_obs_logits = obs_classifier(states_T)
            new_obs_logits = obs_classifier(new_states_T)
            obs_logits = torch.cat([old_obs_logits, new_obs_logits], dim = 0)
            input_dict['full_fit_idx'] = torch.cat([input_dict['fit_idx'], input_dict['new_fit_idx']], dim = 0)
        else:
            obs_logits = obs_classifier(states_T)
            if 'fit_idx' in input_dict.keys():
                input_dict['full_fit_idx'] = input_dict['fit_idx']
        # likelihood network
        obs_state_logits = torch.reshape(obs_likelihood(input_dict['tool_idx'].long()), (input_dict['batch_size'], self.num_obs, self.num_states))
        obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)
        
        if not obs_classifier.training:
            state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_classifier(states_T), dim = 1).max(1)[1]]
        else:
            obs_probs_test = F.softmax(obs_classifier(states_T), dim = 1)
            obs_state_probs = F.softmax(obs_state_logits, dim=1)
            state_logprobs = torch.log((obs_state_probs * obs_probs_test.unsqueeze(2).repeat_interleave(self.num_states, dim = 2)).sum(1))

        return pos_ests_obs, pos_ests_obs_noise, obs_logits, state_logprobs, obs_state_logprobs, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

        prior_noise = input_dict['pos_prior_var']

        y = pos_ests_mean - input_dict['pos_prior_mean']
        S = prior_noise + pos_ests_obs_noise
        K = prior_noise / S

        pos_post = input_dict['pos_prior_mean'] + K * y
        pos_post_var = (1 - K) * prior_noise

        state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

        return {
            'pos_est': pos_post,
            'pos_est_params': (pos_post, 1 / pos_post_var),
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'fit_idx' : input_dict['full_fit_idx'],
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

            # obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            # print("Pos Error", (0.01 * pos_ests_mean - input_dict['rel_proprio'][0,-1,:2]).norm(p=2).item())
            # print("Uncertainty ", pos_ests_obs_noise.squeeze().cpu().numpy())
            return pos_ests_mean.squeeze().cpu().numpy(), pos_ests_obs_noise.squeeze().cpu().numpy()

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
        
        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['idxs_flat'] = np.arange(input_dict['batch_size'] * input_dict['sequence_size'])

        input_dict['idxs_batch'], input_dict['idxs_sequence'] = np.unravel_index(input_dict['idxs_flat'], (input_dict['batch_size'], input_dict['sequence_size']))

        input_dict["force_reshaped"] = torch.zeros((input_dict['batch_size']*input_dict['sequence_size'],\
         input_dict['force'].size(2), input_dict['force'].size(3))).float().to(self.device)

        input_dict["force_reshaped"][input_dict['idxs_flat']] = input_dict["force"][input_dict['idxs_batch'], input_dict['idxs_sequence']]

        # input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
        #  (input_dict["force"].size(0) * input_dict["force"].size(1), \
        #  input_dict["force"].size(2), input_dict["force"].size(3)))

        # input_dict["batch_size"] = input_dict["force"].size(0)
        # input_dict["sequence_size"] = input_dict["force"].size(1)

        if 'rel_proprio' in input_dict.keys():
            key = 'rel_proprio'
        else:
            key = 'proprio'
            
        input_dict['proprio_diff'] = torch.where(input_dict[key][:,1:] != 0,\
         input_dict[key][:,1:] - input_dict[key][:,:-1], torch.zeros_like(input_dict[key][:,1:]))
        # input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
        #  input_dict['proprio'][:,1:] - input_dict['proprio'][:,0].unsqueeze(1).repeat_interleave(input_dict['sequence_size'], 1)\
        #  , torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        # input_dict["rel_proprio"] = input_dict['proprio']
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['state_idx'] = input_dict['hole_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        #input_dict['new_tool_idx'] = input_dict['tool_idx']
        input_dict['rel_pos_estimate'] = 100 * (input_dict['proprio'][:,-1,:2] - input_dict['proprio'][:,0,:2]).repeat_interleave(T,0)
        input_dict['final_pos'] = 100 * input_dict['proprio'][:,-1,:2]
