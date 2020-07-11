import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np
import time
from models_modules import *
import yaml
import itertools
import random
from multinomial import *
from collections import OrderedDict

import sys

sys.path.insert(0, "../") 

from project_utils import *

def get_ref_model_dict():
    return {
        'Options_Sensor': Options_Sensor,
        'Options_LikelihoodNet': Options_LikelihoodNet,
        'History_Encoder': History_Encoder,
    }

######################################
# Defining Custom Macromodels for project
#######################################

#### All models must have three input arguments: model_name, init_args, device
class History_Encoder(Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']

        self.action_dim = init_args['action_dim']
        self.proprio_size = init_args['proprio_size'] + 8
        self.force_size = init_args['force_size']

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.num_ensembles = 1

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim

        self.num_tl = 4
        self.num_cl = 1
        self.flatten = nn.Flatten()
        self.uc = False

        self.tool_dim = 48

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                Embedding(model_name + "_tool_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                ResNetFCN(model_name + "_frc_enc_context" + str(i),\
            self.frc_enc_size + self.tool_dim, self.frc_enc_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                Transformer_Comparer(model_name + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                FCN(model_name + "_transformation" + str(i), self.state_size, self.state_size,\
                 1, device = self.device).to(self.device),\

                Embedding(model_name + "_hole_embed" + str(i),\
                         self.num_states, self.state_size, device= self.device).to(self.device),\

                ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_logits(self, input_dict, model_tuple):
        tool_embed, frc_enc, frc_enc_context, state_transdec, transformation, hole_embed, pos_estimator = model_tuple #origin_cov = model_tuple # 

        ##### Calculating Series Encoding
        # Step 1. frc encoding
        tool_embeds = tool_embed(input_dict["tool_idx"].long()).unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

        tool_embeds_unshaped = torch.reshape(tool_embeds, (input_dict['batch_size']*input_dict['sequence_size'], tool_embeds.size(2)))

        frc_encs_unshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
        frc_encs_context_unshaped = frc_enc_context(torch.cat([frc_encs_unshaped, tool_embeds_unshaped], dim = 1))
        frc_encs = torch.reshape(frc_encs_context_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_enc_size))

        states_t = torch.cat([input_dict["states"], frc_encs, input_dict["action"]], dim = 2).transpose(0,1)

        if "padding_mask" not in input_dict.keys():
            seq_encs = state_transdec(states_t).max(0)[0]
        else:
            seq_encs = state_transdec(states_t, padding_mask = input_dict["padding_mask_extended"]).max(0)[0]

        #### Calculating Options Logits
        hole0_embed = hole_embed(torch.zeros((input_dict['batch_size'])).to(self.device).long())
        hole1_embed = hole_embed(torch.ones((input_dict['batch_size'])).to(self.device).long())
        hole2_embed = hole_embed(2 * torch.ones((input_dict['batch_size'])).to(self.device).long())

        seq_pos = transformation(seq_encs)

        options_logits = torch.cat([\
            ((seq_encs - seq_pos)  * hole0_embed).sum(1).unsqueeze(1),\
             ((seq_encs - seq_pos) * hole1_embed).sum(1).unsqueeze(1),\
             ((seq_encs - seq_pos) * hole2_embed).sum(1).unsqueeze(1)], dim = 1)

        hole_embeds = hole_embed(input_dict['state_idx'].long())

        pos_ests = 0.01 * pos_estimator(seq_pos + hole_embeds)

        return options_logits, pos_ests

    def getall_logits(self, input_dict):

        ol_list = []
        pe_list = []

        for i in range(self.num_ensembles):
            if "padding_mask" in input_dict.keys():
                input_dict["padding_mask_extended"] = self.get_input_dropout(input_dict["padding_mask"])

            ol, pe = self.get_logits(input_dict, self.ensemble_list[i])

            ol_list.append(ol)
            pe_list.append(pe)

        return ol_list, pe_list #, seq_encs ol_list #

    def get_input_dropout(self, padding_masks):
        # print("Padding mask\n", padding_masks[0])
        input_dropout = F.dropout(torch.ones(padding_masks.size()).float().to(self.device), p = (1 - (self.dropout_prob / 1))).bool()

        padding_masks_extended = torch.where(padding_masks == False, input_dropout, padding_masks)

        # print("New Padding mask\n", padding_masks_extended[0])

        return  padding_masks_extended

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        ol_list, pe_list = self.getall_logits(input_dict)

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(ol_list[i]).unsqueeze(0))
            ol_list[i] = ol_list[i].unsqueeze(0)
            pe_list[i] = pe_list[i].unsqueeze(0)

        return {
            'options_class': torch.cat(ol_list, dim = 0),
            'rel_pos_est': torch.cat(pe_list, dim = 0),
            'options_inputs': torch.cat(inputs_list, dim = 0),
        }

    def enc(self, input_dict):
        with torch.no_grad():
            self.eval()

            self.test_time_process_inputs(input_dict)
            self.process_inputs(input_dict)

            tool_embed, frc_enc, frc_enc_context, state_transdec, transformation, hole_embed, pos_estimator = self.ensemble_list[0] #origin_cov = model_tuple # 

            ##### Calculating Series Encoding
            # Step 1. frc encoding
            tool_embeds = tool_embed(input_dict["tool_idx"].long()).unsqueeze(0).unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)
            # print(tool_embeds.size())
            tool_embeds_unshaped = torch.reshape(tool_embeds, (input_dict['batch_size']*input_dict['sequence_size'], tool_embeds.size(2)))

            frc_encs_unshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
            frc_encs_context_unshaped = frc_enc_context(torch.cat([frc_encs_unshaped, tool_embeds_unshaped], dim = 1))
            frc_encs = torch.reshape(frc_encs_context_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_enc_size))

            states_t = torch.cat([input_dict["states"], frc_encs, input_dict["action"]], dim = 2).transpose(0,1)

            if "padding_mask" not in input_dict.keys():
                seq_encs = state_transdec(states_t).max(0)[0]
            else:
                seq_encs = state_transdec(states_t, padding_mask = input_dict["padding_mask_extended"]).max(0)[0]

            #### Calculating Options Logits
            hole0_embed = hole_embed(torch.zeros((input_dict['batch_size'])).to(self.device).long())
            hole1_embed = hole_embed(torch.ones((input_dict['batch_size'])).to(self.device).long())
            hole2_embed = hole_embed(2 * torch.ones((input_dict['batch_size'])).to(self.device).long())

            seq_pos = transformation(seq_encs)

            options_logits = torch.cat([\
                ((seq_encs - seq_pos)  * hole0_embed).sum(1).unsqueeze(1),\
                 ((seq_encs - seq_pos) * hole1_embed).sum(1).unsqueeze(1),\
                 ((seq_encs - seq_pos) * hole2_embed).sum(1).unsqueeze(1)], dim = 1)

            options_probs = F.softmax(options_logits, dim = 1)

            hole_embeds_all = torch.cat([hole0_embed.unsqueeze(1), hole1_embed.unsqueeze(1), hole2_embed.unsqueeze(1)], dim = 1)

            hole_embeds = hole_embeds_all[torch.arange(input_dict['batch_size']), options_probs.max(1)[1]]

            # print((hole_embeds + seq_pos).size())

            return (hole_embeds + seq_pos).squeeze(0)

    def process_inputs(self, input_dict):
        input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
         - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
         .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
         .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
         .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
        self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
         .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
         .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
         .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["states"] = torch.cat([input_dict["rel_proprio_diff"], input_dict["contact_diff"]], dim = 2)
        
        input_dict["batch_size"] = input_dict["rel_proprio_diff"].size(0)
        input_dict["sequence_size"] = input_dict["rel_proprio_diff"].size(1)
         
    def test_time_process_inputs(self, input_dict):

        # print(input_dict.keys())
        xy_pos = input_dict['rel_proprio'][:,:,:2]
        s_size = input_dict['rel_proprio'].size(1)
        b_size = input_dict['rel_proprio'].size(0)

        if s_size < 3:
            xy_2pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_4pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        elif s_size < 5:
            xy_2pos = torch.cat([\
                torch.zeros((b_size, 1, 2)).to(self.device).float(),
                xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
            xy_4pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        elif s_size < 9:
            xy_2pos = torch.cat([\
                torch.zeros((b_size, 1, 2)).to(self.device).float(),
                xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
            xy_4pos = torch.cat([\
                torch.zeros((b_size, 3, 2)).to(self.device).float(),
                xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

            xy_8pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
            xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()

        elif s_size < 17:
            xy_2pos = torch.cat([\
                torch.zeros((b_size, 1, 2)).to(self.device).float(),
                xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
            xy_4pos = torch.cat([\
                torch.zeros((b_size, 3, 2)).to(self.device).float(),
                xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

            xy_8pos = torch.cat([\
                torch.zeros((b_size, 7, 2)).to(self.device).float(),
                xy_pos[:,8:] - xy_pos[:,:-8]], dim = 1)

            xy_16pos = torch.zeros((b_size, s_size - 1, 2)).to(self.device).float()
        else:
            xy_2pos = torch.cat([\
                torch.zeros((b_size, 1, 2)).to(self.device).float(),
                xy_pos[:,2:] - xy_pos[:,:-2]], dim = 1)
            
            xy_4pos = torch.cat([\
                torch.zeros((b_size, 3, 2)).to(self.device).float(),
                xy_pos[:,4:] - xy_pos[:,:-4]], dim = 1)

            xy_8pos = torch.cat([\
                torch.zeros((b_size, 7, 2)).to(self.device).float(),
                xy_pos[:,8:] - xy_pos[:,:-8]], dim = 1)

            xy_16pos = torch.cat([\
                torch.zeros((b_size, 15, 2)).to(self.device).float(),
                xy_pos[:,16:] - xy_pos[:,:-16]], dim = 1)


        input_dict["rel_proprio_diff"] = torch.cat([input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1],\
            xy_2pos, xy_4pos, xy_8pos, xy_16pos], dim = 2)

        # print(input_dict['rel_proprio_diff'].size())
        input_dict["contact_diff"] = input_dict["contact"][:,1:] - input_dict["contact"][:, :-1]
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
        input_dict["action"] = input_dict["action"][:, :-1]
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1]

class Options_Sensor(Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_options = init_args['num_options']

        self.action_dim = init_args['action_dim']
        self.proprio_size = init_args['proprio_size'] + 8
        self.force_size = init_args['force_size']

        self.tool_dim = init_args['tool_dim']

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(init_args['force_mean']).to(self.device).float()
        self.force_std = torch.from_numpy(init_args['force_std']).to(self.device).float()

        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.num_ensembles = 3

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim

        self.num_tl = 4
        self.num_cl = 2
        self.flatten = nn.Flatten()
        self.uc = True

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                Embedding(model_name + "_tool_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                ResNetFCN(model_name + "_frc_enc_context" + str(i),\
            self.frc_enc_size + self.tool_dim, self.frc_enc_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                Transformer_Comparer(model_name + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                ResNetFCN(model_name + "_options_class" + str(i),\
            self.state_size, self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device)
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_logits(self, input_dict, model_tuple):
        tool_embed, frc_enc, frc_enc_context, state_transdec, states_class = model_tuple #origin_cov = model_tuple # 

        ##### Calculating Series Encoding
        # Step 1. frc encoding
        # print(input_dict["forces_reshaped"].size())
        # print(input_dict["forces_reshaped"].max())
        # print(input_dict["forces_reshaped"].min())
        # print(torch.isnan(input_dict["forces_reshaped"]).sum())
        tool_embeds = tool_embed(input_dict["tool_idx"].long()).unsqueeze(1).repeat_interleave(input_dict['sequence_size'], dim = 1)

        tool_embeds_unshaped = torch.reshape(tool_embeds, (input_dict['batch_size']*input_dict['sequence_size'], tool_embeds.size(2)))

        frc_encs_unshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
        frc_encs_context_unshaped = frc_enc_context(torch.cat([frc_encs_unshaped, tool_embeds_unshaped]))
        frc_encs = torch.reshape(frc_encs_context_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_enc_size))

        # img_encs_unshaped = self.flatten(img_enc(input_dict["rgbd_reshaped"]))
        # img_encs = torch.reshape(img_encs_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.img_enc_size))
        # Step 2. full state sequence encoding
        states_t = torch.cat([input_dict["states"], frc_encs, input_dict["action"]], dim = 2).transpose(0,1)

        if "padding_mask" not in input_dict.keys():
            seq_encs = state_transdec(states_t).max(0)[0]
        else:
            seq_encs = state_transdec(states_t, padding_mask = input_dict["padding_mask_extended"]).max(0)[0]

        #### Calculating Options Logits
        # print(seq_encs.size(), tool_embeds.size(), policy_embeds.size(), input_dict["tool_idx"].size())
        states_logits = states_class(seq_encs)

        options_logits = torch.zeros((input_dict['batch_size'], 2))

        options_logits[:,0] += (states_logits * input_dict['tool_vector']).sum(1)

        options_logits[:,1] += (states_logits * (1 - input_dict['tool_vector'])).sum(1)

        return options_logits

    def getall_logits(self, input_dict):

        ol_list = []

        for i in range(self.num_ensembles):
            if "padding_mask" in input_dict.keys():
                input_dict["padding_mask_extended"] = self.get_input_dropout(input_dict["padding_mask"])

            ol = self.get_logits(input_dict, self.ensemble_list[i])

            ol_list.append(ol)

        return ol_list #, seq_encs ol_list #

    def get_input_dropout(self, padding_masks):
        # print("Padding mask\n", padding_masks[0])
        input_dropout = F.dropout(torch.ones(padding_masks.size()).float().to(self.device), p = (1 - (self.dropout_prob / 1))).bool()

        padding_masks_extended = torch.where(padding_masks == False, input_dropout, padding_masks)

        # print("New Padding mask\n", padding_masks_extended[0])

        return  padding_masks_extended

    def get_uncertainty_quant(self, input_dict):
        with torch.no_grad():
            ol_uncertainty_list = []
            T = 60

            for i in range(T):
                ol_list_sample = self.getall_logits(input_dict)

                for i in range(self.num_ensembles):
                    ol_list_sample[i] = ol_list_sample[i].unsqueeze(0)

                ol_uncertainty_list += ol_list_sample

            uncertainty_logits = torch.cat(ol_uncertainty_list, dim = 0)
            # print(uncertainty_logits.size())

            # for i in range(uncertainty_logits.size(0)):
            #     for j in range(uncertainty_logits.size(1)):
            #         print(F.softmax(uncertainty_logits[i,j], dim = 0))
            uncertainty_votes = uncertainty_logits.max(2)[1]

            uncertainty = torch.zeros((input_dict["batch_size"], self.num_options)).float().to(self.device)

            for i in range(self.num_options):
                i_votes = torch.where(uncertainty_votes == i, torch.ones_like(uncertainty_votes), torch.zeros_like(uncertainty_votes)).sum(0)
                uncertainty[:, i] = i_votes

            # print(uncertainty)
            uncertainty = uncertainty / uncertainty.sum(1).unsqueeze(1).repeat_interleave(self.num_options, dim = 1)

            return uncertainty

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        ol_list = self.getall_logits(input_dict)

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(ol_list[i]).unsqueeze(0))
            ol_list[i] = ol_list[i].unsqueeze(0)

        # uncertainty = self.get_uncertainty_quant(input_dict)

        # print(uncertainty.size())

        # uncertainty_list = [probs2inputs(uncertainty).unsqueeze(0)]
        # for i in range(self.num_tools):
        #     for j in range(self.num_states):
        #         tb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        #         sb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        # #         print("Uncertainty:\n", tb * sb * uncertainty)
        # print("Uncertainty: ", uncertainty)

        return {
            'options_class': torch.cat(ol_list, dim = 0),
            'options_inputs': torch.cat(inputs_list, dim = 0),
            # 'uncertainty_inputs': probs2inputs(uncertainty),
        }

    def probs(self, input_dict): 
        with torch.no_grad():
            self.eval()

            self.test_time_process_inputs(input_dict)
            self.process_inputs(input_dict)

            ol_list = self.getall_logits(input_dict)

            # for i in range(self.num_ensembles):
            #     print(F.softmax(ol_list[i], dim = 1))

            probs = F.softmax(random.choice(ol_list), dim = 1)

            return probs.max(1)[1]

    def process_inputs(self, input_dict):
        input_dict["force"] = (input_dict["force_hi_freq"].transpose(2,3)\
         - self.force_mean.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
         .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
         .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
         .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)) / \
        self.force_std.unsqueeze(0).unsqueeze(1).unsqueeze(3)\
         .repeat_interleave(input_dict["force_hi_freq"].size(0), dim = 0)\
         .repeat_interleave(input_dict["force_hi_freq"].size(1), dim = 1)\
         .repeat_interleave(input_dict["force_hi_freq"].size(2), dim = 3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3))) 

        input_dict["states"] = torch.cat([input_dict["rel_proprio_diff"], input_dict["contact_diff"]], dim = 2)
        
        input_dict["batch_size"] = input_dict["rel_proprio_diff"].size(0)
        input_dict["sequence_size"] = input_dict["rel_proprio_diff"].size(1)

         
    def test_time_process_inputs(self, input_dict):

        # print(input_dict.keys())

        input_dict["rel_proprio_diff"] = input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1]

        # print(input_dict['rel_proprio_diff'].size())
        input_dict["contact_diff"] = input_dict["contact"][:,1:] - input_dict["contact"][:, :-1]
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
        input_dict["action"] = input_dict["action"][:, :-1]
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1]

        if len(input_dict['tool_idx'].size()) == 0:
            input_dict['tool_idx'] = input_dict['tool_idx'].unsqueeze(0)

class Options_LikelihoodNet(Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_options = init_args['num_options']
        self.num_states = init_args['num_states']
        self.num_policies = init_args['num_policies']

        self.tool_dim = init_args['tool_dim']
        self.state_dim = self.tool_dim
        self.policy_dim = init_args['policy_dim']
        self.dropout_prob = 0.1 # init_args['dropout_prob']

        self.nl = 3

        self.tool_embed = Embedding(model_name + "_tool_embed",\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device)

        self.model_list.append(self.tool_embed)

        self.state_embed = Embedding(model_name + "_state_embed",\
                         self.num_states, self.state_dim, device= self.device).to(self.device)

        self.model_list.append(self.state_embed)

        self.policy_embed = Embedding(model_name + "_policy_embed",\
         self.num_policies, self.policy_dim, device= self.device).to(self.device)
        
        self.model_list.append(self.policy_embed)

        self.likelihood_net = ResNetFCN(model_name + "_likelihood_net", self.tool_dim + self.state_dim + self.policy_dim,\
         self.num_options, self.nl, dropout = True, dropout_prob = self.dropout_prob, uc = False, device = self.device).to(self.device)\

        self.model_list.append(self.likelihood_net)

    def forward(self, input_dict):
        tool_embed = self.state_embed(input_dict["tool_idx"].long())
        state_embed = self.state_embed(input_dict["state_idx"].long())
        pol_embed = self.policy_embed(input_dict["pol_idx"].long())

        likelihood_logits = self.likelihood_net(torch.cat([tool_embed, state_embed, policy_embed], dim =1))

        likelihood_inputs = logits2inputs(likelihood_logits)

        # print(likelihood_inputs.size())

        return {
            'likelihood_inputs': likelihood_inputs,
        }

    def logprobs(self, input_dict, obs_idx, with_margin):
        self.eval()

        tool_embed = self.state_embed(input_dict["tool_idx"].long())
        state_embed = self.state_embed(input_dict["state_idx"].long())
        pol_embed = self.policy_embed(input_dict["pol_idx"].long())

        likelihood_logits = self.likelihood_net(torch.cat([tool_embed, state_embed, policy_embed], dim =1))

        if with_margin:
            uninfo_constant = 0.2            
            likelihood_logprobs = F.log_softmax(torch.log(F.softmax(likelihood_logits, dim = 1) + uninfo_constant), dim = 1)
        else:
            likelihood_logprobs = F.log_softmax(likelihood_logits, dim=1)

        # print(likelihood_logprobs)
        # print(obs_idx)

        # print(likelihood_logprobs.size())

        likelihood_logprob = likelihood_logprobs[torch.arange(likelihood_logprobs.size(0)), obs_idx]

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

        return likelihood_logprob

