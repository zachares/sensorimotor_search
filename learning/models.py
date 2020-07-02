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

import sys

sys.path.insert(0, "../") 

from project_utils import *

def declare_models(cfg, models_folder, device):
    info_flow = cfg['info_flow']
    model_dict = {}
    ###############################################
    ##### Declaring models to be trained ##########
    #################################################
    ##### Note if a path has been provided then the model will load a previous model

    ### NEEDS TO BE FIXED ####
    num_tools = 3
    num_states = 3
    num_options = 2


    if "Options_Sensor" in info_flow.keys():
        if info_flow["Options_Sensor"]["model_folder"] is not "":
            with open(info_flow["Options_Sensor"]["model_folder"] + "learning_params.yml", 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)
        else:
            cfg2 = cfg

        force_size =cfg2['custom_params']['force_size'] 
        proprio_size = cfg2['custom_params']['proprio_size'] 
        action_size =cfg2['custom_params']['action_size']
        dropout_prob =cfg2['custom_params']['dropout_prob']
        image_size = cfg2['custom_params']['image_size']
        num_policies = cfg2['num_policies']

        model_dict["Options_Sensor"] = Options_Sensor(models_folder, "Options_Sensor", info_flow, image_size,\
         force_size, proprio_size, action_size, num_tools, num_options, num_policies, dropout_prob, device = device).to(device)
    
    if "Options_ConfNet" in info_flow.keys():

        if info_flow["Options_ConfNet"]["model_folder"] is not "":
            with open(info_flow["Options_ConfNet"]["model_folder"] + "learning_params.yml", 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)
        else:
            cfg2 = cfg

        macro_action_size = cfg2['custom_params']['macro_action_size']
        dropout_prob =cfg2['custom_params']['dropout_prob']

        model_dict["Options_ConfNet"] = Options_ConfNet(models_folder, "Options_ConfNet", info_flow, macro_action_size, num_tools, num_states, num_options, dropout_prob, device = device).to(device)

    print("Finished Initialization")
    return model_dict

######################################
# Defining Custom Macromodels for project
#######################################

#### see Resnet in models_utils for example of how to set up a macromodel
class Options_Sensor(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, image_size, force_size,\
     proprio_size, action_size, num_tools, num_options, num_policies, dropout_prob, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = num_tools
        self.num_states = 3
        self.num_options = num_options

        self.num_ensembles = 3

        self.action_dim = action_size
        self.proprio_size = proprio_size
        self.image_size = image_size
        self.force_size = force_size
        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.img_enc_size = 128
        self.dropout_prob = dropout_prob

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim # + self.img_enc_size

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = True

        self.policy_dim = 32
        self.tool_dim = 6
        self.num_policy = num_policies

        self.shape_embed = Embedding(save_folder + "_shape_embed", load_folder + "_shape_embed",\
         self.num_tools, self.tool_dim, device= self.device).to(self.device)

        self.model_list.append(self.shape_embed)

        self.policy_embed = Embedding(save_folder + "_policy_embed", load_folder + "_policy_embed",\
         self.num_policy, self.policy_dim, device= self.device).to(self.device)
        
        self.model_list.append(self.policy_embed)

        self.vision_list = []

        for i in range(self.num_states):
            self.vision_list.append((\
                CONV2DN(save_folder + "_pos_enc" + str(i), load_folder + "_pos_enc" + str(i),\
                 (self.image_size[0], self.image_size[1], self.image_size[2]), (self.img_enc_size, 1, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = dropout_prob,\
                   uc = False, device = self.device).to(self.device),\
            
                ResNetFCN(save_folder + "_pos_est" + str(i), load_folder + "_pos_est" + str(i),\
                2 * self.img_enc_size, 2, self.num_cl, dropout = True, dropout_prob = dropout_prob, \
                uc = False, device = self.device).to(self.device)))

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                CONV1DN(save_folder + "_frc_enc" + str(i), load_folder + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                Transformer_Comparer(save_folder + "_state_transdec" + str(i), load_folder + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, dropout_prob = dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                ResNetFCN(save_folder + "_options_class" + str(i), load_folder + "_options_class" + str(i),\
            self.state_size + self.tool_dim + self.policy_dim, self.num_options, self.num_cl, dropout = True, dropout_prob = dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

        for model_tuple in self.vision_list:
            for model in model_tuple:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_logits(self, input_dict, model_tuple):
        frc_enc, state_transdec, options_class = model_tuple #origin_cov = model_tuple # 

        ##### Calculating Series Encoding
        # Step 1. frc encoding
        # print(input_dict["forces_reshaped"].size())
        # print(input_dict["forces_reshaped"].max())
        # print(input_dict["forces_reshaped"].min())
        # print(torch.isnan(input_dict["forces_reshaped"]).sum())
        frc_encs_unshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))
        frc_encs = torch.reshape(frc_encs_unshaped, (input_dict["batch_size"], input_dict["sequence_size"], self.frc_enc_size))

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
        full_enc = torch.cat([seq_encs, input_dict["tool_embed"], input_dict["pol_embed"]], dim = 1)

        options_logits = options_class(full_enc)

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

    def est_poses(self, input_dict):
        pos_est = []

        for i in range(len(self.vision_list)):
            img_enc, pos_estimator = self.vision_list[i]

            pos_est.append((pos_estimator(torch.cat([self.flatten(img_enc(input_dict['rgbd_first'])),\
             self.flatten(img_enc(input_dict['rgbd_last']))], dim = 1))).unsqueeze(1))

        return torch.cat(pos_est, dim = 1)
    def forward(self, input_dict):
        self.process_inputs(input_dict)

        pos_est = self.est_poses(input_dict) / 10

        ol_list = self.getall_logits(input_dict)

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(ol_list[i]).unsqueeze(0))
            ol_list[i] = ol_list[i].unsqueeze(0)

        # uncertainty = self.get_uncertainty_quant(input_dict)

        # uncertainty_list = [probs2inputs(uncertainty).unsqueeze(0)]
        # for i in range(self.num_tools):
        #     for j in range(self.num_states):
        #         tb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        #         sb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        # #         print("Uncertainty:\n", tb * sb * uncertainty)
        # print("Uncertainty: ", uncertainty)

        return {
            'options_class': torch.cat(ol_list, dim = 0),
            'pos_est': pos_est,
            'options_inputs': torch.cat(inputs_list, dim = 0),
            # 'uncertainty_inputs': probs2inputs(uncertainty),
            'state_embed': input_dict['state_embed'],
            'tool_embed': input_dict['tool_embed'],
            'policy_embed': input_dict['pol_embed'],
        }

    def process_inputs(self, input_dict):
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["states"] = torch.cat([input_dict["rel_proprio_diff"], input_dict["contact_diff"]], dim = 2)
        
        input_dict["batch_size"] = input_dict["rel_proprio_diff"].size(0)
        input_dict["sequence_size"] = input_dict["rel_proprio_diff"].size(1)

        if 'state_idx' in input_dict.keys():
            input_dict['state_embed'] = self.shape_embed(input_dict["state_idx"].long())

        input_dict['tool_embed'] = self.shape_embed(input_dict["tool_idx"].long()) 
        input_dict['pol_embed'] = self.policy_embed(input_dict["pol_idx"].long())

    def img_pos_estimate(self, rgbd):
        return torch.reshape(self.img_pos_est(torch.cat([\
            self.flatten(self.img_enc(rgbd.unsqueeze(0))),\
             self.flatten(self.img_enc(rgbd.unsqueeze(0)))], dim = 1)), (1, 3, 3)).squeeze()

    def embeds(self, input_dict):
        with torch.no_grad():
            self.eval()
            batch_size = input_dict["macro_action"].size(0)

            input_dict["state_embed"] = self.shape_embed(input_dict["state_idx"].long()).repeat_interleave(batch_size, dim = 0)
            input_dict["tool_embed"] = self.shape_embed(input_dict["tool_idx"].long()).repeat_interleave(batch_size, dim = 0) 
            input_dict["policy_embed"] = self.policy_embed(input_dict["pol_idx"].long())

            # print(input_dict["state_embed"].size())
            # print(input_dict["tool_embed"].size())
            # print(input_dict["policy_embed"].size())

    def pos_est(self, input_dict): 
        with torch.no_grad():
            self.eval()
            input_dict["rel_proprio_diff"] = input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1]
            input_dict["contact_diff"] = input_dict["contact"][:,1:] - input_dict["contact"][:, :-1]
            input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
            input_dict["action"] = input_dict["action"][:, :-1]
            input_dict["rgbd_first"] = input_dict["rgbd"][:, 0]
            input_dict["rgbd_last"] = input_dict["rgbd"][:,-1]

            self.process_inputs(input_dict)

            uncertainty, pe_mean = self.get_uncertainty_quant(input_dict)

            print(pe_mean)

            return pe_mean[input_dict['cand_idx'].long()]

    def probs(self, input_dict): 
        with torch.no_grad():
            self.eval()
            input_dict["rel_proprio_diff"] = input_dict["rel_proprio"][:,1:] - input_dict["rel_proprio"][:, :-1]
            input_dict["contact_diff"] = input_dict["contact"][:,1:] - input_dict["contact"][:, :-1]
            input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:]
            input_dict["action"] = input_dict["action"][:, :-1]
            input_dict["rgbd_first"] = input_dict["rgbd"][:, 0]
            input_dict["rgbd_last"] = input_dict["rgbd"][:,-1]

            self.process_inputs(input_dict)

            ol_list, pe_list = self.getall_logits(input_dict)

            # for i in range(self.num_ensembles):
            #     print(F.softmax(ol_list[i], dim = 1))

            probs = F.softmax(random.choice(ol_list), dim = 1)

            return probs.max(1)[1]

class Options_ConfNet(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, macro_action_size, num_tools, num_states, num_options, dropout_prob, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = num_tools
        self.num_options = num_options
        self.num_states = num_states

        self.macro_action_size = macro_action_size
        self.z_dim = 16
        self.tool_dim = 6
        self.state_dim = 6
        self.policy_dim = 32
        self.nl = 4
        self.num_ensembles = 1

        self.dropout_prob = 0.1 # dropout_prob

        # first number indicates index in peg vector, second number indicates number in ensemble
        # self.model_dict = {}
        # for i in range(self.num_tools):
        #     for j in range(self.num_states):
        #         self.model_dict[(i,j)] = (\
        #             ResNetFCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
        #             self.macro_action_size, self.z_dim, 1, dropout = True, dropout_prob = dropout_prob,\
        #              uc = False, device = self.device).to(self.device),\
        #             # FCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
        #             # self.macro_action_size, self.z_dim, 1, device = self.device).to(self.device),\
        #              ResNetFCN(save_folder + "_conf_pred" + str(i) + str(j), load_folder + "_conf_pred" + str(i) + str(j),\
        #             self.z_dim, self.num_options, self.nl, dropout = True, dropout_prob = dropout_prob,\
        #              uc = False, device = self.device).to(self.device)\
        #              )

        self.model_dict = {}
        for i in range(self.num_ensembles):
            # for j in range(self.num_states):
                self.model_dict[i] = (\
                    ResNetFCN(save_folder + "_expand_state" + str(i), load_folder + "_expand_state" + str(i),\
                    self.macro_action_size, self.z_dim, self.nl, dropout = True, dropout_prob = self.dropout_prob,\
                     uc = False, device = self.device).to(self.device),\
                     ResNetFCN(save_folder + "_conf_pred" + str(i), load_folder + "_conf_pred" + str(i),\
                    self.z_dim + self.tool_dim + self.state_dim + self.policy_dim, self.num_options, self.nl, dropout = True, dropout_prob = self.dropout_prob,\
                     uc = False, device = self.device).to(self.device)\
                     )

        for key in self.model_dict.keys():
            for model in self.model_dict[key]:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, input_dict):
        # conf_logits = torch.zeros_like(macro_action[:, :self.num_options])
        # assert self.num_tools == peg_type.size(1), "Incorrectly sized tool vector for model"
        # assert self.num_states == input_dict["state_type"].size(1), "Incorrectly sized option vector for model"

        cf_list = []

        # print("Start")
        # print("Start pos\n", macro_action[:,:3])
        # print("Velo\n", macro_action[:, 3:6])
        # print("End pos\n",macro_action[:,-4:-1])
        # print("Num steps\n", macro_action[:, -1])
        # for key in self.model_dict.keys():
        #     expand_state, conf_pred = self.model_dict[key]
        #     i, j = key #peg, option
        #     tool_bool = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim=1)
        #     state_bool = state_type[:,j].unsqueeze(1).repeat_interleave(self.num_options, dim=1)

        #     # print(torch.where((tool_bool * state_bool) != 0, F.softmax(conf_pred(expand_state(macro_action)), dim = 1), torch.zeros_like(tool_bool)))

        #     conf_logits += tool_bool * state_bool * conf_pred(expand_state(macro_action)) #peg_bool *

        #     # print(F.softmax(conf_logits))

        # # print("End")
        # cf_list.append(conf_logits)        

        for i in range(self.num_ensembles):
            expand_state, conf_pred = self.model_dict[i]

            cf_list.append(conf_pred(torch.cat([expand_state(input_dict["macro_action_t"]),\
             input_dict["tool_embed"], input_dict["state_embed"], input_dict["policy_embed"]], dim =1)))

        return cf_list

    def transform_action(self, p0):
        d0 = p0.norm(p=2, dim =1)
        angle0 = T_angle(torch.atan2(p0[:,1], p0[:,0]))

        return torch.cat([p0, d0.unsqueeze(1), angle0.unsqueeze(1)], dim =1)

    def forward(self, input_dict):
        input_dict["macro_action_t"] = self.transform_action(input_dict["macro_action"])
        cf_list = self.get_pred(input_dict)

        # print(macro_action[:,:3])
        conf_logits = cf_list[0]
        # for i in range(self.num_tools):
        #     for j in range(self.num_states):
        #         tb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        #         sb = tool_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim = 1)
        #         print("Conf Probs: ", tb*sb*F.softmax(conf_logits, dim = 1))

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(cf_list[i]).unsqueeze(0))
            cf_list[i] = cf_list[i].unsqueeze(0)

        return {
            'confusion_inputs': torch.cat(inputs_list, dim=0),
        }

    def logprobs(self, input_dict, obs_idx, with_margin):
        self.eval()

        input_dict["macro_action_t"] = self.transform_action(input_dict["macro_action"])

        conf_logits = self.get_pred(input_dict)[0]

        if with_margin:
            uninfo_constant = 0.2            
            conf_logprobs = F.log_softmax(torch.log(F.softmax(conf_logits, dim = 1) + uninfo_constant), dim = 1)
        else:
            conf_logprobs = F.log_softmax(conf_logits, dim=1)

        # print(conf_logprobs)
        # print(obs_idx)

        # print(conf_logprobs.size())

        conf_logprob = conf_logprobs[torch.arange(conf_logprobs.size(0)), obs_idx]

        # print(conf_logprob.size())

        # print(conf_logprob)

        # print("Peg Type: ", peg_type)
        # print("Option_Type: ", option_type)
        # print("Macro action: ", macro_action)
        # print("Obs Idx: ", obs_idx)
        # print("Conf probs: ", F.softmax(conf_logits, dim = 1))
        # print("Conf probs less: ", torch.exp(conf_logprobs))
        # print("Conf logprobs: ", conf_logprobs)
        # print("Conf logprob: ", conf_logprob)

        return conf_logprob
