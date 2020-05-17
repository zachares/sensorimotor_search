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

def declare_models(cfg, models_folder, device):
    info_flow = cfg['info_flow']
    model_dict = {}
    ###############################################
    ##### Declaring models to be trained ##########
    #################################################
    ##### Note if a path has been provided then the model will load a previous model

    dataset_path = cfg['dataloading_params']['dataset_path']

    with open(dataset_path + "datacollection_params.yml", 'r') as ymlfile:
        cfg1 = yaml.safe_load(ymlfile)

    tool_types = cfg1['peg_names']
    option_types = cfg1['fit_names']

    num_tools = len(tool_types)
    num_options = len(option_types)

    if "Options_Sensor" in info_flow.keys():
        force_size =cfg['custom_params']['force_size'] 
        proprio_size = cfg['custom_params']['proprio_size'] 
        action_size =cfg['custom_params']['action_size']
        model_dict["Options_Sensor"] = Options_Sensor(models_folder, "Options_Sensor", info_flow, force_size, proprio_size, action_size, num_tools, num_options, device = device).to(device)
    
    if "Options_ConfNet" in info_flow.keys():
        macro_action_size = cfg['custom_params']['macro_action_size']
        model_dict["Options_ConfNet"] = Options_ConfNet(models_folder, "Options_ConfNet", info_flow, macro_action_size, num_tools, num_options, device = device).to(device)

    if "Options_ConfMat" in info_flow.keys():
        model_dict["Options_ConfMat"] = Options_ConfMat(models_folder, "Options_ConfMat", self.info_flow, num_tools, num_options, device = device).to(device)

    print("Finished Initialization")
    return model_dict

######################################
# Defining Custom Macromodels for project
#######################################

#### see Resnet in models_utils for example of how to set up a macromodel
class Options_Sensor(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size, proprio_size, action_size, num_tools, num_options, device = None):
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

        self.action_dim = action_size
        self.proprio_size = proprio_size
        self.force_size = force_size
        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim

        self.num_tl = 4
        self.num_cl = 3

        for i in range(self.num_tools):
            self.ensemble_list.append((\
                CONV1DN(save_folder + "_frc_enc" + str(i), load_folder + "_frc_enc" + str(i), (self.force_size[0], self.force_size[1]),\
             (self.frc_enc_size, 1), False, True, 1, device = self.device).to(self.device),\

         #        CONV2DN(save_folder + "_fft_enc" + str(i), load_folder + "_fft_enc" + str(i),\
         # (self.force_size[0], 27, 2), (8, 3, 2), False, True, 1, device = self.device).to(self.device),\

                Transformer_Decoder(save_folder + "_state_transdec" + str(i), load_folder + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, device = self.device).to(self.device),\

                ResNetFCN(save_folder + "_options_class" + str(i), load_folder + "_options_class" + str(i),\
            self.state_size, self.num_options, self.num_cl, device = self.device).to(self.device)\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        # fft = torch.rfft(force, 2, normalized=False, onesided=True)
        # frc_enc = model(torch.cat([fft, torch.zeros_like(fft[:,:,0,:]).unsqueeze(2)], dim = 2))
        return model(force)

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

    def get_data(self, states, forces, actions, model_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, transformer, options_class = model_tuple #origin_cov = model_tuple # 

        seq_encs = self.get_enc(states, forces, actions, (frc_enc, transformer), batch_size, sequence_size, padding_masks = padding_masks)

        # print("Sequence Encoding:", seq_encs)
        options_logits = options_class(seq_encs)
       
        return options_logits

    def get_logits(self, proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks = None):
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

        states = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2)], dim = 2)

        options_logits = torch.zeros((batch_size, self.num_options)).float().to(self.device)

        for i in range(self.num_tools):
            peg_ol = peg_type[:,i].unsqueeze(1).repeat_interleave(options_logits.size(1), dim=1)

            options_logits += peg_ol * self.get_data(states, forces_reshaped, actions,\
             self.ensemble_list[i], batch_size, sequence_size, padding_masks)

        return options_logits

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)

        options_logits = self.get_logits(proprio_diffs, contact_diffs, forces, actions, peg_type, padding_masks)

        probs = F.softmax(options_logits, dim = 1)

        return {
            'options_class': options_logits,
            'option_est': probs.max(1)[1],
        }

    def probs(self, input_dict):
        self.eval()
        proprios = input_dict["proprio"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contacts = input_dict["contact"].to(self.device)
        actions = input_dict["action"].to(self.device)[:, :-1]
        peg_type = input_dict["peg_type"].to(self.device)        

        # macro_action = input_dict["macro_action"].to(self.device)

        proprio_diffs = proprios[:,1:] - proprios[:, :-1]
        contact_diffs = contacts[:,1:] - contacts[:, :-1]
        force_clipped = forces[:,1:]

        options_logits = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, peg_type)

        # print("Observation Probs: ", F.softmax(options_logits, dim = 1))

        return F.softmax(options_logits, dim = 1)

class Options_ConfNet(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, macro_action_size, num_tools, num_options, device = None):
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

        self.macro_action_size = macro_action_size
        self.z_dim = 32
        self.nl = 2

        # first number indicates index in peg vector, second number indicates number in ensemble
        self.model_dict = {}
        for i in range(self.num_tools):
            for j in range(self.num_options):
                self.model_dict[(i,j)] = (\
                    FCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
                    self.macro_action_size, self.z_dim, 1, device = self.device).to(self.device),\
                     ResNetFCN(save_folder + "_conf_pred" + str(i) + str(j), load_folder + "_conf_pred" + str(i) + str(j),\
                    self.z_dim, self.num_options, self.nl, device = self.device).to(self.device)\
                     )

        for key in self.model_dict.keys():
            for model in self.model_dict[key]:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, macro_action, peg_type, option_type):
        conf_logits = torch.zeros_like(macro_action[:, :self.num_options])

        for key in self.model_dict.keys():
            expand_state, conf_pred = self.model_dict[key]
            i, j = key #peg, option
            peg_bool = peg_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim=1)
            option_bool = option_type[:,j].unsqueeze(1).repeat_interleave(self.num_options, dim=1)

            conf_logits += peg_bool * option_bool * conf_pred(expand_state(macro_action))

        return conf_logits

    def forward(self, input_dict):
        macro_action = input_dict["macro_action"].to(self.device)
        peg_type = input_dict["peg_type"].to(self.device)
        option_type = input_dict["option_type"].to(self.device)

        conf_logits = self.get_pred(macro_action, peg_type, option_type)

        return {
            'conf_class': conf_logits,
        }

    def logits(self, peg_type, macro_action):
        batch_size = peg_type.size(0)
        self.eval()
        logits_list = []

        for i in range(self.num_options):
            option_type = torch.zeros(self.num_options).float().to(self.device).unsqueeze(0).repeat_interleave(batch_size, dim = 0)
            option_type[:, i] = 1.0
            conf_logits = self.get_pred(macro_action, peg_type, option_type)

            print(conf_logits.size())

            print("Peg Type: ", peg_type)
            print("Option_type: ", Option_type)
            print("Conf Probs: ", F.softmax(conf_logits, dim = 0))

            logits_list.append(conf_logits.unsqueeze(1))

        return torch.cat(logits_list, dim = 1)

    def conf_logprobs(self, peg_type, option_type, macro_action, obs_idx):
        self.eval()
        conf_logits = self.get_pred(macro_action, peg_type, option_type)

        uninfo_constant = 0.2
            
        conf_logprobs = F.log_softmax(torch.log(F.softmax(conf_logits, dim = 1) + uninfo_constant), dim = 1)

        conf_logprob = conf_logprobs[torch.arange(conf_logprobs.size(0)), obs_idx]

        # print("Peg Type: ", peg_type)
        # print("Option_Type: ", option_type)
        # print("Macro action: ", macro_action)
        # print("Obs Idx: ", obs_idx)
        # print("Conf probs: ", F.softmax(conf_logits, dim = 1))
        # print("Conf probs less: ", torch.exp(conf_logprobs))
        # print("Conf logprobs: ", conf_logprobs)
        # print("Conf logprob: ", conf_logprob)

        return conf_logprob

class Options_ConfMat(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, num_tools, num_options, device = None):
        super().__init__()

        if info_flow[model_name]["model_folder"] != "":
            load_folder = info_flow[model_name]["model_folder"] + model_name
        else:
            load_folder = model_folder + model_name

        save_folder = model_folder + model_name 

        self.device = device
        self.model_list = []

        self.num_tools = num_tools
        self.num_options = num_options

        self.conf_mat = Params(save_folder + "_conf_mat", load_folder + "_conf_mat",\
         (self.num_tools, self.num_options, self.num_options), device = self.device)\
                 
        self.conf_mat().data[:] = 0
        self.epoch_num = 0

        self.model_list.append(self.conf_mat)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def forward(self, input_dict):
        peg_type = input_dict["peg_type"].to(self.device)
        option_type = input_dict["option_type"].to(self.device)
        option_est = input_dict["option_est"].to(self.device)
        epoch = input_dict["epoch"].to(self.device)

        rows = peg_type.max(1)[1]
        cols = hole_type.max(1)[1]
        widths = option_est.clone()
        
        for i in range(rows.size(0)):
            self.conf_mat().data[rows[i], cols[i], widths[i]] += 1

        if epoch != self.epoch_num:
            self.epoch_num = epoch
            for i in range(self.num_tools):
                print("Confusion Matrix " + str(i) + ":\n", self.conf_mat().data[i])

    def logits(self, peg_type, macro_action):
        self.eval()
        logits_list = []

        for j in range(peg_type.size(0)):
            row = peg_type[j].max(0)[1]
            logits_list.append(self.conf_mat().data[row].unsqueeze(0))

        return torch.log(torch.cat(logits_list, dim = 0))

    def conf_logprobs(self, peg_type, hole_type, macro_action, obs_idx):
        self.eval()
        logprobs = torch.zeros_like(peg_type)

        uninfo_constant = 0.2

        for i in range(peg_type.size(0)):
            row = peg_type[i].max(0)[1]
            col = hole_type[i].max(0)[1]

            logprobs[i] = F.log_softmax(torch.log((self.conf_mat().data[row, col] / torch.sum(self.conf_mat().data[row, col])) + uninfo_constant), dim = 0)

        logprob = logprobs[torch.arange(logprobs.size(0)), obs_idx]

        # print(obs_idx)
        # print(torch.exp(logprob))

        return logprob 
