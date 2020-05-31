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

def declare_models(cfg, models_folder, device):
    info_flow = cfg['info_flow']
    model_dict = {}
    ###############################################
    ##### Declaring models to be trained ##########
    #################################################
    ##### Note if a path has been provided then the model will load a previous model

    dataset_path = cfg['dataset_path']

    with open(dataset_path + "datacollection_params.yml", 'r') as ymlfile:
        cfg1 = yaml.safe_load(ymlfile)

    tool_types = cfg1['peg_names']
    state_types = cfg1['hole_names']
    option_types = cfg1['fit_names']

    num_tools = len(tool_types)
    num_states = len(state_types)
    num_options = len(option_types)

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

        model_dict["Options_Sensor"] = Options_Sensor(models_folder, "Options_Sensor", info_flow, force_size, proprio_size, action_size, num_tools, num_options, dropout_prob, device = device).to(device)
    
    if "Options_ConfNet" in info_flow.keys():

        if info_flow["Options_ConfNet"]["model_folder"] is not "":
            with open(info_flow["Options_ConfNet"]["model_folder"] + "learning_params.yml", 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)
        else:
            cfg2 = cfg

        macro_action_size = cfg2['custom_params']['macro_action_size']
        dropout_prob =cfg2['custom_params']['dropout_prob']

        model_dict["Options_ConfNet"] = Options_ConfNet(models_folder, "Options_ConfNet", info_flow, macro_action_size, num_tools, num_states, num_options, dropout_prob, device = device).to(device)

    if "Options_ConfMat" in info_flow.keys():
        model_dict["Options_ConfMat"] = Options_ConfMat(models_folder, "Options_ConfMat", info_flow, num_tools, num_options, device = device).to(device)

    if "Distance_Predictor" in info_flow.keys():

        if info_flow["Distance_Predictor"]["model_folder"] is not "":
            with open(info_flow["Distance_Predictor"]["model_folder"] + "learning_params.yml", 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)
        else:
            cfg2 = cfg

        macro_action_size = cfg2['custom_params']['macro_action_size']

        model_dict["Distance_Predictor"] = Distance_Predictor(models_folder, "Distance_Predictor", info_flow, macro_action_size, num_tools, num_options, device = device).to(device)
    
    print("Finished Initialization")
    return model_dict

######################################
# Defining Custom Macromodels for project
#######################################

#### see Resnet in models_utils for example of how to set up a macromodel
class Options_Sensor(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, force_size,\
     proprio_size, action_size, num_tools, num_options, dropout_prob, device = None):
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

        self.num_ensembles = 3

        self.action_dim = action_size
        self.proprio_size = proprio_size
        self.force_size = force_size
        self.contact_size = 1
        self.frc_enc_size =  8 * 3 * 2
        self.dropout_prob = dropout_prob

        self.state_size = self.proprio_size + self.contact_size + self.frc_enc_size + self.action_dim

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                CONV1DN(save_folder + "_frc_enc" + str(i), load_folder + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = False, dropout = True, dropout_prob = dropout_prob,\
                   uc = True, device = self.device).to(self.device),\

                Transformer_Comparer(save_folder + "_state_transdec" + str(i), load_folder + "_state_transdec" + str(i),\
         self.state_size, self.num_tl, dropout_prob = dropout_prob, uc = True, device = self.device).to(self.device),\

                ResNetFCN(save_folder + "_options_class" + str(i), load_folder + "_options_class" + str(i),\
            self.state_size + self.num_tools, self.num_options, self.num_cl, dropout = True, dropout_prob = dropout_prob, \
            uc = True, device = self.device).to(self.device)\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_frc(self, force, model):
        return self.flatten(model(force))

    def get_enc(self, states, forces, actions, encoder_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, state_transdec = encoder_tuple

        frc_encs_unshaped = self.get_frc(forces, frc_enc)

        frc_encs = torch.reshape(frc_encs_unshaped, (batch_size, sequence_size, self.frc_enc_size))

        states_t = torch.cat([states, frc_encs, actions], dim = 2).transpose(0,1)

        if padding_masks is None:
            seq_encs = state_transdec(states_t).max(0)[0]
        else:
            seq_encs = state_transdec(states_t, padding_mask = padding_masks).max(0)[0]

        return seq_encs

    def get_data(self, states, forces, actions, tool_type, model_tuple, batch_size, sequence_size, padding_masks = None):
        frc_enc, transformer, options_class = model_tuple #origin_cov = model_tuple # 

        seq_encs = self.get_enc(states, forces, actions, (frc_enc, transformer), batch_size, sequence_size, padding_masks = padding_masks)

        # print("Sequence Encoding:", seq_encs)
        options_logits = options_class(torch.cat([seq_encs, tool_type], dim = 1))
       
        return options_logits

    def get_logits(self, proprio_diffs, contact_diffs, forces, actions, tool_type, padding_masks = None):

        assert self.num_tools == tool_type.size(1), "Incorrectly sized tool vector for model"
        
        batch_size = proprio_diffs.size(0)
        sequence_size = proprio_diffs.size(1)

        forces_reshaped = torch.reshape(forces, (forces.size(0) * forces.size(1), forces.size(2), forces.size(3)))#.contiguous()

        states = torch.cat([proprio_diffs, contact_diffs.unsqueeze(2)], dim = 2)

        options_logits = torch.zeros((batch_size, self.num_options)).float().to(self.device)
        seq_encs = torch.zeros((batch_size, self.state_size)).float().to(self.device)

        ol_list = []

        for i in range(self.num_ensembles):
            ol = self.get_data(states, forces_reshaped, actions, tool_type,\
             self.ensemble_list[i], batch_size, sequence_size, padding_masks)

            # options_logits += tool_ol * ol
            # # seq_encs += tool_se * se

            ol_list.append(ol)

        return ol_list #, seq_encs ol_list #

    def forward(self, input_dict):
        proprio_diffs = input_dict["proprio_diff"].to(self.device)
        actions = input_dict["action"].to(self.device)
        forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
        contact_diffs = input_dict["contact_diff"].to(self.device)
        tool_type = input_dict["tool_type"].to(self.device)
        padding_masks = input_dict["padding_mask"].to(self.device)

        # self.eval()

        ol_list = self.get_logits(proprio_diffs, contact_diffs, forces, actions, tool_type, padding_masks)

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(ol_list[i]).unsqueeze(0))
            ol_list[i] = ol_list[i].unsqueeze(0)

        with torch.no_grad():
            uncertainty_list = []
            T = 60

            for i in range(T):
                ol_list_sample = self.get_logits(proprio_diffs, contact_diffs, forces, actions, tool_type, padding_masks)

                for i in range(self.num_ensembles):
                    ol_list_sample[i] = ol_list_sample[i].unsqueeze(0)

                uncertainty_list += ol_list_sample

            uncertainty_logits = torch.cat(uncertainty_list, dim = 0)
            # print(uncertainty_logits.size())

            # for i in range(uncertainty_logits.size(0)):
            #     for j in range(uncertainty_logits.size(1)):
            #         print(F.softmax(uncertainty_logits[i,j], dim = 0))
            uncertainty_votes = uncertainty_logits.max(2)[1]

            uncertainty = torch.zeros((actions.size(0), self.num_options)).float().to(self.device)

            for i in range(self.num_options):
                i_votes = torch.where(uncertainty_votes == i, torch.ones_like(uncertainty_votes), torch.zeros_like(uncertainty_votes)).sum(0)
                uncertainty[:, i] = i_votes

            uncertainty = uncertainty / uncertainty.sum(1).unsqueeze(1).repeat_interleave(self.num_options, dim = 1)

            # print(uncertainty)

        return {
            'options_class': torch.cat(ol_list, dim = 0),
            'options_inputs': torch.cat(inputs_list, dim = 0),
            'uncertainty_inputs': probs2inputs(uncertainty),
        }

    def probs(self, input_dict): 
        with torch.no_grad():
            self.eval()
            proprios = input_dict["proprio"].to(self.device)
            forces = input_dict["force_hi_freq"].to(self.device).transpose(2,3)
            contacts = input_dict["contact"].to(self.device)
            actions = input_dict["action"].to(self.device)[:, :-1]
            tool_type = input_dict["tool_type"].to(self.device)        

            # macro_action = input_dict["macro_action"].to(self.device)

            proprio_diffs = proprios[:,1:] - proprios[:, :-1]
            contact_diffs = contacts[:,1:] - contacts[:, :-1]
            force_clipped = forces[:,1:]

            ol_list = self.get_logits(proprio_diffs, contact_diffs, force_clipped, actions, tool_type)

            return np.random.choice(ol_list).max(1)[1]

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
        self.z_dim = 32
        self.nl = 3
        self.num_ensembles = 1

        self.dropout_prob = dropout_prob

        # first number indicates index in peg vector, second number indicates number in ensemble
        # self.model_dict = {}
        # for i in range(self.num_tools):
        #     for j in range(self.num_states):
        #         self.model_dict[(i,j)] = (\
        #             ResNetFCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
        #             self.macro_action_size, self.z_dim, 1, device = self.device).to(self.device),\
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
                    self.z_dim + self.num_tools + self.num_states, self.num_options, self.nl, dropout = True, dropout_prob = self.dropout_prob,\
                     uc = False, device = self.device).to(self.device)\
                     )

        for key in self.model_dict.keys():
            for model in self.model_dict[key]:
                self.model_list.append(model)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def get_pred(self, macro_action, tool_type, state_type):
        conf_logits = torch.zeros_like(macro_action[:, :self.num_options])

        # assert self.num_tools == peg_type.size(1), "Incorrectly sized tool vector for model"
        assert self.num_states == state_type.size(1), "Incorrectly sized option vector for model"

        # for key in self.model_dict.keys():
        #     expand_state, conf_pred = self.model_dict[key]
        #     i, j = key #peg, option
        #     # peg_bool = peg_type[:,i].unsqueeze(1).repeat_interleave(self.num_options, dim=1)
        #     state_bool = state_type[:,j].unsqueeze(1).repeat_interleave(self.num_options, dim=1)

        #     conf_logits += state_bool * conf_pred(expand_state(macro_action)) #peg_bool *

        # return conf_logits

        cf_list = []

        for i in range(self.num_ensembles):
            expand_state, conf_pred = self.model_dict[i]

            cf_list.append(conf_pred(torch.cat([expand_state(macro_action), tool_type, state_type], dim =1)))

        return cf_list           

    def forward(self, input_dict):
        macro_action = input_dict["macro_action"].to(self.device)
        tool_type = input_dict["tool_type"].to(self.device)
        state_type = input_dict["state_type"].to(self.device)

        cf_list = self.get_pred(macro_action, tool_type, state_type)

        inputs_list =[]

        for i in range(self.num_ensembles):
            inputs_list.append(logits2inputs(cf_list[i]).unsqueeze(0))
            cf_list[i] = cf_list[i].unsqueeze(0)

        return {
            'confusion_inputs': torch.cat(inputs_list, dim=0),
        }

    def logits(self, peg_type, macro_action):
        with torch.no_grad():
            self.eval()
            batch_size = peg_type.size(0)
            logits_list = []

            for i in range(self.num_states):
                state_type = torch.zeros(self.num_states).float().to(self.device).unsqueeze(0).repeat_interleave(batch_size, dim = 0)
                state_type[:, i] = 1.0
                conf_logits = self.get_pred(macro_action, peg_type, state_type)

                # print(conf_logits.size())

                # print("Peg Type: ", peg_type)
                # print("Option_type: ", option_type)
                # print("Conf Probs: ", F.softmax(conf_logits, dim = 1))

                logits_list.append(conf_logits.unsqueeze(1))

            return torch.cat(logits_list, dim = 1)

    def conf_logprobs(self, peg_type, state_type, macro_action, obs_idx):
        with torch.no_grad():
            self.eval()
            conf_logits = self.get_pred(macro_action, peg_type, state_type)

            uninfo_constant = 0.2
                
            conf_logprobs = F.log_softmax(torch.log(F.softmax(conf_logits, dim = 1) + uninfo_constant), dim = 1)

            # print(conf_logprobs)
            # print(obs_idx)

            conf_logprob = conf_logprobs[torch.arange(conf_logprobs.size(0)), obs_idx]

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

# class Options_ConfMat(Proto_Macromodel):
#     def __init__(self, model_folder, model_name, info_flow, num_tools, num_options, device = None):
#         super().__init__()

#         if info_flow[model_name]["model_folder"] != "":
#             load_folder = info_flow[model_name]["model_folder"] + model_name
#         else:
#             load_folder = model_folder + model_name

#         save_folder = model_folder + model_name 

#         self.device = device
#         self.model_list = []

#         self.num_tools = num_tools
#         self.num_options = num_options

#         self.conf_mat = Params(save_folder + "_conf_mat", load_folder + "_conf_mat",\
#          (self.num_tools, self.num_options, self.num_options), device = self.device)\
                 
#         self.conf_mat().data[:] = 0
#         self.epoch_num = 0

#         self.model_list.append(self.conf_mat)

#         if info_flow[model_name]["model_folder"] != "":
#             self.load(info_flow[model_name]["epoch_num"])

#     def forward(self, input_dict):
#         peg_type = input_dict["peg_type"].to(self.device)
#         option_type = input_dict["option_type"].to(self.device)
#         option_est = input_dict["options_est"].to(self.device)
#         epoch = input_dict["epoch"].to(self.device)

#         assert self.num_tools == peg_type.size(1), "Incorrectly sized tool vector for model"
#         assert self.num_options == option_type.size(1), "Incorrectl sized option vector for model"

#         rows = peg_type.max(1)[1]
#         cols = option_type.max(1)[1]
#         widths = option_est.clone()
        
#         for i in range(rows.size(0)):
#             self.conf_mat().data[rows[i], cols[i], widths[i]] += 1

#         if epoch != self.epoch_num and epoch % 10 == 0:
#             self.epoch_num = epoch
#             for i in range(self.num_tools):
#                 print("Confusion Matrix " + str(i) + ":\n", self.conf_mat().data[i])

#     def logits(self, peg_type, macro_action):
#         self.eval()
#         logits_list = []

#         assert self.num_tools == peg_type.size(1), "Incorrectly sized tool vector for model"

#         for j in range(peg_type.size(0)):
#             row = peg_type[j].max(0)[1]
#             logits_list.append(self.conf_mat().data[row].unsqueeze(0))

#         return torch.log(torch.cat(logits_list, dim = 0))

#     def conf_logprobs(self, peg_type, option_type, macro_action, obs_idx):
#         self.eval()

#         assert self.num_tools == peg_type.size(1), "Incorrectly sized tool vector for model"
#         assert self.num_options == option_type.size(1), "Incorrectl sized option vector for model"

#         logprobs = torch.zeros_like(peg_type)

#         uninfo_constant = 0.2

#         for i in range(peg_type.size(0)):
#             row = peg_type[i].max(0)[1]
#             col = option_type[i].max(0)[1]

#             logprobs[i] = F.log_softmax(torch.log((self.conf_mat().data[row, col] / torch.sum(self.conf_mat().data[row, col])) + uninfo_constant), dim = 0)

#         logprob = logprobs[torch.arange(logprobs.size(0)), obs_idx]

#         # print(obs_idx)
#         # print(torch.exp(logprob))

#         return logprob 

# class Distance_Predictor(Proto_Macromodel):
#     def __init__(self, model_folder, model_name, info_flow, macro_action_size, num_tools, num_options, device = None):
#         super().__init__()

#         if info_flow[model_name]["model_folder"] != "":
#             load_folder = info_flow[model_name]["model_folder"] + model_name
#         else:
#             load_folder = model_folder + model_name

#         save_folder = model_folder + model_name 

#         self.device = device
#         self.model_list = []
#         self.ensemble_list = []

#         self.num_tools = num_tools
#         self.num_options = num_options
#         self.tool_idx = tuple(itertools.combinations_with_replacement(range(self.num_tools), 2))
#         self.option_idx = tuple(itertools.combinations_with_replacement(range(self.num_options), 2))
#         self.num_t = len(self.tool_idx)
#         self.num_o = len(self.option_idx)

#         # print(self.tool_idx)
#         # print(self.option_idx)
#         self.macro_action_size = macro_action_size
#         self.z_dim = 32
#         self.nl = 4

#         # first number indicates index in peg vector, second number indicates number in ensemble
#         self.model_dict = {}
#         for i in range(self.num_t):
#             for j in range(self.num_o):
#                 self.model_dict[(i,j)] = (\
#                     FCN(save_folder + "_expand_state" + str(i) + str(j), load_folder + "_expand_state" + str(i) + str(j),\
#                     2 * self.macro_action_size, self.z_dim, 1, device = self.device).to(self.device),\
#                      ResNetFCN(save_folder + "_conf_pred" + str(i) + str(j), load_folder + "_conf_pred" + str(i) + str(j),\
#                     self.z_dim, 1, self.nl, device = self.device).to(self.device)\
#                      )

#         for key in self.model_dict.keys():
#             for model in self.model_dict[key]:
#                 self.model_list.append(model)

#         if info_flow[model_name]["model_folder"] != "":
#             self.load(info_flow[model_name]["epoch_num"])

#     def get_pred(self, macro_action0, macro_action1, tool0, tool1, option0, option1):
#         distances = torch.zeros(macro_action0.size(0)).float().to(self.device).unsqueeze(1)

#         assert self.num_tools == tool1.size(1), "Incorrectly sized tool vector for model"
#         assert self.num_options == option1.size(1), "Incorrectly sized option vector for model"

#         for key in self.model_dict.keys():
#             expand_state, distance_pred = self.model_dict[key]
#             i, j = key #peg_idx_idx, option_idx_idx
#             t0, t1 = self.tool_idx[i]
#             o0, o1 = self.option_idx[j]

#             if t0 == t1:
#                 tb0 = tool0[:,t0].unsqueeze(1)
#                 tb1 = tool1[:,t1].unsqueeze(1)
#             else:
#                 tb0 = tool0[:,t0].unsqueeze(1) + tool0[:,t1].unsqueeze(1)
#                 tb1 = tool1[:,t1].unsqueeze(1) + tool1[:,t0].unsqueeze(1)

#             if o0 == o1:
#                 ob0 = option0[:,o0].unsqueeze(1)
#                 ob1 = option1[:,o1].unsqueeze(1)
#             else:
#                 ob0 = option0[:,o0].unsqueeze(1) + option0[:,o1].unsqueeze(1)
#                 ob1 = option1[:,o1].unsqueeze(1) + option1[:,o0].unsqueeze(1)

#             distances += tb0 * tb1 * ob0 * ob1 * torch.abs(distance_pred(expand_state(torch.cat([macro_action0, macro_action1], dim =1))))

#         return distances.squeeze()

#     def forward(self, input_dict):
#         macro_action0 = input_dict["macro_action"].to(self.device)
#         tool_type0 = input_dict["tool_type"].to(self.device)
#         option_type0 = input_dict["option_type"].to(self.device)
#         permutation = input_dict["permutation"]

#         macro_action1 = macro_action0[permutation]
#         tool_type1 = tool_type0[permutation]
#         option_type1 = option_type0[permutation]

#         distances = self.get_pred(macro_action0, macro_action1, tool_type0, tool_type1, option_type0, option_type1)

#         return {
#             'distances': distances,
#         }

#     def distances(self, tool_type, macro_action):
#         self.eval()
#         batch_size = tool_type.size(0)
#         distances = torch.zeros(batch_size).float().to(device)

#         for i in range(self.num_options - 1):
#             for j in range(i + 1, self.num_options):
#                 if i == j:
#                     continue

#                 option_i = torch.zeros((batch_size, self.num_options)).float().to(self.device)
#                 option_j = torch.zeros((batch_size, self.num_options)).float().to(self.device)

#                 option_i[:, i] = 1.0
#                 option_j[:,j] = 1.0

#                 distances += self.get_pred(macro_action, macro_action, tool_type, tool_type, option_i, option_j)

#         return distances
