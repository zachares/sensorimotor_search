import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

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

def cholesky_dec(noise_matrix): #converting noise parameters to cholesky decomposition form
    chol_dec = noise_matrix.tril()
    chol_diag = torch.abs(chol_dec.diag())
    chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] *= 0
    chol_dec[torch.arange(chol_dec.size(0)), torch.arange(chol_dec.size(1))] += chol_diag     
    return chol_dec

def sample_multiv_gauss(mean, num_samples, chol_dec = None, var = None):
    # mean dim - batch * state
    # var dim - batch * state * state
    # chol dec dim - batch * state * state
    epsilon = Normal(0, 1).sample((mean.size(0), mean.size(1), num_samples))
    if chol_dec is None:
        chol_dec = torch.cholesky(var)
    return mean.unsqueeze(1).repeat_interleave(num_samples, 1) + torch.bmm(chol_dec.unsqueeze(1), epsilon).transpose(1,2)


def multiv_gauss_params(samples): # samples are in sequence along dimension 1
    # (batch x samples x state) dimensions
    sample_mean = samples.mean(1)
    samples_zeromean = samples - sample_mean.unsqueeze(1).repeat_interleave(samples.size(1), 1)
    sample_cov = torch.bmm(samples_zeromean.transpose(1,2), samples_zeromean) / self.num_particles # an unbiased estimate would be (self.num_particles  - 1)
    return sample_mean, sample_var

def multiv_gauss_logprob(samples, means, var = None, chol_dec = None):
    # means dim - batch x num particles x samples
    # samples dim - batch x samples
    # chol_dec dim - samples x samples
    # if chol_dec provided assuming all distributions have the same covariance matrix
    if chol_dec is None:
        cholesky_decomposition = torch.cholesky(var)
        var_det = torch.diagonal(cholesky_decomposition, dim1 = 1, dim2 = 2).prod(1)
        var_inv = torch.inverse(var)
    else:
        var_det = chol_dec.diag().prod().unsqueeze(means.size(1))
        var_inv = torch.cholesky_inverse(chol_dec).unsqueeze(0).repeat_interleave(samples.size(0), 0) # dim batch x samples x samples

    log_prob_const = -0.5 * samples.size(1) * torch.log(2 * np.pi) - 0.5 * torch.log(var_det)
    diff = (samples.unsqueeze(1).repeat_interleave(means.size(1), 1) - means) # dim batch x num particles x samples
    log_prob_sample = - 0.5 * torch.diagonal(torch.bmm(torch.bmm(diff, var_inv), diff.transpose(1,2)), dim1 = 1, dim2 = 2)
    return log_prob_const + log_prob_sample

def multiv_gauss_imp_samp_mean_params(samples, log_probs): # samples are in sequence along dimension 1
    norm_probs = F.softmax(log_probs, dim = 1)
    weighted_samples = norm_probs.unsqueeze(2).repeat_interleave(samples.size(2), 2) * samples
    sample_mean = weighted_samples.sum(1)
    samples_zeromean = samples - sample_mean.unsqueeze(1).repeat_interleave(self.num_particles, 1)
    sample_cov = torch.bmm((norm_probs.unsqueeze(2).repeat_interleave(samples_zeromean.size(2), 2) *\
     samples_zeromean).transpose(1,2), samples_zeromean) 
    return sample_mean, sample_cov

######### code for a Differentiable Gaussian Particle Filter with additive Gaussian noise for the dynamics and observation model
class Gaussian_Particle_Dynamics(Proto_Macromodel):
    def __init__(self, model_folder, model_name, info_flow, state_size, observation_size, action_dim, offset, num_particles = 20, device = None, curriculum = None):
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
        self.state_size = state_size
        self.observation_size = observation_size
        self.num_particles = num_particles
        self.offset = offset

        self.dyn_noise = Params(save_folder + "_dyn_noise", load_folder + "_dyn_noise", (self.state_size, self.state_size), device = self.device) 
        self.obs_noise = Params(save_folder + "_obs_noise", load_folder + "_obs_noise", (self.observation_size, self.observation_size), device = self.device) 

        self.dyn_model = ResNetFCN(save_folder + "_dyn_model", load_folder + "_obs_model", self.state_size + self.action_dim, self.state_size, 4, device = self.device)
        self.obs_model = ResNetFCN(save_folder + "_obs_model", load_folder + "_obs_model", self.state_size, self.observation_size, 4, device = self.device)

        self.model_list.append(self.dyn_model)
        self.model_list.append(self.obs_model)
        self.model_list.append(self.dyn_noise)
        self.model_list.append(self.obs_noise)

        if info_flow[model_name]["model_folder"] != "":
            self.load(info_flow[model_name]["epoch_num"])

    def dynamics(self, state_par, action):

        action_par = action.unsqueeze(1).repeat_interleave(self.num_particles, 1)

        state_pred_par_wout_noise = self.dyn_model(torch.cat([state_par.reshape(state_par.size(0) * state_par.size(1), state_par.size(2)),\
            action_par.reshape(action_par.size(0) * action_par.size(1), action_par.size(2))], dim = 1)).reshape(action.size(0), self.num_particles, self.state_size)

        cholesky_decomposition = cholesky_dec(self.dyn_noise()).unsqueeze(0).repeat_interleave(action.size(0),0)

        dyn_noise_samples = sample_multiv_gauss(torch.zeros_like(cholesky_decomposition[:,0]), self.num_particles,\
         chol_dec = cholesky_decomposition)

        state_pred_par = state_pred_par_wout_noise + dyn_noise_samples

        state_pred_mean, state_pred_var = multiv_gauss_params(state_pred_par)

        return state_pred_mean, state_pred_var

    def observation(self, state_pred_par, observation):

        obs_par = self.obs_model(state_pred_par)

        cholesky_decomposition = cholesky_dec(self.obs_noise()) # dim - obs x obs

        obs_logprobs = multiv_gauss_logprob(obs_par, observation, chol_dec = cholesky_decomposition)

        state_mean, state_var = multiv_gauss_imp_samp_mean_params(state_pred_par, obs_logprobs)

        return state_mean, state_var

    def forward(self, input_dict):

        states = input_dict["state"].to(self.device)
        observations = input_dict["observation"].to(self.device)
        actions = input_dict["action"].to(self.device)
        
        epoch =  int(input_dict["epoch"].detach().item())

        state_mean = states[:,0,:]

        state_params_list = []

        if self.curriculum is not None:
            steps = actions.size(1)
            if len(self.curriculum) - 1 >= epoch:
                steps = self.curriculum[epoch]
        else:
            steps = actions.size(1)

        for idx in range(steps):
            action = actions[:,idx]
            observation = observations[:,idx + 1]

            if idx == 0:
                state_par = state_mean.unsqueeze(1).repeat_interleave(self.num_particles, 1)
            else:
                state_par = sample_multiv_gauss(state_mean, self.num_particles, var = state_var)

            state_pred_mean, state_pred_var = self.dynamics(state_par, action)

            state_pred_par = sample_multiv_gauss(state_pred_mean, self.num_particles, var = state_pred_var)

            state_mean, state_var = self.observation(state_pred_par, observation)

            if idx >= self.offset:
                state_params_list.append((state_mean, state_var))

        return {
            "state_params": state_params_list,
        }
