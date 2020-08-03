import yaml
import numpy as np
import time
import h5py
import sys
import copy
import os
import random
import itertools
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

import project_utils as pu

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import SensSearchWrapper
import models_modules as mm
import torch

import rlkit
from rlkit.torch import pytorch_util as ptu

import perception_learning as pl
import utils_sl as sl

if __name__ == '__main__':
	###################################################
	### Loading run parameters from yaml file
	#####################################################
	with open("estimate_observation_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	display_bool = cfg["logging_params"]["display_bool"]
	collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
	ctrl_freq = cfg["control_params"]["control_freq"]
	horizon = cfg["task_params"]["horizon"]
	image_size = cfg['logging_params']['image_size']
	collect_depth = cfg['logging_params']['collect_depth']
	camera_name = cfg['logging_params']['camera_name']
	seed = cfg['task_params']['seed']
	ctrl_freq = cfg['control_params']['control_freq']
	noise_scale  = 0.5
	cfg['task_params']['noise_scale'] = noise_scale
	logging_folder = cfg["logging_params"]["logging_folder"]
	num_samples = cfg['logging_params']['num_samples']

	name = "estimate_observation_params.yml"

	print("Saving ", name, " to: ", logging_folder + name)

	with open(logging_folder + name, 'w') as ymlfile2:
		yaml.dump(cfg, ymlfile2)

	##########################################################
	### Setting up hardware for loading models
	###########################################################
	use_cuda = cfg['use_cuda']
	device = torch.device("cuda:0" if use_cuda else "cpu")
	random.seed(seed)
	np.random.seed(seed)
	ptu.set_gpu_mode(use_cuda, gpu_id=0)
	ptu.set_device(0)

	if use_cuda:
	    torch.cuda.manual_seed(seed)
	else:
	    torch.manual_seed(seed)

	if use_cuda:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")
	
	##############################
	### Setting up the test environment and extracting the goal locations
	######################################################### 
	print("Robot operating with control frequency: ", ctrl_freq)
	robo_env = robosuite.make("PandaPegInsertion",\
		has_renderer= display_bool,\
		ignore_done=True,\
		use_camera_obs= not display_bool and collect_vision_bool,\
		has_offscreen_renderer = not display_bool and collect_vision_bool,\
		gripper_visualization=False,\
		control_freq=ctrl_freq,\
		gripper_type ="CrossPegwForce",\
		controller='position',\
		camera_name=camera_name,\
		camera_depth=collect_depth,\
		camera_width=image_size,\
		camera_height=image_size,\
		horizon = horizon)

	if 'info_flow' in cfg.keys():
		ref_model_dict = pl.get_ref_model_dict()
		model_dict = sl.declare_models(ref_model_dict, cfg, device)	

		if 'SAC_Policy' in cfg['info_flow'].keys():
			data = torch.load(self.cfg['SAC_Policy']["model_folder"] + "itr_" + str(self.cfg['SAC_Policy']["epoch"]) + ".pkl")
			model_dict['SAC_Policy'] = data['exploration/policy'].to(device)
			model_dict['policy'] = model_dict['SAC_Policy']
		else:
			spiral_mp = pu.Spiral2D_Motion_Primitive()
			# best parameters from parameter grid search
			spiral_mp.rt = 0.026
			spiral_mp.nr = 2.2
			spiral_mp.pressure = 0.003
			model_dict['policy'] = spiral_mp.trajectory

		if 'History_Encoder_Transformer' in model_dict.keys():
			model_dict['encoder'] = model_dict['History_Encoder_Transformer']
			model_dict['sensor'] = model_dict['History_Encoder_Transformer']
	else:
		raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode= 2, **model_dict)

	# ll = likelihood
	ll_size = (env.sensor.num_states, 2, env.sensor.num_observationsm)
	ll_model = mm.Params('likelihood_model', ll_size).to(device)
	ll_model.model.p[:] = 0.0

	env.sensor.likelihood_model = ll_model
	env.sensor.model_list.append(env.sensor.likelihood_model)
	# env.sensor.save(9999, env.sensor.loading_folder)

	for trial_num in range(num_samples):
		env.reset(initialize=False)
		cand_idx = env.robo_env.cand_idx
		obs_idxs, pos_est = env.big_step(cand_idx)

		print("Observation: ", env.robo_env.hole_names[obs_idxs[1]], " Ground Truth: ", env.robo_env.hole_sites[cand_idx][0])
		print("Pos Estimate: ", pos_est, " Ground Truth Rel Pos: ", env.current_pos2D)
		print("Pos Error: ", np.linalg.norm(pos_est.cpu().numpy() - env.current_pos2D))
		
		ll_idxs = tuple([cand_idx] + list(obs_idxs))

		env.sensor.likelihood_model.model.p[ll_idxs] += 1
		# print(env.sensor.likelihood_model())

	env.sensor.likelihood_model.p[:] = env.sensor.likelihood_model.p[:] / num_samples
	print(env.sensor.likelihood_model())
	# env.sensor.save(9999, env.sensor.loading_folder)	



