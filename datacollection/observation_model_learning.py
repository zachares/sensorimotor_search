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
			data = torch.load(cfg['info_flow']['SAC_Policy']["model_folder"] + "itr_" + str(cfg['info_flow']['SAC_Policy']["epoch"]) + ".pkl")
			model_dict['SAC_Policy'] = data['exploration/policy'].to(device)
			model_dict['policy'] = model_dict['SAC_Policy']
			cfg['policy_keys'] = cfg['info_flow']['SAC_Policy']['policy_keys']
			cfg['state_size'] = cfg['info_flow']['SAC_Policy']['state_size'] 
		# else:
		# 	raise Exception('No policy provided to perform evaluaiton')

		for model_name in cfg['info_flow'].keys():
			if model_name == 'SAC_Policy':
				continue

			if cfg['info_flow'][model_name]['sensor']:
				model_dict['sensor'] = model_dict[model_name]
				print("Sensor: ", model_name)

				if model_name == 'StateSensor_wConstantUncertainty':
					likelihood_embedding = model_dict['sensor'].ensemble_list[0][-1]
					likelihood_params = torch.cat([\
						torch.reshape(likelihood_embedding(pu.toTorch(np.array([0]), device).long()), (1,3,3)),\
						torch.reshape(likelihood_embedding(pu.toTorch(np.array([1]), device).long()), (1,3,3)),\
						torch.reshape(likelihood_embedding(pu.toTorch(np.array([2]), device).long()), (1,3,3)),\
						], dim = 0)

					loglikelihood_matrix = F.log_softmax(likelihood_params, dim = 1)
				else:
					loglikelihood_matrix = None


			if cfg['info_flow'][model_name]['encoder']:
				model_dict['encoder'] = model_dict[model_name]
				print("Encoder: ", model_name)

	else:
		model_dict = {}
		# raise Exception('sensor must be provided to estimate observation model')

	env = SensSearchWrapper(robo_env, cfg, selection_mode = 2, **model_dict)

	# env.sensor.likelihood_model.model.p[:] = 3.0
	# prior_samples = env.sensor.likelihood_model.model.p[:].sum().item()
	# env.sensor.success_params.model.p[:] = 0.0

	test_success = cfg['test_success']
	num_tools = 3

	if test_success: # testing policy effectiveness
		print("TESTING POLICY")
		env.mode = 0
		num_samples = num_tools * 50
		success_params = np.ones((num_tools, 2))

	else: # testing network accuracy
		print("TESTING PERCEPTION NETWORKS")
		env.mode = 2
		classification_accuracy = np.zeros(2)
		position_estimation = []
		num_samples = 100

	trial_num = 0
	num_attempts = 3

	# print("Success Params: ", success_params,\
	 # success_params / np.repeat(np.expand_dims(np.sum(success_params, axis = 1), axis = 1), 2, axis = 1))

	while trial_num < num_samples:
		print("Trial Num ", trial_num + 1, " out of ", num_samples, " trials")
		env.robo_env.reload = False
		env.reset(initialize=False, config_type = '3_small_objects_fit')
		action_idx = env.robo_env.cand_idx
		substate_idx = env.robo_env.hole_idx
		tool_idx = env.robo_env.tool_idx
		pos_init = copy.deepcopy(env.robo_env.hole_sites[action_idx][-1][:2])

		for attempt in range(num_attempts):
			if env.done_bool:
				continue

			pos_temp_init = copy.deepcopy(env.robo_env.hole_sites[action_idx][-1][:2])

			got_stuck = env.big_step(action_idx)

			pos_final = env.robo_env.hole_sites[action_idx][-1][:2]	

			pos_actual = env.robo_env.hole_sites[action_idx][-3][:2]

			error_init = np.linalg.norm(pos_temp_init - pos_actual)
			error_final = np.linalg.norm(pos_final - pos_actual)
			
			error_diff_per_step = error_init - error_final

			print("Error Change: ", error_diff_per_step)

			env.robo_env.reload = True
			env.reset(initialize=False, config_type = '3_small_objects_fit')

		if test_success and not got_stuck:
			if env.done_bool:
				success_params[tool_idx,1] += 1
			else:
				success_params[tool_idx,0] += 1

			trial_num += 1

		elif not env.done_bool and not got_stuck:
			pos_final = env.robo_env.hole_sites[action_idx][-1][:2]	

			pos_actual = env.robo_env.hole_sites[action_idx][-3][:2]

			error_init = np.linalg.norm(pos_init - pos_actual)
			error_final = np.linalg.norm(pos_final - pos_actual)
			
			error_diff_per_step = error_init - error_final

			print("Error Change: ", error_diff_per_step)

			position_estimation.append(error_diff_per_step)

			if (env.obs_idx == 0 and env.robo_env.peg_idx == env.robo_env.hole_idx) or\
				(env.obs_idx == 1 and env.robo_env.peg_idx != env.robo_env.hole_idx):
				print("Correct Classification")
				classification_accuracy[0] += 1

			else:
				print("Incorrect Classification")
				classification_accuracy[1] += 1

			trial_num += 1


		# print("Observation: ", env.robo_env.hole_names[obs_idxs[1]], " Ground Truth: ", env.robo_env.hole_sites[cand_idx][0])
		
		# ll_idxs = tuple([tool_idx, substate_idx] + list(obs_idxs))
		# success_idxs = (tool_idx, obs_idxs[0])

		# env.sensor.likelihood_model.model.p[ll_idxs] += 1
		# print(env.sensor.likelihood_model())

	# env.sensor.likelihood_model.model.p[:] = env.sensor.likelihood_model.model.p[:] / (num_samples + prior_samples)
	# env.sensor.success_params.model.p[:] = env.sensor.success_params.model.p[:] / env.sensor.success_params.model.p[:].sum(1).unsqueeze(1).repeat_interleave(2,1)
	# print(env.sensor.likelihood_model())
	# print(env.sensor.success_params())
	# env.sensor.save(9999, env.sensor.loading_folder)

	if test_success:
		print("Success Params: ", success_params,\
		 success_params / np.repeat(np.expand_dims(np.sum(success_params, axis = 1), axis = 1), 2, axis = 1))
	else:
		position_estimation = np.array(position_estimation)
		print("Classification Accuracy: ", classification_accuracy / sum(classification_accuracy))
		print("Mean Position Change: ", np.mean(position_estimation))
		print("STD Position Change: ", np.std(position_estimation))



