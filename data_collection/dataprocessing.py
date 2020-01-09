import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import os
import datacollection_util as dc_T
import keyboard

sys.path.insert(0, "../robosuite/") 

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

	with open("datacollection_params.yml", 'r') as ymlfile:
		cfg = yaml.safe_load(ymlfile)

	filename_list = cfg['datacollection_params']['logging_folder']

    for filename in filename_list:
    	dataset = h5py.File(filename, 'a')

    	actions = np.array(dataset["action"])

    	actions_pos = actions[:3]
    	actions_quad = actions[:,3:]
    	actions_euler_list = []

    	for idx in range(actions.shape[0]):
    		eul = T.mat2euler(T.quat2mat(actions_quad[idx])
    		actions_euler_list.append(np.expand_dims(np.concatenate([actions_pos[idx], eul]), axis = 0))



