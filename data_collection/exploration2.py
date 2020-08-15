import yaml
import numpy as np
import scipy
import scipy.misc
import time
import h5py
import sys
import copy
import os
import matplotlib.pyplot as plt
import random
import itertools

sys.path.insert(0, "../robosuite/")

from datacollection_util import *

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

if __name__ == '__main__':

    peg_types = ["Square"]

    obs_keys = [ "force_hi_freq", "proprio", "action", "contact", "joint_pos", "joint_vel", 'image', 'depth']
    peg_dict = {}

    with open("datacollection_params.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    logging_folder = cfg['datacollection_params']['logging_folder']
    logging_data_bool = cfg['datacollection_params']['logging_data_bool']
    display_bool = cfg['datacollection_params']['display_bool']
    num_trajectories = cfg['datacollection_params']['num_trajectories']
    kp = np.array(cfg['datacollection_params']['kp'])
    noise_parameters = np.array(cfg['datacollection_params']['noise_parameters'])
    ctrl_freq = np.array(cfg['datacollection_params']['control_freq'])

    step_threshold = cfg['datacollection_params']['step_threshold']

    workspace_dim = cfg['datacollection_params']['workspace_dim']
    seed = cfg['datacollection_params']['seed']

    random.seed(seed)
    np.random.seed(seed)

    if display_bool:
        obs_keys = [ "force_hi_freq", "proprio", "action", "contact", "joint_pos", "joint_vel"]


    if os.path.isdir(logging_folder) == False and logging_data_bool == 1:
        os.mkdir(logging_folder )

    print("Robot operating with control frequency: ", ctrl_freq)
    env = robosuite.make("PandaPegInsertion", 
    has_renderer= display_bool, ignore_done=True,\
    use_camera_obs= not display_bool, 
    has_offscreen_renderer=not display_bool, 
    gripper_visualization=True, 
    control_freq=ctrl_freq,\
    gripper_type ="SquarePegwForce", 
    controller='position', 
    camera_depth=True,
    camera_width=128,
    camera_height=128,
    )
    # env.viewer.set_camera(camera_id=2)

    tol = 0.008
    tol_ang = 100 #this is so high since we are only doing position control

    ori_action = np.array([np.pi, 0, np.pi])
    plus_offset = np.array([0, 0, 0.07, 0,0,0])
    peg_idx = 0

    hole_poses = []
    for idx, peg_type in enumerate(peg_types):
        peg_top_site = peg_type + "Peg_top_site"
        peg_bottom_site = peg_type + "Peg_bottom_site"
        top = np.concatenate([env._get_sitepos(peg_top_site) - np.array([0, 0, 0.01]), ori_action])
        # print(top)
        # print(env._get_sitepos(peg_bottom_site))
        top_height = top[2]
        hole_poses.append((top, peg_type))


    macro_actions = slidepoints(workspace_dim, num_trajectories)

    file_num = 0

    obs = {}

    fixed_params = (kp, noise_parameters, tol, tol_ang, step_threshold, display_bool,top_height, obs_keys)

    noise_list = []

    def append_x(l, item, times):
        for i in range(times):
            l.append(item)
    # append_x(noise_list, 0.0, 50)
    append_x(noise_list, 0.1, 20)


    # print(noise_list)
    option_file_num = 0

    for noise in noise_list:
        peg_type = "Square"
        hole_type = "Square"
        hole_idx = peg_types.index(hole_type)

        gripper_type = peg_type + "PegwForce" 

        env.reset()
        if display_bool: 
            env.viewer.set_camera(camera_id=3)

        top_goal = hole_poses[hole_idx][0]

        hole_vector = np.zeros(len(peg_types))
        peg_vector = np.zeros(len(peg_types))

        hole_vector[hole_idx] = 1
        peg_vector[peg_types.index(peg_type)] = 1

        points_list = []
        point_idx = 0

        goal_noise = np.zeros(6)
        goal_noise[:3] = np.random.normal(0.0, 1.0, 3) *0.01

        # macro_action = macro_actions[i]
        # init_point = np.concatenate([macro_action[:3] + top_goal[:3], ori_action])
        # final_point = np.concatenate([macro_action[6:] + top_goal[:3], ori_action])

        points_list.append((top_goal + plus_offset + goal_noise, 0, "top plus"))
        points_list.append((top_goal + np.array([0, 0, 0.04, 0,0,0]) + goal_noise, 0, "top"))
        goal_noise[:3] = np.random.normal(0.0, 1.0, 3) *0.01


        points_list.append((top_goal + np.array([0, 0, 0.03, 0,0,0]) + goal_noise, 0, "top"))
        goal_noise[:3] = np.random.normal(0.0, 1.0, 3) *0.01

        points_list.append((top_goal+ np.array([0, 0, 0.02, 0,0,0]) + goal_noise, 0, "init_point"))
        goal_noise[:3] = np.random.normal(0.0, 1.0, 3) *0.01

        points_list.append((top_goal+ np.array([0, 0, 0.02, 0,0,0]) + goal_noise, 0, "init_point"))
        goal_noise[:3] = np.random.normal(0.0, 1.0, 3) *0.02
        points_list.append((top_goal+ np.array([0, 0, 0.01, 0,0,0]) + goal_noise, 0, "init_point"))
        
        for i in range(10): 
            goal_noise = np.zeros(6)

            goal_noise[:2] = np.random.normal(0.0, 1.0, 2) *0.02
            points_list.append((top_goal+ np.array([0, 0, 0.01, 0,0,0]), 1, "init_point"))
        
        for i in range(10):
            goal_noise = np.zeros(6)

            goal_noise[:2] = np.random.normal(0.0, 1.0, 2) *0.02
            points_list.append((top_goal + goal_noise, 1, "init_point"))

        for i in range(10): 
            goal_noise = np.zeros(6)

            goal_noise[:2] = np.random.normal(0.0, 1.0, 2) *0.02
            points_list.append((top_goal+ np.array([0, 0, 0.01, 0,0,0]), 1, "init_point"))
        
        for i in range(10):
            goal_noise = np.zeros(6)

            goal_noise[:2] = np.random.normal(0.0, 1.0, 2) *0.02
            points_list.append((top_goal + goal_noise, 1, "init_point"))
       


        # points_list.append((final_point, 1, "final_point"))
        # points_list.append((top_goal + plus_offset, 0, "top plus"))

        # point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list, point_idx, obs)
        # point_idx, done_bool, obs = movetogoal(env, top_goal, fixed_params, points_list,  point_idx, obs)

        # print("moved to initial position")

        obs_dict = {}

        while point_idx < len(points_list):
            # if len(obs_dict.keys())>0: 
            #   print(points_list[point_idx][1])
            #   print("num: ", len(obs_dict['proprio']))
            # else:
            #   try: 
            #       print(points_list[point_idx][1], " ... nothing yet")
            #   except:
            #       pass
            point_idx, done_bool, obs, obs_dict = movetogoal(env, top_goal, 
                                                fixed_params, points_list, point_idx, 
                                                obs, obs_dict, noise_std=noise)

            # print("points: ", len(obs_dict['proprio']))
            if len(obs_dict['proprio']) >1000: 
                if logging_data_bool == 1:
                    file_num += 1
                    option_file_num += 1
                    # print("On ", file_num, " of ", len(macro_actions) * len(peg_hole_options), " trajectories")
                    file_name = logging_folder + "exploration_noise" + str(noise) + "_" + str(option_file_num).zfill(4) + ".h5"

                    dataset = h5py.File(file_name, 'w')

                    for key in obs_dict.keys():

                        key_list = obs_dict[key]

                        if key == obs_keys[0]:
                            print("Number of points: ", len(key_list))

                        key_array = np.concatenate(key_list, axis = 0)
                        chunk_size = (1,) + key_array[0].shape
                        dataset.create_dataset(key, data= key_array[:1000], chunks = chunk_size)


                dataset.close()

                print("Saving to file: ", file_name)
                break

    print("Finished Data Collection")
