import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer, SelectiveEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.sac import SACTrainer
from rlkit.launchers.launcher_util import run_experiment
import sys 

import torch
import argparse
import json
import numpy as np

import robosuite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import SensSearchWrapper
import os
import yaml

# from models import *
# from supervised_learning_utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def experiment(variant):

    config_path = variant['config_path']

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    logging_folder = cfg["logging_params"]["logging_folder"]
    name = "datacollection_params"

    print("Saving ", name, " to: ", logging_folder + name + ".yml")

    with open(logging_folder + name + ".yml", 'w') as ymlfile2:
        yaml.dump(cfg, ymlfile2)

    display_bool = cfg["logging_params"]["display_bool"]
    collect_vision_bool = cfg["logging_params"]["collect_vision_bool"]
    ctrl_freq = cfg["control_params"]["control_freq"]
    horizon = cfg["control_params"]["horizon"]
    image_size = cfg['logging_params']['image_size']
    collect_depth = cfg['logging_params']['collect_depth']
    camera_name = cfg['logging_params']['camera_name']

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
        device = torch.device("cuda")
        ref_model_dict = get_ref_model_dict()
        model_dict = declare_models(ref_model_dict, cfg, device)

        expl_env = SensSearchWrapper(robo_env, config_path, mode = "train2insert_choose_match", encoder = model_dict['History_Encoder'])
        eval_env = SensSearchWrapper(robo_env, config_path, mode = "differ_choose", encoder = model_dict['History_Encoder'])
    else:
        expl_env = SensSearchWrapper(robo_env, config_path, mode = "train2insert_choose_match")
        eval_env = SensSearchWrapper(robo_env, config_path, mode = "differ_choose")

    obs_dim = cfg['state_dim']
    action_dim = cfg['action_dim']

    # print(cfg['state_dim'])
    # print(cfg['action_dim'])
    
    if variant['file'] is None:
        M = variant['layer_size']
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M],
        )
        eval_policy = policy #MakeDeterministic(policy)
    else:
        print("Loading policy ", variant['file'])
        data = torch.load(variant['file'])
        eval_policy = data['evaluation/policy']
        policy = data['exploration/policy']
        qf1 = data['trainer/qf1']
        qf2 = data['trainer/qf2']
        target_qf1= data['trainer/target_qf1']
        target_qf2= data['trainer/target_qf2']

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        return_info=False,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        return_info=False,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file', type=str, default=None,
    #                     help='path to the snapshot file')
    parser.add_argument('--cpu', default=False, action='store_true')
    # parser.add_argument('--load', type=str, required=True)
    # parser.add_argument('--epoch_stats', type=str, required=True)
    # parser.add_argument('--env', type=str, default='default', choices=ENV_TYPES)
    # parser.add_argument('--test', action='store_true')
    # parser.add_argument('--vae_model', type=str, required=True)
    # parser.add_argument('--goal', type=float, nargs='+', default=None) 
    args = parser.parse_args()
    #todo: pip install instead
    config_path="datacollection_params.yml"

    with open(config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    horizon = cfg["control_params"]["horizon"]
    logging_folder = cfg["logging_params"]["logging_folder"]
    logging_data_bool = cfg["logging_params"]["logging_data_bool"]
    num_paths = 10

    if logging_data_bool == 1.0:
        eval_num_paths = num_paths
    else:
        eval_num_paths = 1.0

    if cfg['pretrained_path'] is "":
        file = None
    else:
        file = cfg['pretrained_path']

    variant = dict(
        file=file,
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=400,
            num_eval_steps_per_epoch= eval_num_paths * horizon,
            num_trains_per_train_loop=num_paths * 100,
            num_expl_steps_per_train_loop=num_paths * horizon,
            min_num_steps_before_training=0,
            max_path_length=horizon,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=10,
            use_automatic_entropy_tuning=True,
            target_entropy=-6,
        ),
        cuda=True,
        config_path="datacollection_params.yml",
    )

    run_experiment(experiment,
                   exp_prefix='sac-robosuite',
                   variant=variant,
                   use_gpu=True,
                   mode='local',
                   snapshot_mode="all",
                   base_log_dir=logging_folder)
