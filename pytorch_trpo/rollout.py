import argparse
import os
from itertools import count
import time
import datetime

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from real_kuka_gym.envs.kuka_reaching_env import MultimodalEnv

# from gym_sai2.envs import Peg1Env, Peg1MultimodalEnv

import tensorboardX

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="peg1-multimodal-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--checkpoint', default=None, metavar='G',
                    help='path to checkpoint')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--log-dir', type=str, default='/scr1/Developer/Projects/pytorch-trpo/log',
                    help='where to save tensorboard data')
parser.add_argument('--eval-interval', type=int, default=10, help='how often to eval')
parser.add_argument('--eval-eps', type=int, default=5, help='how many times to eval')
args = parser.parse_args()

env = MultimodalEnv(model=args.rep_model, controller_type=args.controller)
args.z_dim = 128
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

summary_writer = tensorboardX.SummaryWriter(log_dir)


def select_action(state, deterministic=False):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    if not deterministic:
        action = torch.normal(action_mean, action_std)
    else:
        action = action_mean  # action is mode
    return action

running_state = ZFilter((num_inputs,), clip=5)

ckpt = torch.load(args.checkpoint)
policy_net.load_state_dict(ckpt['policy_net'])
value_net.load_state_dict(ckpt['value_net'])

running_state.rs._M = ckpt['running_M']
running_state.rs._S = ckpt['running_S']
running_state.rs._n = ckpt['running_n']

eval_hole = []
eval_rewards = []
eval_completed = []
eval_touched = []

for eval_episode in range(args.eval_eps):

    ep_reward = 0 
    hole = False
    inserted = False
    touched = False

    env.unwrapped.move_box_reset()
    state = env.reset()
    state = running_state(state)
    for t in range(500):
        action = select_action(state, deterministic=False)
        action = action.data[0].numpy()
        next_state, reward, done, info = env.step(action)
        
        value = env.unwrapped.observation_space_dict["goal_delta_position"]
        goal_delta_pos = env.unwrapped.observation_buffer[value[1][0]:value[1][1]]

        if np.abs(goal_delta_pos[0]) < 0.05 and \
            np.abs(goal_delta_pos[1]) < 0.05 and \
            goal_delta_pos[2] < 0.106 and goal_delta_pos[2] > 0.10:
                touched = True

        ep_reward += reward
        state = running_state(next_state)
        hole = info['hole'] if not hole else hole
        inserted = info['inserted'] if not inserted else inserted
        if inserted:
            break

    eval_touched.append(touched)
    eval_hole.append(hole)
    eval_completed.append(inserted)
    eval_rewards.append(ep_reward)

    reward_batch = np.mean(np.asarray(eval_rewards))
    inserted_batch =  np.mean(np.asarray(eval_hole))
    completed_batch = np.mean(np.asarray(eval_completed))
    # print('touched:', touched, 'inserted:', hole, 'completed:', inserted)
    print('reward:', np.mean(np.asarray(eval_rewards)))
    print('touched:', np.mean(np.asarray(eval_touched)))
    print('inserted:', np.mean(np.asarray(eval_hole)))
    print('completed:', np.mean(np.asarray(eval_completed)))


    summary_writer.add_scalar('train_stats/mean_reward', reward_batch, i_training_episode)
    summary_writer.add_scalar('train_stats/mean_inserted', inserted_batch, i_training_episode)
    summary_writer.add_scalar('train_stats/mean_completed', completed_batch, i_training_episode)
