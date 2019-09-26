import os
from itertools import count
import time
import datetime
import sys
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

import yaml
import numpy as np
import scipy
import scipy.misc
import h5py
import sys

sys.path.append("/scr-ssd/sensorimotor_search") 

from sim_env.block_stacking import *
from policy_training.models import *

import tensorboardX


if __name__ == '__main__':

    ######## loading training parameters
    with open("trpo_params.yml", 'r') as ymlfile:
        cfg_0 = yaml.safe_load(ymlfile)

    debugging_val = cfg_0['debugging_params']['debugging_val']
    saving_val = cfg_0['save_trpo_params']['saving_val']

    use_gpu_flag = cfg_0['training_params']['use_GPU']

    z_dim = cfg_0["rep_model_params"]["z_dim"]
    num_actions = cfg_0["rep_model_params"]["num_actions"]
    threshold = cfg_0["rep_model_params"]["threshold"]
    model_path = cfg_0['rep_model_params']['oe_path']

    logging_folder = cfg_0['logging_params']['logging_folder']
    run_notes = cfg_0['logging_params']['run_notes']

    gamma = cfg_0['training_params']['gamma']
    tau = cfg_0['training_params']['tau']
    batch_size = cfg_0['training_params']['batch_size']
    eval_interval = cfg_0['evaluation_params']['eval_interval']
    horizon = cfg_0['training_params']['horizon']
    num_frames_max = cfg_0['training_params']['num_frames']
    l2_reg = cfg_0['training_params']['l2_reg']
    max_kl = cfg_0['training_params']['max_kl']
    damping = cfg_0['training_params']['damping']


    with open("/scr-ssd/sensorimotor_search/sim_env/game_params.yml", 'r') as ymlfile:
        cfg_1 = yaml.safe_load(ymlfile)

    cell_size = cfg_1['game_params']['cell_size']
    cols = cfg_1['game_params']['cols']
    rows = cfg_1['game_params']['rows']
    delay = cfg_1['game_params']['delay']
    maxfps = cfg_1['game_params']['maxfps']
    num_goals = cfg_1['game_params']['num_goals']

    ######## standard debugging flags ##############
    if saving_val == 1:
        saving_flag = True
        if os.path.isdir(cfg['save_trpo_params']['saving_path']) == False:
            os.mkdir(cfg['save_trpo_params']['saving_path'])

        saving_folder = cfg['save_trpo_params']['saving_path']

    else:
        saving_flag = False

    if debugging_val == 1.0:
        debugging_flag = True
        var = input("Debugging flag activated. No Results will be saved. Continue with debugging [y,n]: ")
        if var != "y":
            debugging_flag = False
    else:
        debugging_flag = False

    ####### setting up use of GPU ########################
    use_cuda = use_gpu_flag and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True

    torch.set_default_tensor_type('torch.DoubleTensor')  #### NOTE DEFAULT DATA TYPE

    ####### loading policy network ##################
    policy_net = Observations_Encoder(rows, cols, 3, z_dim, num_goals, num_actions, threshold, device = device).to(device)

    policy_net.train()
    
    #### Initializing Environment
    env = env_BP_w_display(num_goals)

    #### this is a low pass filter which improves training
    running_state = ZFilter(((3, rows, cols)), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)


    if model_path != "":
        print("MODEL LOADED")
        ckpt = torch.load(model_path)
        policy_net.load_state_dict(ckpt['policy_net'])
        value_net.load_state_dict(ckpt['value_net'])

        running_state.rs._M = ckpt['running_M']
        running_state.rs._S = ckpt['running_S']
        running_state.rs._n = ckpt['running_n']
    else:
        print("NO RL MODEL LOADED")

    ##### Logging Folders #####

    if debugging_flag == False:
        if not os.path.exists(logging_folder):
            os.makedirs(logging_folder)

        t_now = time.time()
        time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d_%H%M%S_%f')

        log_dir = logging_folder + "/" + time_str + "_" + run_notes + '_gamma_{}_tau_{}_bs_{}'.\
        format(gamma, tau, batch_size)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        summary_writer = tensorboardX.SummaryWriter(log_dir)
        print("Logging Folder: ", log_dir)

    ### Defining Loss function and optimization

    val_loss = 0

    def update_params(batch, gamma, tau, l2_reg, max_kl, damping):
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state).squeeze()
        values = value_net(Variable(states))
        x_poses = torch.Tensor(batch.x_pos)

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            global val_loss
            set_flat_params_to(value_net, torch.Tensor(flat_params))
            for param in value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = value_net(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            value_loss.backward()
            val_loss = value_loss.item()
            # print("Value Loss: ", val_loss)
            return (value_loss.data.double().item(), get_flat_grad_from(value_net).data.double().numpy())

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        # print("States: ", states)
        # print("States Size: ", states.size())

        probs = policy_net((states, x_poses)).squeeze()

        # print("Actions: ", actions)
        # print("probs: ", probs)
        fixed_log_prob = (torch.log(probs)*Variable(actions)).sum(1).data.clone() #normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            
            # print("Action Size: ", actions.size())

            if volatile:
                with torch.no_grad():
                    probs = policy_net(Variable(states)).squeeze()
            else:
                probs = policy_net(Variable(states)).squeeze() 
                              

            # print("Probs Size: ", probs.size())
            log_prob = (torch.log(probs)*Variable(actions)).sum(1) #normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)


            # print("Log Probs Size: ", log_prob.size())
            # print("Advantages Size: ", advantages.size())
            action_loss = -Variable(advantages).squeeze() * torch.exp(log_prob - Variable(fixed_log_prob))
            # print("Action Loss Size: ", action_loss.size())
            # print("Action Loss: ", action_loss.mean().item())
            return action_loss.mean()


        def get_kl():
            actprobs = policy_net((Variable(states), Variable(x_poses))) + 1e-8

            old_actprobs = Variable(actprobs.data)

            kl = old_actprobs * torch.log(old_actprobs / actprobs)

            return kl.sum(1, keepdim=True)


        probs_old = policy_net((Variable(states), Variable(x_poses)))

        loss = trpo_step(policy_net, get_loss, get_kl, max_kl, damping)
        
        probs_new = policy_net((Variable(states), Variable(x_poses))) + 1e-8

        kl = torch.sum(probs_old * torch.log(probs_old / probs_new), 1)

        return loss, kl.mean()

    i_training_episode = 0

    for i_episode in count(0):
        memory = Memory()

        num_steps = 0
        num_frames = 0
        reward_batch = 0
        num_episodes = 0
        min_reward_sum, max_reward_sum = None, None


        if eval_interval != 1: 
            while num_steps < batch_size * horizon and num_frames < num_frames_max:
                
                if saving_flag:
                    board_image_list = []
                    action_list = []

                board_image, x_pos, reward = env.step(6)

                goal_bound_array = env.goal_bound_array

                state = torch.from_numpy(running_state(board_image)).double().to(device).unsqueeze(0)

                i_training_episode += 1
                print("collecting trajectory #{} for training".format(i_training_episode))

                reward_sum = 0

                for t in range(horizon): # Don't infinite loop while learning

                    probs = policy_net((state, x_pos, goal_bound_array))

                    action = probs.multinomial(1)

                    action_array = np.zeros((num_actions))

                    action_array[action] = 1

                    if action > 1:
                        action += 1

                    prev_x_pos = copy.copy(x_pos)

                    board_image, x_pos, reward = env.step(action.numpy())

                    if saving_flag:

                        board_image_list.append(np.expand_dims(scipy.misc.imresize(curr_image, size=(128,128,3)), axis = 0).astype(np.uint8))        

                        action_list.append(action)

                    if t == 0:
                        reward = 0

                    if reward != 0:
                        print("Reward: ", reward)

                    reward_sum += reward

                    next_state = torch.from_numpy(running_state(board_image)).double().to(device).unsqueeze(0)

                    mask = 1

                    if t == horizon - 1:
                        mask = 0

                    memory.push(state.cpu().detach().numpy(), np.array([action_array]), mask, next_state.cpu().detach().numpy(), reward, prev_x_pos, x_pos)

                    state = next_state

                if saving_flag:
                    file_name = saving_folder + "trpo_horizon_" + str(horizon) + "_iteration_" + str(i_training_episode) + ".h5"

                    dataset = h5py.File(file_name, 'w')

                    board_image_array = np.concatenate(board_image_list, axis = 0)
                    board_image_array = np.transpose(board_image_array, (0,2,1,3))
                    chunk_size = (1,) + board_image_array[0].shape 
                    dataset.create_dataset("board_image", data= board_image_array, chunks = chunk_size)

                    action_array = np.concatenate(action_list, axis = 0)
                    dataset.create_dataset("action", data= action_array)

                    dataset.close()

                num_steps += (t-1)
                num_episodes += 1

                reward_batch += reward_sum
                max_reward_sum = reward_sum if max_reward_sum is None else max(reward_sum, max_reward_sum)
                min_reward_sum = reward_sum if min_reward_sum is None else min(reward_sum, min_reward_sum)

            board_image, x_pos, reward = env.step(6)

            goal_bound_array = env.goal_bound_array

            reward_batch /= num_episodes

            batch = memory.sample()
            action_loss, kl_loss = update_params(batch, gamma, tau, l2_reg, max_kl, damping)
            print("Policy Updated!")

            if debugging_flag == False:

                summary_writer.add_scalar('train_stats/reward_max', max_reward_sum, i_training_episode)
                summary_writer.add_scalar('train_stats/reward_min', min_reward_sum, i_training_episode)

                summary_writer.add_scalar('losses/policy_loss', action_loss.item(), i_training_episode)
                summary_writer.add_scalar('losses/kl', kl_loss.item(), i_training_episode)
                summary_writer.add_scalar('losses/value_loss', val_loss, i_training_episode)

            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))

        # if i_episode % args.eval_interval == 0: 
        #     if args.eval_eps > 0 & i_episode == 0:

        #         eval_rewards = []
        #         eval_inserted = []
        #         eval_completed = []
        #         eval_touched = []

        #         for eval_episode in range(args.eval_eps):

        #             ep_reward = 0 
        #             inserted = False
        #             completed = False
        #             touched = False

        #             state = env.step(5)
        #             state = running_state(state)

        #             print("collecting trajectory #{} for evaluation".format(eval_episode))
        #             for t in range(args.horizon):
        #                 action = select_action(state, deterministic=False)
        #                 action = action.data[0].numpy()
        #                 next_state, reward, done, info = env.step(action)
        #                 ep_reward += reward
        #                 state = running_state(next_state)

        #                 value = env.unwrapped.observation_space_dict["goal_delta_position"]
        #                 goal_delta_pos = env.unwrapped.observation_buffer[value[1][0]:value[1][1]]



                # summary_writer.add_scalar('eval_stats/mean_inserted', np.mean(np.asarray(eval_inserted)), i_episode)
                # summary_writer.add_scalar('eval_stats/mean_reward', np.mean(np.asarray(eval_rewards)), i_episode)
                # summary_writer.add_scalar('eval_stats/mean_completed', np.mean(np.asarray(eval_completed)), i_episode)
                # summary_writer.add_scalar('eval_stats/mean_touched', np.mean(np.asarray(eval_touched)), i_episode)

            # print("Done Evaluation")
            # time.sleep(3)
            # # exit(0)
            # os._exit

            if eval_interval !=1 and debugging_flag == False and i_episode % 5 == 0: 
                ckpt_path = os.path.join(log_dir, "trial0_" + "itr_{}.ckpt".format(i_episode))
                print('saving checkpoint to {}'.format(ckpt_path))
                model_checkpoints = {
                    'policy_net': policy_net.state_dict(),
                    'value_net': value_net.state_dict(),
                    'running_M': running_state.rs._M,
                    'running_S': running_state.rs._S,
                    'running_n': running_state.rs._n,
                }
                torch.save(model_checkpoints, ckpt_path)

        num_frames += num_steps
        # if num_frames >= args.num_frames:
        #     break
