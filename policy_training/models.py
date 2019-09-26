import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Categorical
import torchvision
import copy
import numpy as np

def get_conv_layer_params(rows, cols, input_chan, output_chan):

    #assume output_chan is a multiple of 2
    chan_list = [input_chan, 16]

    while chan_list[-1] != output_chan:
        chan_list.append(chan_list[-1] * 2)

    prime_fact_rows = get_prime_fact(rows)
    prime_fact_cols = get_prime_fact(cols)

    if len(prime_fact_cols) > len(prime_fact_rows):

        while len(prime_fact_cols) > len(prime_fact_rows):
            prime_fact_rows.append(1)

    elif len(prime_fact_cols) < len(prime_fact_rows):

        while len(prime_fact_cols) < len(prime_fact_rows):
            prime_fact_cols.append(1)

    if len(prime_fact_cols) > len(chan_list):

        while len(prime_fact_cols) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact_cols) < len(chan_list):
        print("here is the problem")

        idx = 1

        while len(prime_fact_cols) < len(chan_list):

            prime_fact_cols.insert(idx, 1)
            prime_fact_rows.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact_cols):

                idx = 1

    e_p = np.zeros((8,len(prime_fact_rows))).astype(np.int16) #encoder parameters

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact_rows)):

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact_rows[idx] == 3:
            e_p[2,idx] = 5
            e_p[4, idx] = 3
        elif prime_fact_rows[idx] == 2:
            e_p[2,idx] = 4
            e_p[4, idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[4, idx] = 1
        # fourth row col kernel
        # sixth row col stride
        if prime_fact_cols[idx] == 3:
            e_p[3,idx] = 5
            e_p[5, idx] = 3
        elif prime_fact_cols[idx] == 2:
            e_p[3,idx] = 4
            e_p[5, idx] = 2
        else:
            e_p[3,idx] = 3
            e_p[5, idx] = 1

        # seventh row row padding
        e_p[6, idx] = 1
        # eighth row col padding
        e_p[7,idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    # print("E P list")
    # print(e_p_list)

    return e_p_list

def get_prime_fact(num):

    #assume num is factorized by powers of 2 and 3
    temp_num = copy.copy(num)
    prime_fact_list = []

    while temp_num != 1:

        if temp_num % 3 == 0:
            temp_num = temp_num / 3
            prime_fact_list.append(3)

        elif temp_num % 2 == 0:
            temp_num = temp_num / 2
            prime_fact_list.append(2)

    return prime_fact_list  

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Observations_Encoder(nn.Module):

    def __init__(self, rows, cols, init_C, z_dim, num_goals, num_actions, threshold, device = None):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.rows = rows
        self.cols = cols
        self.num_goals = num_goals
        self.num_actions = num_actions
        self.threshold = threshold

        # -----------------------
        # observation encoder
        # -----------------------

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_conv_layer_params(rows, cols, init_C, self.z_dim) # encoder parameters

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.Conv2d(e_p[0] , e_p[1], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))

            if idx != (len(e_p_list) - 1):
                layer_list.append(nn.BatchNorm2d(e_p[1]))

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        layer_list.append(Flatten())
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        # layer_list.append(nn.Softmax(dim=1)) # only for learning a multinomial distribution
      	
        self.observation_encoder = nn.Sequential(*layer_list)

        layer_list = []

        layer_list.append(nn.Linear(self.z_dim + 1, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, 1))

        self.confidence_prediction = nn.Sequential(*layer_list)

        layer_list = []

        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.num_goals - 1))

        self.switching_policy = nn.Sequential(*layer_list)

        layer_list = []

        layer_list.append(nn.Linear(self.z_dim + 1, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.z_dim))
        layer_list.append(nn.LeakyReLU(0.1, inplace = True))
        layer_list.append(nn.Linear(self.z_dim, self.num_actions))

        self.motion_policy = nn.Sequential(*layer_list)
        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_tuple):

        image, x_pos, goals, goal_bounds = input_tuple
        #### image must be a tensor of size batch_size, 3, rows, cols
        #### x_pos must be a tensor of size batch_size, 1
        #### goals must be a tensor of size batch_size, 1
        #### goal_bounds must be a tensor of size batch_size, num_goals, 2

        #### calculating latent space representation of image
        z = self.observation_encoder(image) #### z is a tensor of size batch_size, z_dim

        #### calculating confidence score that policy is exploring the right area
        confidence_score = torch.sigmoid(self.confidence_prediction(torch.cat((z.transpose(0,1), goals.transpose(0,1))).transpose(0,1)))

        ##### setting new exploration area if confidence score is less than the confidence score        
        if self.num_goals == 2:
            delta_goal = torch.where(confidence_score < self.threshold, torch.ones_like(goals), torch.zeros_like(goals))

        else:
            delta_goal = torch.where(confidence_score < self.threshold, self.switching_policy(z).multinomial(1) + 1, torch.zeros_like(goals))

        new_goals = (delta_goal + goals) % self.num_goals

        ##### calculating and "clipping" action probs based on goal area and state
        action_probs_pred = F.softmax(self.motion_policy(torch.cat((z.transpose(0,1), new_goals.transpose(0,1))).transpose(0,1)), dim = 1)

        left_action = torch.zeros_like(action_probs_pred)

        left_action[:, 0] = 1

        right_action = torch.zeros_like(action_probs_pred)

        right_action[:, 1] = 1

        x_pos_large = x_pos.repeat(1, self.num_goals)
        lower_goal_bound = goal_bounds[np.arange(goal_bounds.size()[0]), new_goals, 0].unsqueeze(1).repeat(1, self.num_goals)
        upper_goal_bound = goal_bounds[np.arange(goal_bounds.size()[0]), new_goals, 1].unsqueeze(1).repeat(1, self.num_goals)

        action_probs_lb_clipped = torch.where(x_pos_large < lower_goal_bound, right_action, action_probs_pred)
        action_probs_clipped = torch.where(x_pos_large > upper_goal_bound, left_action, action_probs_lb_clipped)

        return action_probs_clipped

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

