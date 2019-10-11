import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            )

        self.output_layer = nn.Linear(64, num_outputs)
        self.output_layer.weight.data.mul_(0.1)
        self.output_layer.bias.data.mul_(0.0)

        #### Only required for continuous action spaces
        # self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        # print("Input Size: ", x.size())
        h = self.policy_net(x)
        output = self.output_layer(h)

        # print("Output: ", output)
        probs = torch.softmax(output, dim = 1)
        # print("Probs: ", probs)
        return probs


class Value(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            )

        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        output = self.value_net(x)

        return self.value_head(output)



