import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Categorical
import torchvision
import copy
import yaml
import numpy as np

### DeCNN network
### Network for point clouds
### Network for video
### Network for sequential point clouds
### Sequential data deconv
### a CNN supporting  5 and 7 kernel sizes
### add dilation to your CNNs

def get_2Dconv_params(input_size, output_size):

    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size
    # assume output_chan is a multiple of 2
    # assume input height and width have only two prime factors 2 and 3 for now
    # assume that output height and output width are factors of input height and input width respectively
    chan_list = [input_chan, 16]

    while chan_list[-1] != output_chan:
        chan_list.append(chan_list[-1] * 2)

    prime_fact_rows = get_prime_fact(input_width // output_width)
    prime_fact_cols = get_prime_fact(input_height // output_height)

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

    return e_p_list

def get_1Dconv_params(input_size, output_size):

    input_chan, input_width = input_size
    output_chan, output_width = output_size

    # assume output_chan is a multiple of 2
    # assume input height have only two prime factors 2 and 3 for now
    # assume that output width are factors of input width
    chan_list = [input_chan, 16]

    while chan_list[-1] != output_chan:
        chan_list.append(chan_list[-1] * 2)

    prime_fact = get_prime_fact(input_width // output_width)

    if len(prime_fact) > len(chan_list):

        while len(prime_fact) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact) < len(chan_list):

        idx = 1

        while len(prime_fact) < len(chan_list):

            prime_fact.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact):

                idx = 1

    e_p = np.zeros((5,len(prime_fact))).astype(np.int16) #encoder parameters

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact)):

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact[idx] == 3:
            e_p[2,idx] = 5
            e_p[3, idx] = 3
        elif prime_fact[idx] == 2:
            e_p[2,idx] = 4
            e_p[3, idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[3, idx] = 1
        # seventh row row padding
        e_p[4, idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    return e_p_list

def get_prime_fact(num):

    #assume num is factorized by powers of 2 and 3
    temp_num = copy.copy(num)
    prime_fact_list = []

    while temp_num != 1:

        # if temp_num % 7 == 0:
        #     temp_num = temp_num / 7
        #     prime_fact_list.append(7)

        # elif temp_num % 5 == 0:
        #     temp_num = temp_num / 5
        #     prime_fact_list.append(5)

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

#########################################
# Current Model Types Supported 
########################################

# All models have four other methods

# 1. init - initializes the network with the inputs requested by the user
# 2. forward - returns the output of the network for a given input
# 3. save - saves the model
# 4. load - loads a model if there is a nonempty path corresponding to that model in the yaml file

#### super class of all models for logging and loading models
class Proto_Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = None
        self.model_name = model_name
    
    def forward(self, input):
        return self.model(input)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def save(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.model_name, epoch_num)
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.model_name, epoch_num)
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)

#### a convolutional network
class CONV2DN(Proto_Model):
    def __init__(self, model_name, input_size, output_size, output_activation_layer_bool, flatten_bool, num_fc_layers, device = None):
        super().__init__(model_name + "_cnn")

        # assume output_chan is a multiple of 2
        # assume input height and width have only two prime factors 2 and 3 for now
        # assume that output height and output width are factors of input height and input width respectively
        # activation type leaky relu and network uses batch normalization
        self.device = device

        self.input_size = input_size
        self.output_size = output_size
        output_chan, output_height, output_width = output_size

        self.output_activation_layer_bool = output_activation_layer_bool
        self.flatten_bool = flatten_bool
        self.num_fc_layers = num_fc_layers

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_2Dconv_params(input_size, output_size) # encoder parameters

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.Conv2d(e_p[0] , e_p[1], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))

            if idx != (len(e_p_list) - 1):
                layer_list.append(nn.BatchNorm2d(e_p[1]))

            if idx != (len(e_p_list) - 1) and num_fc_layers == 0 and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        if flatten_bool or num_fc_layers != 0:
            layer_list.append(Flatten())
            num_outputs = output_width * output_height * output_chan

        for idx in range(num_fc_layers):

            layer_list.append(nn.Linear(num_outputs, num_outputs))

            if idx == (num_fc_layers - 1) and output_activation_layer_bool == False:
                continue

            layer_list.append(nn.BatchNorm1d(num_outputs))
            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        self.model = nn.Sequential(*layer_list)

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

#### a 2D deconvolutional network
class DECONV2DN(Proto_Model):
    def __init__(self, model_name, input_size, output_size, output_activation_layer_bool, device = None):
        super().__init__(model_name + "_dcnn")

        # assume output_chan is a multiple of 2
        # assume input height and width have only two prime factors 2 and 3 for now
        # assume that output height and output width are factors of input height and input width respectively
        # activation type leaky relu and network uses batch normalization
        self.device = device

        self.input_size = input_size
        self.output_size = output_size

        self.output_activation_layer_bool = output_activation_layer_bool
        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_2Dconv_params(output_size, input_size) # encoder parameters
        e_p_list.reverse()

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.ConvTranspose2d(e_p[1] , e_p[0], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))

            if idx != (len(e_p_list) - 1):
                layer_list.append(nn.BatchNorm2d(e_p[0]))

            if idx != (len(e_p_list) - 1) and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        self.model = nn.Sequential(*layer_list)

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

    def forward(self, rep):
        assert rep.size()[1] == self.input_size[0], "input channel dim does not match network requirements"

        return self.model(rep.unsqueeze(2).unsqueeze(3).repeat(1,1,2,2))

#### a time series network
class CONV1DN(Proto_Model):
    def __init__(self, model_name, input_size, output_size, output_activation_layer_bool, flatten_bool, num_fc_layers, device = None):
        super().__init__(model_name + "_1dconv")

        # assume output_chan is a multiple of 2
        # assume input height and width have only two prime factors 2 and 3 for now
        # assume that output height and output width are factors of input height and input width respectively
        # activation type leaky relu and network uses batch normalization
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.output_activation_layer_bool = output_activation_layer_bool
        self.flatten_bool = flatten_bool
        self.num_fc_layers = num_fc_layers

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_1Dconv_params(input_size, output_size) # encoder parameters

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.Conv1d(e_p[0] , e_p[1], kernel_size= e_p[2],\
                stride=e_p[3], padding=e_p[4], bias=True))

            if idx != (len(e_p_list) - 1):
                layer_list.append(nn.BatchNorm1d(e_p[1]))

            if idx != (len(e_p_list) - 1) and num_fc_layers == 0 and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        if flatten_bool or num_fc_layers != 0:
            layer_list.append(Flatten())
            num_outputs = output_width * output_height * output_channels

        for idx in range(num_fc_layers):

            layer_list.append(nn.Linear(num_outputs, num_outputs))

            if idx == (num_fc_layers - 1) and output_activation_layer_bool == False:
                continue

            layer_list.append(nn.BatchNorm1d(num_outputs))
            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        self.model = nn.Sequential(*layer_list)

#### a 1D deconvolutional network
class DECONV1DN(Proto_Model):
    def __init__(self, model_name, input_size, output_size, output_activation_layer_bool, device = None):
        super().__init__(model_name + "_1ddeconv")

        # assume output_chan is a multiple of 2
        # assume input height and width have only two prime factors 2 and 3 for now
        # assume that output height and output width are factors of input height and input width respectively
        # activation type leaky relu and network uses batch normalization
        self.device = device

        self.input_size = input_size
        self.output_size = output_size

        self.output_activation_layer_bool = output_activation_layer_bool
        self.num_fc_layers = num_fc_layers

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_1Dconv_params(output_size, input_size) # encoder parameters
        e_p_list.reverse()

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.ConvTranspose1d(e_p[0] , e_p[1], kernel_size= e_p[2],\
                stride=e_p[3], padding=e_p[4], bias=True))

            if idx != (len(e_p_list) - 1):
                layer_list.append(nn.BatchNorm1d(e_p[1]))

            if idx != (len(e_p_list) - 1) and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        self.model = nn.Sequential(*layer_list)

    def forward(self, rep):
        assert rep.size()[1] == self.input_size[0], "input channel dim does not match network requirements"
        tiled_feat = rep.view(rep.size()[0], rep.size()[1], 1).expand(-1, -1, self.input_size[1])
        return self.model(tiled_feat)

#### a fully connected network
class FCN(Proto_Model):
    def __init__(self, model_name, input_channels, output_channels, num_layers, middle_channels_list = [],  device = None):
        super().__init__(model_name + "_fcn")

        #### activation layers: leaky relu
        #### no batch normalization 

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers

        self.middle_channels_list = middle_channels_list

        if len(self.middle_channels_list) == 0:
            mc_list_bool = False
        else:
            mc_list_bool = True

        # -----------------------
        # Fully connected network
        # -----------------------
        layer_list = []

        for idx in range(self.num_layers):

            if mc_list_bool == False:
                if idx == 0:
                    layer_list.append(nn.Linear(input_channels, output_channels))
                    layer_list.append(nn.BatchNorm1d(output_channels))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                elif idx == self.num_layers - 1:
                    layer_list.append(nn.Linear(output_channels, output_channels))
                else:
                    layer_list.append(nn.Linear(output_channels, output_channels))
                    layer_list.append(nn.BatchNorm1d(output_channels))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))

            else:
                if idx == 0:
                    layer_list.append(nn.Linear(input_channels, middle_channels_list[idx]))
                    layer_list.append(nn.BatchNorm1d(middle_channels_list[idx]))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                elif idx == self.num_layers - 1:
                    layer_list.append(nn.Linear(middle_channels_list[-1], output_channels))
                else:
                    layer_list.append(nn.Linear(middle_channels_list[idx - 1], middle_channels_list[idx]))
                    layer_list.append(nn.BatchNorm1d(middle_channels_list[idx]))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))

        self.model = nn.Sequential(*layer_list)
######################################
# Current Macromodel Types Supported
#####################################

# All macromodels have four main methods

# 1. init - initializes the network with the inputs requested by the user
# 2. forward - returns the output of the network for a given input
# 3. save - saves the model
# 4. load - loads a model if there is a nonempty path corresponding to that model in the yaml file

# Note the ProtoMacromodel below does not have a forward method, this must defined by the subclasses which 
# call the ProtoMacromodel as a superclass

#### super class for models of models for logging and loading models
class Proto_Macromodel(nn.Module):
    def __init__(self):
        super().__init__()   
        self.model_list = []

    def save(self, epoch_num):
        for model in self.model_list:
            model.save(epoch_num)

    def load(self, epoch_num):
        for model in self.model_list:
            model.load(epoch_num)

#### a long short-term memory network
class LSTM(Proto_Macromodel):
    def __init__(self, model_name, input_channels, output_channels, fg_list =[], ig_list = [], cg_list = [], og_list = [], device = None):
        super().__init__()

        self.device = device
        self.input_chan = input_channels
        self.output_chan = output_channels

        # -----------------------
        # Long Short-Term Memory Network
        # -----------------------
        if len(fg_list) == 0:
            num_layers = 3
        else:
            num_layers = len(fg_list)

        self.forget_gate_encoder = FCN(model_name + "_lstm_fg", input_channels + output_channels,\
         output_channels, num_layers, fg_list, device = device)

        if len(ig_list) == 0:
            num_layers = 3
        else:
            num_layers = len(ig_list)

        self.input_gate_encoder = FCN(model_name + "_lstm_ig", input_channels + output_channels,\
         output_channels, num_layers, ig_list, device = device)

        if len(og_list) == 0:
            num_layers = 3
        else:
            num_layers = len(og_list)

        self.output_gate_encoder = FCN(model_name + "_lstm_og", input_channels + output_channels,\
         output_channels, num_layers, og_list, device = device)

        if len(cg_list) == 0:
            num_layers = 3
        else:
            num_layers = len(cg_list)

        self.candidate_gate_encoder = FCN(model_name + "_lstm_cg", input_channels + output_channels,\
         output_channels, num_layers, cg_list, device = device)

        self.model_list.append(self.forget_gate_encoder)
        self.model_list.append(self.input_gate_encoder)
        self.model_list.append(self.output_gate_encoder )
        self.model_list.append(self.candidate_gate_encoder )

    def forward(self, x_t, h_prev = None, c_prev = None):

        if h_prev is not None:
            h_t = h_prev.clone()
            c_t = c_prev.clone()
        else:
            h_t = torch.zeros(x_t.size(0), self.output_chan, dtype=torch.float).to(self.device)
            c_t = torch.zeros(x_t.size(0), self.output_chan, dtype=torch.float).to(self.device)

        input_tensor = torch.cat([h_t, x_t], 1)

        f_t = torch.sigmoid(self.forget_gate_encoder(input_tensor))
        i_t = torch.sigmoid(self.input_gate_encoder(input_tensor))
        c_est_t = torch.tanh(self.candidate_gate_encoder(input_tensor))
        o_t = torch.sigmoid(self.output_gate_encoder(input_tensor))

        c_t = f_t * c_t + i_t * c_est_t

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

#########################################
# Params class for learning specific parameters
#########################################

#### a set of parameters that can be optimized, not a mapping

# All params have three main methods

# 1. init - initializes the network with the inputs requested by the user
# 3. save - saves the model
# 4. load - loads a model if there is a nonempty path corresponding to that model in the yaml file
class Params(object):
    def __init__(self, model_name, size, device, init_values = None):
        self.device = device

        if init_values == None:
            self.parameters = torch.ones(size).to(self.device).requires_grad_(True)

        else:
            self.parameters = init_values.clone().to(self.device).requires_grad_(True)

        self.model_name = model_name + "_params"

        self.load()

    def save(self, epoch_num):
        ckpt_path = '{}_{}.{}'.format(self.model_name, epoch_num, "pt")
        torch.save(self.parameters, ckpt_path)
        self.model_logger.save(ckpt_path)
        print("Saved Model to: ", ckpt_path)

    def load(self, epoch_num):
        ckpt_path = '{}_{}.{}'.format(self.model_name, epoch_num, "pt")
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)


