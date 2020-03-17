import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.distributions import Normal
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
### add dilation to your CNNs
#########################################
# Params class for learning specific parameters
#########################################

#### a set of parameters that can be optimized, not a mapping

# All params have three main methods

# 1. init - initializes the network with the inputs requested by the user
# 3. save - saves the model
# 4. load - loads a model
class Params(nn.Module):
    def __init__(self, save_name, load_name, size, device, init_values = None):
        super().__init__()
        self.device = device
        self.model = nn.Parameter(nn.init.uniform_(torch.empty(size)))
        self.save_name = save_name
        self.load_name = load_name
        self.size = size
        self.parallel = False

    def forward(self):
        return self.model

    def save(self, epoch_num):
        ckpt_path = '{}_{}.{}'.format(self.save_name, epoch_num, "pt")
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model, ckpt_path)

    def load(self, epoch_num):
        # pass
        ckpt_path = '{}_{}.{}'.format(self.load_name, epoch_num, "pt")
        self.model.data = torch.load(ckpt_path)
        print("Loaded Model to: ", ckpt_path)

    def set_parallel(self, parallel_bool):
        if parallel_bool:
            params = self.model.clone().data
            self.model =  nn.DataParallel(nn.Parameter(nn.init.uniform_(torch.empty(size))))
            self.model.data = params

        self.parallel = parallel_bool


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
    def __init__(self, save_name, load_name, device):
        super().__init__()
        self.model = None
        self.save_name = save_name
        self.load_name = load_name
        self.device = device
        self.parallel = False

    def forward(self, input):
        return self.model(input)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def save(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.save_name, epoch_num)
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, epoch_num):
        ckpt_path = '{}_{}'.format(self.load_name, epoch_num)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)

    def set_parallel(self, parallel_bool):
        if parallel_bool:
           self.model =  nn.DataParallel(self.model)

        self.parallel = parallel_bool

class PlanarFlow(Proto_Model):
    def __init__(self, save_name, load_name, channels = 20, num_layers = 16, device= None): #dim=20, K=16):
        super().__init__(save_name + "_plana_flow", load_name + "_plana_flow", device = device)

        self.device = device
        self.channels = channels
        self.num_layers = num_layers

        layer_list = []

        for idx in range(self.num_layers):
            layer_list.append(PlanarTransform(self.channels))

        self.model = nn.Sequential(*layer_list)

    # def forward(self, z, logdet=False):
    #     zK = z
    #     SLDJ = 0.
    #     for transform in self.transforms:
    #         out = transform(zK, logdet=logdet)
    #         if logdet:
    #             SLDJ += out[1]
    #             zK = out[0]
    #         else:
    #             zK = out
                
    #     if logdet:
    #         return zK, SLDJ
    #     return zK
#### a convolutional network
class CONV2DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, output_activation_layer_bool, flatten_bool, num_fc_layers, batchnorm_bool = True, dropout_bool = False, device = None):
        super().__init__(save_name + "_cnn", load_name + "_cnn", device = device)

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

        # print(e_p_list)

        layer_list = []

        for idx, e_p in enumerate(e_p_list):
            layer_list.append(nn.Conv2d(e_p[0] , e_p[1], kernel_size=(e_p[2], e_p[3]),\
                stride=(e_p[4], e_p[5]), padding=(e_p[6], e_p[7]), bias=True))

            if idx != (len(e_p_list) - 1) and batchnorm_bool:
                layer_list.append(nn.BatchNorm2d(e_p[1]))

            if idx != (len(e_p_list) - 1) and num_fc_layers == 0 and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))
            
            if dropout_bool:
                layer_list.append(nn.Dropout2d())

        if flatten_bool or num_fc_layers != 0:
            layer_list.append(Flatten())
            num_outputs = output_width * output_height * output_chan

        for idx in range(num_fc_layers):

            layer_list.append(nn.Linear(num_outputs, num_outputs))

            if idx == (num_fc_layers - 1) and output_activation_layer_bool == False:
                continue

            if batchnorm_bool:
                layer_list.append(nn.BatchNorm1d(num_outputs))

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

            if dropout_bool:
                layer_list.append(nn.Dropout())

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
    def __init__(self, save_name, load_name, input_size, output_size, output_activation_layer_bool, batchnorm_bool = True, dropout_bool = False, device = None):
        super().__init__(save_name + "_dcnn", load_name + "_dcnn", device = device)

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

            if idx != (len(e_p_list) - 1) and batchnorm_bool:
                layer_list.append(nn.BatchNorm2d(e_p[0]))

            if idx != (len(e_p_list) - 1) and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))
            if dropout_bool:
                layer_list.append(nn.Dropout2d())

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
    def __init__(self, save_name, load_name, input_size, output_size, output_activation_layer_bool, flatten_bool, num_fc_layers, batchnorm_bool = True, dropout_bool = False, device = None):
        super().__init__(save_name + "_1dconv", load_name + "_1dconv", device = device)

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
        output_chan, output_width = self.output_size

        #assume that the prime factorization of rows and cols is composed of only powers of 3 and 2
        e_p_list = get_1Dconv_params(input_size, output_size) # encoder parameters

        # print("EP List")

        # print(e_p_list)

        layer_list = []

        for idx, e_p in enumerate(e_p_list):

            layer_list.append(nn.Conv1d(e_p[0] , e_p[1], kernel_size= e_p[2],\
                stride=e_p[3], padding=e_p[4], bias=True))

            if idx != (len(e_p_list) - 1) and batchnorm_bool:
                layer_list.append(nn.BatchNorm1d(e_p[1]))

            if idx != (len(e_p_list) - 1) and num_fc_layers == 0 and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))
            if dropout_bool:
                layer_list.append(nn.Dropout())

        if flatten_bool or num_fc_layers != 0:
            layer_list.append(Flatten())
            num_outputs = output_width * output_chan

        for idx in range(num_fc_layers):

            layer_list.append(nn.Linear(num_outputs, num_outputs))

            if idx == (num_fc_layers - 1) and output_activation_layer_bool == False:
                continue

            if batchnorm_bool:
                layer_list.append(nn.BatchNorm1d(num_outputs))
            layer_list.append(nn.LeakyReLU(0.1, inplace = True))

            if dropout_bool:
                layer_list.append(nn.Dropout())                

        self.model = nn.Sequential(*layer_list)

#### a 1D deconvolutional network
class DECONV1DN(Proto_Model):
    def __init__(self, save_name, load_name, input_size, output_size, output_activation_layer_bool, batchnorm_bool = True, dropout_bool = False, device = None):
        super().__init__(save_name + "_1ddeconv", load_name + "_1ddeconv", device = device)

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

            if idx != (len(e_p_list) - 1) and batchnorm_bool:
                layer_list.append(nn.BatchNorm1d(e_p[1]))

            if idx != (len(e_p_list) - 1) and output_activation_layer_bool == False:
                continue         

            layer_list.append(nn.LeakyReLU(0.1, inplace = True))
            if dropout_bool:
                layer_list.append(nn.Dropout())

        self.model = nn.Sequential(*layer_list)

    def forward(self, rep):
        assert rep.size()[1] == self.input_size[0], "input channel dim does not match network requirements"
        tiled_feat = rep.view(rep.size()[0], rep.size()[1], 1).expand(-1, -1, self.input_size[1])
        return self.model(tiled_feat)

#### a fully connected network
class FCN(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, num_layers, middle_channels_list = [], batchnorm_bool = True, dropout_bool = False,  device = None):
        super().__init__(save_name + "_fcn", load_name + "_fcn", device = device)

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
                    if batchnorm_bool:
                        layer_list.append(nn.BatchNorm1d(output_channels))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                    if dropout_bool:
                        layer_list.append(nn.Dropout())
                elif idx == self.num_layers - 1:
                    layer_list.append(nn.Linear(output_channels, output_channels))
                else:
                    layer_list.append(nn.Linear(output_channels, output_channels))
                    if batchnorm_bool:
                        layer_list.append(nn.BatchNorm1d(output_channels))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                    if dropout_bool:
                        layer_list.append(nn.Dropout())

            else:
                if idx == 0:
                    layer_list.append(nn.Linear(input_channels, middle_channels_list[idx]))
                    if batchnorm_bool:
                        layer_list.append(nn.BatchNorm1d(middle_channels_list[idx]))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                    if dropout_bool:
                        layer_list.append(nn.Dropout())
                elif idx == self.num_layers - 1:
                    layer_list.append(nn.Linear(middle_channels_list[-1], output_channels))
                else:
                    layer_list.append(nn.Linear(middle_channels_list[idx - 1], middle_channels_list[idx]))
                    if batchnorm_bool:
                        layer_list.append(nn.BatchNorm1d(middle_channels_list[idx]))
                    layer_list.append(nn.LeakyReLU(0.1, inplace = True))
                    if dropout_bool:
                        layer_list.append(nn.Dropout())

        self.model = nn.Sequential(*layer_list)

### a basic recurrent neural network 
class RNNCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, nonlinearity = 'tanh',  device = None):
        super().__init__(save_name + "_rnn", load_name + "_rnn", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers
        # -----------------------
        # Recurrent neural network
        # -----------------------
        self.model = nn.RNNCell(self.input_channels, self.output_channels, nonlinearity = nonlinearity)

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros((x.size(0), self.output_channels))

        return self.model(x, h)

### a gated recurrent neural network
class GRUCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, device = None):
        super().__init__(save_name + "_gru", load_name + "_gru", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_layers = num_layers
        # -----------------------
        # Recurrent neural network
        # -----------------------
        self.model = nn.GRUCell(self.input_channels, self.output_channels)

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros((x.size(0), self.output_channels))
        return self.model(x, h)

### a long short term memory recurrent neural network
class LSTMCell(Proto_Model):
    def __init__(self, save_name, load_name, input_channels, output_channels, device = None):
        super().__init__(save_name + "_lstm", load_name + "_lstm", device = device)

        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        # -----------------------
        # Recurrent neural network
        # -----------------------

        self.model = nn.LSTMCell(self.input_channels, self.output_channels)

    def forward(self, x, h = None, c=None):
        if h is None or c is None:
            h = torch.zeros(x.size(0), self.output_channels).to(self.device)
            c = torch.zeros(x.size(0), self.output_channels).to(self.device)

        return self.model(x, (h, c))

# class Transformer(Proto_Model):
class Transformer(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_enc_layers, num_dec_layers, z_dim, device = None):
        super().__init__(save_name + "_transformer", load_name + "_transformer", device = device)

        self.device = device
        self.input_size = input_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.z_dim = z_dim


        self.model = nn.Transformer(d_model=self.input_size, nhead=8, num_encoder_layers=self.num_enc_layers,\
         num_decoder_layers=self.num_dec_layers, dim_feedforward=self.z_dim,\
          dropout=0.1, activation='relu')

    def forward(self, source, targ):
        return self.model(source, targ).transpose(0,1)

class Transformer_Encoder(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_layers, norm = None, nhead = 8, dim_feedforward = 2048, dropout = 0.1, activation = 'relu', device = None):
        super().__init__(save_name + "_trans_encoder", load_name + "_trans_encoder", device = device)
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.input_size, self.nhead, dim_feedforward = self.dim_feedforward, dropout = self.dropout,\
            activation = self.activation), num_layers = self.num_layers, norm = self.norm)

class Transformer_Decoder(Proto_Model):
    def __init__(self, save_name, load_name, input_size, num_layers, norm = None, nhead = 8, dim_feedforward = 128, dropout = 0.1, activation = 'relu', device = None):
        super().__init__(save_name + "_trans_decoder", load_name + "_trans_decoder", device = device)
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.model = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.input_size, self.nhead, dim_feedforward = self.dim_feedforward, dropout = self.dropout,\
            activation = self.activation), num_layers = self.num_layers, norm = self.norm)

    def forward(self, tgt_seq, src_seq, mem_padding_mask = None, tgt_padding_mask = None):
        # print("Padding mask size: ", mem_padding_mask.size())
        # print("input size: ", tgt_seq.size())
        if mem_padding_mask is None and tgt_padding_mask is None:
            return self.model(tgt_seq, src_seq)
        elif mem_padding_mask is None :
            return self.model(tgt_seq, src_seq, tgt_key_padding_mask = tgt_padding_mask)
        elif tgt_padding_mask is None:
            return self.model(tgt_seq, src_seq, memory_key_padding_mask = mem_padding_mask)
        else:
            return self.model(tgt_seq, src_seq, memory_key_padding_mask = mem_padding_mask, tgt_key_padding_mask = tgt_padding_mask)
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
        self.parallel = False

    def save(self, epoch_num):
        for model in self.model_list:
            model.save(epoch_num)

    def load(self, epoch_num):
        for model in self.model_list:
            model.load(epoch_num)

    def parameters(self):
        parameters = []
        for model in self.model_list:
            parameters += list(model.parameters())
        return parameters

    def eval(self):
        for model in self.model_list:
            model.eval()

    def set_parallel(self, parallel_bool):
        if parallel_bool:
            for model in self.model_list:
                model.set_parallel(parallel_bool)

        self.parallel = parallel_bool

class ResNetFCN(Proto_Macromodel):
    def __init__(self, save_name, load_name, input_channels, output_channels, num_layers, device = None):
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.save_bool = True
        self.output_channels = output_channels

        # print("Input channels", self.input_channels)
        # print("Output_channels", self.output_channels)

        self.num_layers = num_layers
        self.model_list = []

        for idx in range(self.num_layers):
            if idx == self.num_layers - 1:
                self.model_list.append(FCN(save_name + "_layer_" + str(idx + 1), load_name + "_layer_" + str(idx + 1), self.input_channels, self.output_channels, 2, batchnorm_bool = False, dropout_bool = True, device = self.device).to(self.device))
            else:
                self.model_list.append(FCN(save_name + "_layer_" + str(idx + 1), load_name + "_layer_" + str(idx + 1), self.input_channels, self.input_channels, 2, batchnorm_bool = False, dropout_bool = True, device = self.device).to(self.device))

    def forward(self, x):
        for idx, model in enumerate(self.model_list):
            if idx == 0:
                output = model(x) + x
                residual = output.clone()

            elif idx == len(self.model_list) - 1:
                output = model(output)
            else:
                output = model(output) + residual
                residual = output.clone()
                
        return output



def get_2Dconv_params(input_size, output_size):

    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size

    # if output_chan == 128 or output_chan == 256:
    #     output_2 = output_chan
    # else:
    #     output_2 = 128
    # # assume output_chan is a multiple of 2
    # assume input height and width have only two prime factors 2 and 3 for now
    # assume that output height and output width are factors of input height and input width respectively
    chan_list = [input_chan, 8]

    while chan_list[-1] != output_chan:
        # print(chan_list[-1])
        # print("Target", output_2)
        chan_list.append(chan_list[-1] * 2)

    # if output_chan != 128 and output_chan != 256:
    #     chan_list.append(output_chan)

    prime_fact_rows = get_prime_fact(input_height // output_height)
    prime_fact_cols = get_prime_fact(input_width // output_width)

    # print(prime_fact_rows)

    # print(prime_fact_cols)
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
        if prime_fact_rows[idx] == 7:
            e_p[2,idx] = 9
            e_p[4,idx] = 7
        elif prime_fact_rows[idx] == 5:
            e_p[2,idx] = 7
            e_p[4,idx] = 5
        elif prime_fact_rows[idx] == 3:
            e_p[2,idx] = 5
            e_p[4,idx] = 3
        elif prime_fact_rows[idx] == 2:
            e_p[2,idx] = 4
            e_p[4,idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[4,idx] = 1
        # fourth row col kernel
        # sixth row col stride
        if prime_fact_cols[idx] == 7:
            e_p[3,idx] = 9
            e_p[5,idx] = 7
        elif prime_fact_cols[idx] == 5:
            e_p[3,idx] = 7
            e_p[5,idx] = 5
        elif prime_fact_cols[idx] == 3:
            e_p[3,idx] = 5
            e_p[5,idx] = 3
        elif prime_fact_cols[idx] == 2:
            e_p[3,idx] = 4
            e_p[5,idx] = 2
        else:
            e_p[3,idx] = 3
            e_p[5,idx] = 1

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

    # print("Input channel size: ", input_chan)
    # print("Output channel size: ", output_chan)

    # assume output_chan is a multiple of 2
    # assume input height have only two prime factors 2 and 3 for now
    # assume that output width are factors of input width
    chan_list = [input_chan, 8]

    while chan_list[-1] != output_chan:
        chan_list.append(chan_list[-1] * 2)

    prime_fact = get_prime_fact(input_width // output_width)

    # print(prime_fact)
    # print(chan_list)

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

    # print(prime_fact)
    # print(chan_list)


    e_p = np.zeros((5,len(prime_fact))).astype(np.int16) #encoder parameters

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact)):

        # print(prime_fact[idx])

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact[idx] == 7:
            e_p[2,idx] = 9
            e_p[3, idx] = 7
        elif prime_fact[idx] == 5:
            e_p[2, idx] = 7
            e_p[3,idx] = 5
        elif prime_fact[idx] == 3:
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

        if temp_num % 7 == 0:
            temp_num = temp_num / 7
            prime_fact_list.append(7)
        elif temp_num % 5 == 0:
            temp_num = temp_num / 5
            prime_fact_list.append(5)
        if temp_num % 3 == 0:
            temp_num = temp_num / 3
            prime_fact_list.append(3)
        elif temp_num % 2 == 0:
            temp_num = temp_num / 2
            prime_fact_list.append(2)

    prime_fact_list.sort()
    prime_fact_list.reverse()

    return prime_fact_list 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PlanarTransform(nn.Module):
    def __init__(self, dim=20):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.randn(()) * 0.01)
    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))
    def h(self, x):
        return torch.tanh(x)
    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2
    def forward(self, z, logdet=False):
        # z.size() = batch x dim
        u_dot_w = (self.u @ self.w.t()).view(())
        w_hat = self.w / torch.norm(self.w, p=2) # Unit vector in the direction of w
        u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u # 1 x dim
        affine = z @ self.w.t() + self.b
        z_next = z + u_hat * self.h(affine) # batch x dim
        if logdet:
            psi = self.h_prime(affine) * self.w # batch x dim
            LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8) # batch x 1
            return z_next, LDJ
        return z_next

