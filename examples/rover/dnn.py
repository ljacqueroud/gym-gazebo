import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    1D convolution along the time dimension layer

    input: B x C_in x N
    output: B x C_out x (N-k+1)

    - B: batch size
    - N: number of time series contained
    - C: channels
    - k: kernel size
    """

    def __init__(self, C_in, C_out, k=5, dilation=1, stride=1):
        super(ConvBlock, self).__init__()
        self.network = nn.Sequential(
                nn.Conv1d(C_in, C_out, k, dilation = dilation, stride = stride),
                nn.ReLU(),
                #nn.BatchNorm1d(num_features = C_out),
                #nn.Dropout(0.4)
                )

    def forward(self, x):
        x = self.network(x)
        return x


class LinBlock(nn.Module):
    """
    Linear fully connected layer

    input: B x N_in
    output: B x N_out

    - N: batch size
    - N_in: number of inputs
    - N_out: number of outputs
    """

    def __init__(self, N_in, N_out):
        super(LinBlock, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(N_in, N_out, bias=False),
                nn.ReLU(),                  # could use either Tanh or ReLU
                nn.LayerNorm(normalized_shape = N_out),
                #nn.Dropout(0.2),
                )

    def forward(self, x):
        x = self.network(x)
        return x


class Model(nn.Module):

    def __init__(self, N_channels, N_output, N_steps, N_y):
        super(Model, self).__init__()

        # convolutional layers
        self.convs = nn.Sequential(
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels), 
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                )

        # fully connected layers for TCN
        self.fc_tcn = nn.Sequential(
                LinBlock(N_in = (N_channels*(N_steps-24)), N_out = 128),
                LinBlock(N_in = 128, N_out = 64),
                LinBlock(N_in = 64, N_out = 32),
                nn.Linear(32, N_output, bias=False),
                nn.Softmax(1),                  
                )

        # fully connected layers for TCN with path information
        self.fc_tcn_path = nn.Sequential(
                LinBlock(N_in = (N_channels*(N_steps-24) + N_y), N_out = 128),
                LinBlock(N_in = 128, N_out = 64),
                LinBlock(N_in = 64, N_out = 32),
                nn.Linear(32, N_output, bias=False),
                nn.Softmax(1),                  
                )

        # fully connected layers for simple path follower network
        self.fc_path_follower = nn.Sequential(
                LinBlock(N_in = N_y, N_out = 32),
                LinBlock(N_in = 32, N_out = 32),
                LinBlock(N_in = 32, N_out = 32),
                LinBlock(N_in = 32, N_out = 16),
                LinBlock(N_in = 16, N_out = 8),
                nn.Linear(8, N_output),
                nn.Softmax(1),                 
                )

    def forward_tcn(self, x):
        """ forward for TCN network
        """
        # 1D convolutions
        x = self.convs(x)         

        # reshape x and concatenate with y
        x = x.view(x.shape[0], -1) 

        # fully connected layers
        x = self.fc_tcn(x)

        return x

    def forward_tcn_path(self, x, y):
        """ forward function for TCN with path information
        """
        # 1D convolutions
        x = self.convs(x)         

        # reshape x and concatenate with y
        x = x.view(x.shape[0], -1) 
        x = torch.cat((x,y), 1)

        # fully connected layers
        x = self.fc_tcn_path(x)

        return x

    def forward_fc_path_follower(self, y):
        """ forward function for fully connected network for path following
        """
        # fully connected layers 
        y = self.fc_path_follower(y)

        return y
                
