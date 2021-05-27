import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    1D convolution along the time dimension

    input: B x C_in x N
    output: B x C_out x (N-k+1)

    - B: batch size
    - N: number of time series contained
    - C: channels
    - k: kernel size
    """

    def __init__(self, C_in, C_out, k=5, dilation=1, stride=1):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv1d(C_in, C_out, k, dilation = dilation, stride = stride),
                nn.ReLU(),
                #nn.BatchNorm1d(num_features = C_out),
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
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(N_in, N_out),
                nn.Tanh(),                  # could use either Tanh or ReLU
                #nn.BatchNorm1d()
                )

    def forward(self, x):
        x = self.network(x)
        return x


class Model(nn.Module):

    def __init__(self, N_channels, N_output, N_steps, N_y):
        super().__init__()

        self.convs = nn.Sequential(
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels), 
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                )

        self.fc = nn.Sequential(
                LinBlock(N_in = (N_channels*(N_steps-24) + N_y), N_out = 128),
                LinBlock(N_in = 128, N_out = 64),
                LinBlock(N_in = 64, N_out = 32),
                nn.Linear(32, N_output),
                nn.Softmax(1),                   # try with and without softmax
                )

    def forward(self, x, y):
        # 1D convolutions
        x = self.convs(x)         # encoder pass (convolutions)

        # reshape x and concatenate with y
        x = x.view(x.shape[0], -1) 
        x = torch.cat((x,y), 1)

        # fully connected layers
        x = self.fc(x)         # decoder pass (fully connected)

        return x
                
