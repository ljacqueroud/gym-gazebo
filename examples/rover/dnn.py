import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    1D convolution along the time dimension

    input: B x N x C_in
    output: B x (N-k+1) x C_out

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

    def __init__(self, N_channels, N_output, N_steps):
        super().__init__()

        self.encoder = nn.Sequential(
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                ConvBlock(C_in = N_channels, C_out = N_channels),
                )

        self.decoder = nn.Sequential(
                LinBlock(N_in = N_channels*(N_steps-24), N_out = 128),
                LinBlock(N_in = 128, N_out = 64),
                LinBlock(N_in = 64, N_out = 32),
                nn.Linear(32, N_output),
                #nn.Softmax(),                   # try with and without softmax
                )

    def forward(self, x):
        x = self.encoder(x)     # encoder pass (convolutions)
        x = x.view(-1,          # concatenate output
        x = self.decoder(x)     # decoder pass (fully connected)
        return x
                
