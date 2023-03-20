import torch.nn as nn
import torch
import numpy as np

class ConvNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, image_dims, conv_layers, fcn_layers):
        super(DQNConv, self).__init__()
        # create network layers from variable length list
        self.image_dims = image_dims
        self.conv = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels if i == 0 else conv_layers[i - 1], conv_layers[i], kernel_size=3, stride=1),
            nn.ReLU(),
        ) for i in range(len(conv_layers))])
        self.fc = nn.Sequential(*[nn.Sequential(
            nn.Linear(self._get_conv_out(in_channels, image_dims) if i == 0 else fcn_layers[i-1], fcn_layers[i]),
            nn.ReLU(),
        ) for i in range(len(fcn_layers))])
        self.fc.add_module('last', nn.Linear(fcn_layers[-1], out_channels))

    def _get_conv_out(self, in_channels, image_dims):
        o = self.conv(torch.zeros(1, in_channels, *image_dims))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc:
            x = layer(x)
        return x

class DQNMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128, 128]):
        super(DQNMLP, self).__init__()
        # create network layers from variable length list
        self.net = nn.Sequential(*[nn.Sequential(
            nn.Linear(state_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
            nn.ReLU(),
        ) for i in range(len(hidden_dims))])
        self.net.add_module('last', nn.Linear(hidden_dims[-1], action_dim))


    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class DQNConv(nn.Module):
    def __init__(self, in_channels, image_dims, action_dim, hidden_dims=[128, 128, 128], fc_dims=[128]):
        super(DQNConv, self).__init__()
        # create network layers from variable length list
        self.image_dims = image_dims
        self.conv = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels if i == 0 else hidden_dims[i - 1], hidden_dims[i], kernel_size=3, stride=1),
            nn.ReLU(),
        ) for i in range(len(hidden_dims))])
        self.fc = nn.Sequential(*[nn.Sequential(
            nn.Linear(self._get_conv_out(in_channels, image_dims) if i == 0 else fc_dims[i-1], fc_dims[i]),
            nn.ReLU(),
        ) for i in range(len(fc_dims))])
        self.fc.add_module('last', nn.Linear(fc_dims[-1], action_dim))

    def _get_conv_out(self, in_channels, image_dims):
        o = self.conv(torch.zeros(1, in_channels, *image_dims))
        return int(np.prod(o.size()))

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc:
            x = layer(x)
        return x

class PPOConv(nn.Module):
    def __init__(self, env, in_channels, out_channels, image_dims, conv_layers, fcn_layers):
        super(PPOConv, self).__init__()
        # create network layers from variable length list
        self.actor = ConvNetwork(in_channels, out_channels, image_dims, conv_layers, fcn_layers)
        self.critic = ConvNetwork(in_channels, 1, image_dims, conv_layers, fcn_layers)
    '''
    def train(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
    '''
            
