import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ScanEncoder(nn.Module):
    def __init__(self, num_scan_obs, output_dim, hidden_dims=[128, 64, 32], activation='elu'):
        """ Initialize a ScanEncoder instance.
        
            Args:
                num_scan_obs: Number of scan observations
                output_dim: Dimensionality of the output features
                hidden_dims: List of hidden layer sizes
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        
        super().__init__()
        self.input_dim = num_scan_obs
        self.output_dim = output_dim
        activation = get_activation(activation)

        fc_layers = []
        fc_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        fc_layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                fc_layers.append(nn.Linear(hidden_dims[l], output_dim)) # last layer
            else:
                fc_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                fc_layers.append(activation)
        
        self.scan_encoder = nn.Sequential(*fc_layers)


class LinearVelocityEstimator(nn.Module):
    def __init__(self, num_base_obs, history_buffer_length, output_dim, hidden_dims=[128, 64], activation="elu"):
        """ Initialize a LinearVelocityEstimator instance.

            Args:
                num_base_obs: Size of an individual observation (WITHOUT history stuff)
                history_buffer_length: Length of the history buffer
                output_dim: Dimensionality of the output features
                hidden_dims: List of hidden layer sizes
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        super().__init__()
        self.input_dim = num_base_obs + (num_base_obs*history_buffer_length)
        self.output_dim = output_dim
        activation = get_activation(activation)

        fc_layers = []
        fc_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        fc_layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                fc_layers.append(nn.Linear(hidden_dims[l], output_dim)) # last layer
            else:
                fc_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                fc_layers.append(activation)
        
        self.estimator = nn.Sequential(*fc_layers)
    
    def forward(self, input):
        return self.estimator(input)


class PrivilegedEncoder(nn.Module):

    def __init__(self, num_privileged_obs, encoder_hidden_dims=[64, 20], output_dim=20, activation='elu'):
        """ Initialize a PrivilegedEncoder instance.
        
            Args:
                num_privileged_obs: Number of privileged observations
                output_dim: Dimensionality of the output features
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        
        super().__init__()
        self.activation = get_activation(activation)
        self.num_privileged = num_privileged_obs
        self.output_dim = output_dim
        self.encoder_hidden_dims = encoder_hidden_dims

        fc_layers = []
        fc_layers.append(nn.Linear(num_privileged_obs, encoder_hidden_dims[0]))
        fc_layers.append(self.activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                fc_layers.append(nn.Linear(encoder_hidden_dims[l], output_dim)) # Last layer
            else:
                fc_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                fc_layers.append(self.activation)
        self.priv_encoder = nn.Sequential(*fc_layers)

    def forward(self, privileged_obs):
        """ Forward pass through the privileged obs. encoder.
        """
        return self.priv_encoder(privileged_obs)    


class AdaptationEncoder(nn.Module):

    def __init__(self, num_base_obs, history_buffer_length, output_dim=20, activation='elu'):
        """ Initialize an AdaptationEncoder instance.
        
            Args:
                num_base_obs: Size of an individual observation (WITHOUT history stuff)
                history_buffer_length: Length of the history buffer
                output_dim: Dimensionality of the output features
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        
        super().__init__()
        self.activation = get_activation(activation)
        self.history_buffer_length = history_buffer_length
        self.num_base_obs = num_base_obs
        self.output_dim = output_dim
        channel_size = 10
        
        # 1: linear layer that encodes each observation 
        self.fc_encoder = nn.Sequential(nn.Linear(num_base_obs, 3*channel_size), self.activation)
        
        # 2: convolutional layers that process the history buffer
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels = 3*channel_size, 
                        out_channels = 2*channel_size, 
                        kernel_size = 4, 
                        stride = 2), 
                        self.activation,
            nn.Conv1d(in_channels = 2*channel_size, 
                        out_channels = channel_size, 
                        kernel_size = 2, 
                        stride = 1), 
                        self.activation,
            nn.Flatten())
        
        # 3: final linear layer that maps to the output size
        self.fc_final = nn.Sequential(nn.Linear(3*channel_size, output_dim), self.activation)
    
    def forward(self, obs_history):
        """ Forward pass through the adaptation encoder. Expects (un-flattened) observation history
            of dimension (batch_size, history_buffer_length, num_base_obs).
        """
        projected_obs = self.fc_encoder(obs_history)
        output = self.conv_layers(projected_obs.permute(0, 2, 1)) # permute to (batch_size, channels, seq_len)
        output = self.fc_final(output)
        return output


def get_activation(act_name):
    """ Returns the specified activation function.

        Supported activations: 'elu', 'selu', 'relu', 'crelu', 'lrelu', 'tanh', 'sigmoid'.
        Note: 'crelu' is not implemented in this version, but is included for future use.
    """
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None