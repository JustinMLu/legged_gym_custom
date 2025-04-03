import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class PrivilegedEncoder(nn.Module):

    def __init__(self, num_privileged, output_dim, encoder_hidden_dims=[64, 20], activation='elu'):
        """ Initialize a PrivilegedEncoder instance.
        
            Args:
                num_privileged: Number of privileged observations
                output_dim: Dimensionality of the output features
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        
        super().__init__()
        self.activation = get_activation(activation)
        priv_layers = []
        priv_layers.append(nn.Linear(num_privileged, encoder_hidden_dims[0]))
        priv_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                priv_layers.append(nn.Linear(encoder_hidden_dims[l], output_dim)) # Last layer
            else:
                priv_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                priv_layers.append(activation)
        self.priv_encoder = nn.Sequential(*priv_layers)

    def forward(self, obs):
        """ Forward pass through the privileged obs. encoder.
        """
        return self.priv_encoder(obs)
    
    def __str__(self):
        """ Returns a string representation of the PrivilegedEncoder instance.
        """
        return f"PrivilegedEncoder(num_privileged={self.num_privileged}, output_dim={self.output_dim}, encoder_hidden_dims={self.encoder_hidden_dims}, activation={self.activation})"
                


class HistoryEncoder(nn.Module):

    def __init__(self, history_buffer_length, num_obs, output_dim, activation='elu'):
        """ Initialize a HistoryEncoder instance.
        
            Args:
                history_buffer_length: Length of the history buffer
                num_obs: Number of observations (non-privileged)
                output_dim: Dimensionality of the output features
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
        """
        
        super().__init__()
        self.activation = get_activation(activation)
        channel_size = 10

        # First part: linear layer that encodes each observation 
        self.fc_encoder = nn.Sequential(nn.Linear(num_obs, 3*channel_size), activation)
        
        # Second part: convolutional layers that process the history buffer
        if history_buffer_length == 10:
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
        
        elif history_buffer_length == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3*channel_size, 
                          out_channels = 2*channel_size, 
                          kernel_size = 6, 
                          stride = 2), 
                          self.activation,
                nn.Conv1d(in_channels = 2*channel_size, 
                          out_channels = channel_size, 
                          kernel_size = 4, 
                          stride = 2), 
                          self.activation,
                nn.Flatten())
        
        elif history_buffer_length == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3*channel_size, 
                          out_channels = 2*channel_size, 
                          kernel_size = 8, 
                          stride = 4), 
                          self.activation,
                nn.Conv1d(in_channels = 2*channel_size, 
                          out_channels = channel_size, 
                          kernel_size = 5, 
                          stride = 1), 
                          self.activation,
                nn.Conv1d(in_channels = channel_size, 
                          out_channels = channel_size, 
                          kernel_size = 5, 
                          stride = 1), 
                          self.activation, 
                nn.Flatten())
            
        else:
            raise ValueError("history buffer length must be 10, 20 or 50")
        
        # Third part: final linear layer that maps to the output size
        self.fc_final = nn.Sequential(nn.Linear(3*channel_size, output_dim), self.activation)
    
    def forward(self, obs):
        """ https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            Input: (*, H_in) where * means any number of dimensions including none and H_in = in_features
        """
        projected_obs = self.fc_encoder(obs)
        output = self.conv_layers(projected_obs.permute(0, 2, 1)) # permute to (batch_size, channels, seq_len)
        output = self.fc_final(output)
        return output

    def __str__(self):
        """ Returns a string representation of the HistoryEncoder instance.
        """
        return f"HistoryEncoder(history_buffer_length={self.history_buffer_length}, num_obs={self.num_obs}, output_dim={self.output_dim}, activation={self.activation})"


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