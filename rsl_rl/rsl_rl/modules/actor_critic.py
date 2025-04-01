import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCritic(nn.Module):
    """ Actor-Critic architecture for continuous control reinforcement learning.
        This class subclasses nn.Module and implements the actor and critic networks.
    """

    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        """ Initialize an ActorCritic instance.
        
            Args:
                num_actor_obs: Dimension of actor observation space
                num_critic_obs: Dimension of critic observation space (can differ from actor - e.g privileged info)
                num_actions: Dimension of action space
                actor_hidden_dims: List of hidden layer sizes for actor network
                critic_hidden_dims: List of hidden layer sizes for critic network
                activation: Activation function to use ('elu', 'relu', 'tanh', etc.)
                init_noise_std: Initial standard deviation for action distribution
                **kwargs: Additional arguments (will be ignored with a warning)
        """
        
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        # Call nn.Module constructor
        super(ActorCritic, self).__init__()

        # Get specified activation function, set MLP input dimensions
        activation = get_activation(activation)
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Build the actor network
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions)) # Last layer map to num_actions
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])) # Else use specified dims
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Build the critic network
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1)) # Last layer map to value output
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])) # Else use specified dims
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Print the network architecture
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))   # Add action std as learnable parameter
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    def init_weights(sequential, scales):
        """ To be honest with you I have no idea what this does.
            "not used at the moment" - ETH Zurich nerds
        """
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        """ Reset the actor and critic networks. Currently a no-op.
        """
        pass

    def forward(self):
        """ Forward method override - raises NotImplementedError
        """
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """ Return the mean of the action distribution.
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """ Return the standard deviation of the action distribution.
        """
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """ Compute the entropy of the action distribution.
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """ Forward pass through the actor network, updating the action distribution.
        """
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std) # Gaussian distribution with learnable std

    def act(self, observations, **kwargs):
        """ Forward pass, update action distribution, and sample an action.
        """
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """ Return the log probability of the given actions (under the current distribution).
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """ Forward pass through the actor network, without updating action distribution. 
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """ Forward pass through the critic network - returns value.
        """
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    """ Returns the specified activation function.
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
