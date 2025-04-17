# actor_critic.py
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.support_networks import PrivilegedEncoder, AdaptationEncoder

class ActorCritic(nn.Module):
    """ Actor-Critic architecture for continuous control reinforcement learning.
        This class subclasses nn.Module and implements the actor and critic networks.
    """

    is_recurrent = False
    def __init__(self,  num_proprio,                        # NEW (renamed)       
                        num_privileged_obs,                 # NEW
                        num_critic_obs,
                        num_estimated_obs,
                        num_scan_obs,                     
                        num_actions,                          
                        history_buffer_length,              # NEW
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        latent_encoder_output_dim=20,       # NEW
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        """ Initialize an ActorCritic instance.
        
            Args:
                num_proprio: The number of proprioceptive/base observations
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

        self.num_proprio = num_proprio
        self.num_privileged_obs = num_privileged_obs
        self.history_buffer_length = history_buffer_length
        self.num_critic_obs = num_critic_obs
        self.num_estimated_obs = num_estimated_obs
        self.num_scan_obs = num_scan_obs
        self.num_actions = num_actions

        print("\n============== DEBUG: ActorCritic ATTRIBUTES ==============")
        print(f"num_base_obs: {self.num_proprio}")
        print(f"num_privileged_obs: {self.num_privileged_obs}")
        print(f"num_critic_obs: {self.num_critic_obs}")
        print(f"num_estimated_obs: {self.num_estimated_obs}")
        print(f"num_scan_obs: {self.num_scan_obs}")
        print(f"history_buffer_length: {self.history_buffer_length}")
        print(f"num_actions: {self.num_actions}")
        print("===========================================================\n")

        # Get specified activation function, set MLP input dimensions
        activation = get_activation(activation)

        # [cur. obs | obs. history | latent | estimated_obs ]
        mlp_input_dim_a = num_proprio + (num_proprio*history_buffer_length) + latent_encoder_output_dim + num_estimated_obs + num_scan_obs
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

        # Build encoders (4096 x 9 x 45) --> (4096 x 9 x 30)
        self.adaptation_encoder_ = AdaptationEncoder(num_base_obs=self.num_proprio, 
                                                     history_buffer_length=self.history_buffer_length, 
                                                     output_dim=latent_encoder_output_dim, 
                                                     activation='elu')
        
        self.privileged_encoder_ = PrivilegedEncoder(num_privileged_obs=self.num_privileged_obs, # Linear velocity removed
                                                     encoder_hidden_dims=[64, 20], 
                                                     output_dim=latent_encoder_output_dim, 
                                                     activation='elu') 

        # Print the network architecture
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Adaptation Encoder: {self.adaptation_encoder_}")
        print(f"Privileged Encoder: {self.privileged_encoder_}")

        # Learnable noise std for action distribution
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) 
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

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
        """ Returns the mean of the action distribution.
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """ Returns the standard deviation of the action distribution.
        """
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """ Returns the entropy of the action distribution.
        """
        return self.distribution.entropy().sum(dim=-1)
    
    # ==================================================================================================
    def privileged_encoder(self, privileged_obs_buf):
        """ Forward pass through the privileged encoder.
            Input: Privileged observation tensor
            Returns: Latent vector encoding
        """
        return self.privileged_encoder_(privileged_obs_buf)
    
    def adaptation_encoder(self, obs_buf):
        """ Forward pass through the adaptation encoder.
            Input: full observation tensor - assumes history is at the back for now
            Returns: Latent vector encoding
        """
        hist = obs_buf[:, :-self.num_proprio] # Only the history part -> TODO: Change the CNN architecture to support current as well
        return self.adaptation_encoder_(hist.reshape(-1, self.history_buffer_length, self.num_proprio))
    
    def get_latent(self, obs_buf, privileged_obs_buf, adaptation_mode=False):
        """ Get latent vector using the appropriate encoder (adaptation or privileged).
        """
        if adaptation_mode:
            return self.adaptation_encoder(obs_buf)
        else:
            return self.privileged_encoder(privileged_obs_buf)
        
    def update_distribution(self, obs_buf, privileged_obs_buf, estimated_obs_buf, scan_obs_buf, adaptation_mode=False):
        """ Forward pass through actor network, updating action distribution.
        """
        latent = self.get_latent(obs_buf, privileged_obs_buf, adaptation_mode) 
        actor_input = torch.cat((obs_buf, latent, estimated_obs_buf), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_buf, privileged_obs_buf, estimated_obs_buf, scan_obs_buf, adaptation_mode=False):
        """ Returns an action sampled from the action distribution.
            Calls update_distribution() first.
        """
        self.update_distribution(obs_buf, privileged_obs_buf, estimated_obs_buf, scan_obs_buf, adaptation_mode)
        return self.distribution.sample()
    
    def act_inference(self, obs_buf, privileged_obs_buf, estimated_obs_buf, scan_obs_buf, adaptation_mode=False):
        """ Return action means for inference - does not sample from distribution.
            Does not call update_distribution().
        """
        
        latent = self.get_latent(obs_buf, privileged_obs_buf, adaptation_mode)
        actor_input = torch.cat((obs_buf, latent, estimated_obs_buf), dim=-1)
        return self.actor(actor_input)
    # ==================================================================================================

    def get_actions_log_prob(self, actions):
        """ Return the log probability of the given actions (under the current distribution).
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        """ Returns value of the critic network for the given observations.
        """
        value = self.critic(critic_observations)
        return value
    

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
