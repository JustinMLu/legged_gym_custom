# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # Optimizer (Actor, Critic, Privileged Encoder)
        self.optimizer = optim.Adam([
            {'params': actor_critic.actor.parameters()},
            {'params': actor_critic.critic.parameters()},
            {'params': actor_critic.privileged_encoder_.parameters()}
        ], lr=learning_rate)

        # Optimizer (Adaptation Encoder)
        self.adaptation_optimizer = optim.Adam(self.actor_critic.adaptation_encoder_.parameters(), lr=learning_rate)

        # Transition data object
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.total_updates = 0.0 # number of total updates - either update() or update_dagger()

    def init_storage(self, num_envs, num_transitions_per_env, total_obs_shape, privileged_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, total_obs_shape, privileged_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, critic_obs, adaptation_mode=False):
        """ No longer specifies critic_obs as a parameter - now uses privileged_obs
        """
        
        # Compute actions from obs & privileged obs
        self.transition.actions = self.actor_critic.act(obs, privileged_obs, adaptation_mode).detach()

        # Compute values from critic obs
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # Get action probabilities
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Store observation data in Transition object
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.critic_observations = critic_obs

        # Return actions
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()    # Forward pass thru critic network
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_regularization_loss = 0 

           
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # Loop over minibatches
        for obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                
                # Call ActorCritic act() method
                self.actor_critic.act(obs_batch, privileged_obs_batch, adaptation_mode=False)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # Privileged encoder update
                privileged_latent_batch = self.actor_critic.privileged_encoder(privileged_obs_batch)
                with torch.inference_mode():
                    adaptation_latent_batch = self.actor_critic.adaptation_encoder(obs_batch)

                regularization_loss = (privileged_latent_batch - adaptation_latent_batch.detach()).norm(p=2, dim=1).mean()

                # ================= Regularization coefficient schedule =================
                start_val, end_val, start_step, duration = 0.0, 0.1, 3000, 7000           # Define schedule parameters
                stage = min(max((self.total_updates - start_step) / duration, 0.0), 1.0)  # Calculate stage (0 to 1)
                regularization_coef = start_val + stage * (end_val - start_val)           # Interpolate coefficient
                # =======================================================================

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # # DEBUG PRINTS
                # print(f"Returns batch mean: {returns_batch.mean().item()}")
                # print(f"Value batch mean: {value_batch.mean().item()}")
                # print(f"Value loss: {value_loss.item()}")

                # Compute loss - privileged encoder should mimic adaptation encoder
                loss = surrogate_loss \
                     + self.value_loss_coef * value_loss \
                     - self.entropy_coef * entropy_batch.mean() \
                     + regularization_coef * regularization_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update mean loss metrics (numerator)
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_regularization_loss += regularization_loss.item()

        # Update mean loss metrics (denominator)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_regularization_loss /= num_updates
        self.storage.clear()
        self.increase_update_count()

        return mean_value_loss, mean_surrogate_loss, mean_regularization_loss

    def increase_update_count(self):
        """ Increase the counter that tracks the total number of updates 
        """
        self.total_updates += 1

    def update_dagger(self):
        """ Update only the adaptation encoder to mimic the privileged encoder 
        """
        mean_adaptation_loss = 0

        # Get minibatch generator
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # Loop through batches
        for obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch, _, _, _, _, _, _, _, _ in generator:
            # Act
            with torch.inference_mode():
                self.actor_critic.act(obs_batch, privileged_obs_batch, adaptation_mode=True)

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.privileged_encoder(privileged_obs_batch)
            

            # Get adaptation latent (will accumulate gradients)
            adapt_latent_batch = self.actor_critic.adaptation_encoder(obs_batch)
            
            # Compute loss - adaptation encoder should mimic privileged encoder
            adaptation_loss = (priv_latent_batch.detach() - adapt_latent_batch).norm(p=2, dim=1).mean()
            
            # Update only the adaptation encoder
            self.adaptation_optimizer.zero_grad()
            adaptation_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.adaptation_encoder_.parameters(), self.max_grad_norm)
            self.adaptation_optimizer.step()
            
            mean_adaptation_loss += adaptation_loss.item()
        
        # Calculate average loss
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_adaptation_loss /= num_updates
        self.storage.clear()
        self.increase_update_count()
        
        return mean_adaptation_loss