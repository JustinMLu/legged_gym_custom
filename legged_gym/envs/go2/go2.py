from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
import os

class Go2Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


    def _create_envs(self):
        super()._create_envs()
        
        # Get indices of hip, thigh, and calf links
        hip_names = [s for s in self.body_names if "hip" in s]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        thigh_names = [s for s in self.body_names if "thigh" in s]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])

        calf_names = [s for s in self.body_names if "calf" in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], calf_names[i])


        # Get indices of hip, thigh and calf joints
        hip_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_joint_indices = torch.zeros(len(hip_joint_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i, name in enumerate(hip_joint_names):
            self.hip_joint_indices[i] = self.dof_names.index(name)



    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # Build noise vector
        noise_vec = torch.zeros(self.cfg.env.num_proprio, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands (is this four or three??? I have no real idea)
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions (12)
        noise_vec[45:53] = 0. # phase observations (8)
        return noise_vec


    def _init_custom_buffers(self):
        # ==================================== FEET RELATED INITS ====================================
        # Get feet indices and rigid body states
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
       
       # Init some feet-related buffers
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]    
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        # Init phase related variables
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_fr = torch.zeros(self.num_envs, device=self.device)
        self.phase_fl = torch.zeros(self.num_envs, device=self.device)
        self.phase_bl = torch.zeros(self.num_envs, device=self.device)
        self.phase_br = torch.zeros(self.num_envs, device=self.device)

        # Stores the contact status of each foot (bools)
        self.fl_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fr_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bl_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.br_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Last contact status (MOVED FROM LEGGED_ROBOT.PY)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, 
                                                                                device=self.device, 
                                                                                requires_grad=False)

        # Last contact heights (MOVED FROM LEGGED_ROBOT.PY)
        self.last_contact_heights = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, 
                                                                                       device=self.device,)

        # Feet airtime buffer (MOVED FROM LEGGED_ROBOT.PY)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, 
                                                                                    device=self.device, 
                                                                                    requires_grad=False)
        # ============================================================================================


    def _init_buffers(self):
        """ Initializes the buffers used to store the simulation state and observational data.
            Overloaded to also initialize some custom buffers.
        """
        super()._init_buffers()
        self._init_custom_buffers()


    def reset_idx(self, env_ids):
        """ Reset the environment indices. Overloaded to reset some additional buffers. 
        """
        super().reset_idx(env_ids)
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = 0.
        self.last_contact_heights[env_ids] = 0.


    def update_feet_states(self):
        """ Updates the positions and velocities of the feet.
            Also updates the period and phase of the gait based on commanded velocity.
            Gait phase stuff is put here so it updates every physics step.
        """
        # DEBUG
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print("Base height: ", base_height)

        # Update feet states
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_states = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_states[:, :, :3]
        self.feet_vel = self.feet_states[:, :, 7:10]
        
        # =========================== GAIT CALCULATIONS ===========================
        # Calculate per-leg phase variables
        self.phase = (self.episode_length_buf * self.dt) % self.cfg.env.period / self.cfg.env.period 
        self.phase_fr = (self.phase + self.cfg.env.fr_offset) % 1
        self.phase_bl = (self.phase + self.cfg.env.bl_offset) % 1
        self.phase_fl = (self.phase + self.cfg.env.fl_offset) % 1
        self.phase_br = (self.phase + self.cfg.env.br_offset) % 1

        # Zero out phase variables if small command
        mask = torch.norm(self.commands[:, :3], dim=1) < 0.15 # considers all 3 commands 
        self.phase_fr *= torch.where(mask, 0.0, 1.0)
        self.phase_fl *= torch.where(mask, 0.0, 1.0)
        self.phase_bl *= torch.where(mask, 0.0, 1.0)
        self.phase_br *= torch.where(mask, 0.0, 1.0)

        # =========================== CONTACT DETECTION ===========================
        
        # Check if feet are in contact currently
        cur_fl = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        cur_fr = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        cur_bl = self.contact_forces[:, self.feet_indices[2], 2] > 1.0
        cur_br = self.contact_forces[:, self.feet_indices[3], 2] > 1.0

        # Update contact bools (also counteract buggy PhysX)
        self.fl_contact = torch.logical_or(cur_fl, self.last_contacts[:, 0])
        self.fr_contact = torch.logical_or(cur_fr, self.last_contacts[:, 1])
        self.bl_contact = torch.logical_or(cur_bl, self.last_contacts[:, 2])
        self.br_contact = torch.logical_or(cur_br, self.last_contacts[:, 3])

        # Update last contacts
        self.last_contacts = torch.stack([cur_fl, 
                                          cur_fr, 
                                          cur_bl, 
                                          cur_br], dim=1)
        
        # Update mask of which feet are contacting the ground
        foot_height_update_mask = torch.stack([
            self.fl_contact, self.fr_contact, self.bl_contact, self.br_contact
        ], dim=1)

        # Get current foot heights
        foot_heights = torch.stack([
            self.feet_pos[:, 0, 2],  # FL foot height 
            self.feet_pos[:, 1, 2],  # FR foot height
            self.feet_pos[:, 2, 2],  # BL foot height
            self.feet_pos[:, 3, 2]   # BR foot height
        ], dim=1)

        # Update last contact heights buffer using mask
        self.last_contact_heights = torch.where(foot_height_update_mask,
                                                foot_heights,
                                                self.last_contact_heights)


    def _post_physics_step_callback(self):
        # Update feet states, positions, velociites
        self.update_feet_states()
        return super()._post_physics_step_callback()
    

    def compute_observations(self):
        """ Computes observations for the robot. Overloaded to include unique observations for Go2.
        """
       
        # Calculate sine and cosine of phases for smooth transitions
        sin_phase_fl = torch.sin(2 * np.pi * self.phase_fl).unsqueeze(1) # FL
        cos_phase_fl = torch.cos(2 * np.pi * self.phase_fl).unsqueeze(1)
        sin_phase_fr = torch.sin(2 * np.pi * self.phase_fr).unsqueeze(1) # FR
        cos_phase_fr = torch.cos(2 * np.pi * self.phase_fr).unsqueeze(1)
        sin_phase_bl = torch.sin(2 * np.pi * self.phase_bl).unsqueeze(1) # BL
        cos_phase_bl = torch.cos(2 * np.pi * self.phase_bl).unsqueeze(1)
        sin_phase_br = torch.sin(2 * np.pi * self.phase_br).unsqueeze(1) # BR
        cos_phase_br = torch.cos(2 * np.pi * self.phase_br).unsqueeze(1)

        # Construct phase features
        phase_features = torch.cat([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dim=1)
       
        # Construct observations       
        cur_obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,                     # (3,)
                                self.projected_gravity,                                            # (3,)
                                self.commands[:, :3] * self.commands_scale,                        # (3,)  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,   # (12,) for quadruped
                                self.dof_vel * self.obs_scales.dof_vel,                            # (12,)
                                self.actions                                                       # (12,) last actions
                                ),dim=-1)                                                          # total: (45,)
        
        # Add phase features
        cur_obs_buf = torch.cat([cur_obs_buf, phase_features], dim=1) # total: (53,)

        # Add noise vector
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec
        
        # Update obs buffer (concat. history)
        self.obs_buf = torch.cat([
            self.obs_history_buf.view(self.num_envs, -1),  # Flattened history
            cur_obs_buf                                    # Current observation
        ], dim=-1)

        # EXTREME PARKOUR
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
        #     self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # Update privileged obs buffer
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                             self.privileged_mass_params,     # 4
                                             self.privileged_friction_coeffs, # 1
                                             self.motor_strength[0] - 1,      # 12
                                             self.motor_strength[1] - 1,      # 12
                                             ), dim=-1)
        
        # Update critic obs buffer
        self.critic_obs_buf = torch.cat((
            self.obs_buf.clone().detach(),
            self.privileged_obs_buf.clone().detach()
        ), dim=-1)
        
        
        # Update the history buffer   
        self.obs_history_buf = torch.where((
            self.episode_length_buf <= 1)[:, None, None],
            torch.stack([cur_obs_buf] * (self.cfg.env.history_buffer_length), dim=1),
            torch.cat([self.obs_history_buf[:, 1:], cur_obs_buf.unsqueeze(1)], dim=1)
        )

    # =========================== NEW REWARD FUNCTIONS ===========================
    def _reward_delta_torques(self): # Extreme Parkour -1.0e-7
        """ Penalize changes in torques
        """
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_hip_pos(self): # Extreme Parkour -0.5
        """ Penalize DOF hip positions away from default"""
        default_hip_pos = self.default_dof_pos[:, self.hip_joint_indices]
        cur_hip_pos = self.dof_pos[:, self.hip_joint_indices]
        return torch.sum(torch.square(cur_hip_pos - default_hip_pos), dim=1)
    
    
    def _reward_dof_error(self): # Extreme Parkour -0.04
        """ Penalize DOF positions away from default
        """
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_contact_phase_match(self):
        """ Reward proper foot contact based on gait phase.
            For a trot gait:
            - FR and BL feet should contact when phase < 0.5
            - FL and BR feet should contact when phase >= 0.5
        """
        percent_time_on_ground = 0.5 # TODO: Make this a config variable
        
        # 1 is 100% on ground, 0 is 50% on ground, -1 is 0% on ground
        stance_threshold = 2.0 * percent_time_on_ground - 1.0

        # Check if feet are supposed to be in stance
        fl_stance = torch.sin(2*np.pi*self.phase_fl) <= stance_threshold
        fr_stance = torch.sin(2*np.pi*self.phase_fr) <= stance_threshold
        bl_stance = torch.sin(2*np.pi*self.phase_bl) <= stance_threshold
        br_stance = torch.sin(2*np.pi*self.phase_br) <= stance_threshold
        

        # Reward each foot for contact when it's meant to be stance
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward += torch.where(~(self.fl_contact ^ fl_stance), 0.25, 0.0)
        reward += torch.where(~(self.fr_contact ^ fr_stance), 0.25, 0.0)
        reward += torch.where(~(self.bl_contact ^ bl_stance), 0.25, 0.0)
        reward += torch.where(~(self.br_contact ^ br_stance), 0.25, 0.0)
        
        return reward

    def _reward_stumble_feet(self): # Extreme Parkour -1.0
        """ Penalize feet hitting vertical surfaces. Uses norm of the X and Y contact forces
        """
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    

    def _reward_stumble_calves(self):
        """ Penalize calves hitting vertical surfaces. Uses norm of the X and Y contact forces
        """
        return torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.calf_indices, 2]), dim=1)


# ============================ REWARD FUNCTIONS THAT I DON'T USE lol ===========================
    def _reward_stand_still_v2(self):
        """ Reward maintaining pose with zero commands """
        small_command = torch.norm(self.commands[:, :3], dim=1) < 0.15
        
        # Reward joint positions matching default pose
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        
        # Reward minimal joint velocities
        vel_reward = torch.sum(torch.abs(self.dof_vel), dim=1)
        
        # Reward minimal base movement
        base_vel_reward = torch.sum(torch.abs(self.base_lin_vel), dim=1) + torch.sum(torch.abs(self.base_ang_vel), dim=1)
        
        # Only apply when commands are small
        return (dof_error + vel_reward + base_vel_reward) * small_command
    
    def _reward_feet_air_time(self):
        """ Reward long steps. Need to filter the contacts because 
            the contact reporting of PhysX is unreliable on meshes 
        """
        
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime