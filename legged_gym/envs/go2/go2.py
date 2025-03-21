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
        
        # Find indices of hips, thighs, and calfs (using Go2's URDF joint naming convention)
        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # Build noise vector -must match obs structure before history
        # noise_vec = torch.zeros_like(self.obs_buf[0]) # NOTE: UNCOMMENTING THIS RESULTS IN 1060 vs. 53 MISMATCH IF HISTORY=TRUE
        noise_vec = torch.zeros(self.cfg.env.num_proprio, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands (FOUR FOR ISAAC)
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions (12)
        noise_vec[45:53] = 0. # phase observations (8)
        return noise_vec


    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
       
       # Init some feet-related buffers
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]    
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        # I could put these into a vector but I am doing it this way for easier debug
        self.fl_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fr_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bl_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.br_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # This will be updated by update_feet_states
        self.last_contact_heights = torch.zeros(self.num_envs, self.feet_num, device=self.device)


    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()


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
        # Calculate gait period
        cmd_norm = torch.norm(self.commands[:, :2], dim=1) # NORM DOESN'T CONSIDER ANGULAR YAW VELOCITY
        period = 1.0 / (1.0 + cmd_norm)          
        period = (period * 2.0) * 0.66           # Scale
        period = torch.clamp(period, 0.25, 1.0)  # Clamp result
        
        # Specify per-leg period offsets
        fr_offset = 0.0
        bl_offset = 0.0
        fl_offset = 0.5
        br_offset = 0.5

        # Calculate per-leg phase
        self.phase = (self.episode_length_buf * self.dt) % period / period 
        self.phase_fr = (self.phase + fr_offset) % 1
        self.phase_bl = (self.phase + bl_offset) % 1
        self.phase_fl = (self.phase + fl_offset) % 1
        self.phase_br = (self.phase + br_offset) % 1

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
        
        # TODO: Update last contact heights
        foot_height_update_mask = torch.stack([
            self.fl_contact, self.fr_contact, self.bl_contact, self.br_contact
        ], dim=1)

        foot_heights = torch.stack([
            self.feet_pos[:, 0, 2],  # FL foot height 
            self.feet_pos[:, 1, 2],  # FR foot height
            self.feet_pos[:, 2, 2],  # BL foot height
            self.feet_pos[:, 3, 2]   # BR foot height
        ], dim=1)

        self.last_contact_heights = torch.where(
        foot_height_update_mask,
        foot_heights,
        self.last_contact_heights
    )


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

        # Zero out if small command
        mask = torch.norm(self.commands[:, :2], dim=1) < 0.2 # (num_envs,)
        phase_features *= torch.where(mask.unsqueeze(1), 0.0, 1.0)

        
        # Construct observations       
        cur_obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,                      # (3,)
                                self.projected_gravity,                                             # (3,)
                                self.commands[:, :3] * self.commands_scale,                         # (3,)  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # (12,) for quadruped
                                self.dof_vel * self.obs_scales.dof_vel,                             # (12,)
                                self.actions                                                        # (12,) last actions
                                ),dim=-1)                                                           # total: (45,)
        
        # Add phase info to observations
        cur_obs_buf = torch.cat([cur_obs_buf, phase_features], dim=1) # total: (53,)

        # Add noise
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec

        # Update and use history buffer
        if self.cfg.env.enable_history:

            # Update self.obs_buf
            self.obs_buf = torch.cat([
                self.obs_history.view(self.num_envs, -1),  # Flattened history
                cur_obs_buf                                # Current observation
            ], dim=-1)

            # Update history buffer 
            self.obs_history = torch.where((
                self.episode_length_buf <= 1)[:, None, None],
                torch.stack([cur_obs_buf] * (self.cfg.env.buffer_length), dim=1),
                torch.cat([self.obs_history[:, 1:], cur_obs_buf.unsqueeze(1)], dim=1)
            )


    # =========================== NEW REWARD FUNCTIONS ===========================
    def _reward_delta_torques(self): # Extreme Parkour -1.0e-7
        """ Penalize changes in torques
        """
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_hip_pos(self): # Extreme Parkour -0.5
        """ Penalize DOF hip positions away from default"""
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
    
    def _reward_calf_pos(self):
        """ Penalize DOF calf positions away from default"""
        return torch.sum(torch.square(self.dof_pos[:, self.calf_indices] - self.default_dof_pos[:, self.calf_indices] - 0.3), dim=1)
    
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

            FOR NOW, DO NOT ENABLE BOTH THIS AND _REWARD_FEET_AIR_TIME SIMULTANEOUSLY!
            BOTH OF THEM USE self.last_contacts AND WILL OVERWRITE ONE ANOTHER'S STORED
            PREVIOUS CONTACT INFORMATION!
        """

        # Check if feet are supposed to be in stance
        fl_stance = torch.sin(2*np.pi*self.phase_fl) < 0.0 # i locked da duty factor 
        fr_stance = torch.sin(2*np.pi*self.phase_fr) < 0.0
        bl_stance = torch.sin(2*np.pi*self.phase_bl) < 0.0
        br_stance = torch.sin(2*np.pi*self.phase_br) < 0.0
        

        # Reward each foot for contact when it's meant to be stance
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward += torch.where(~(self.fl_contact ^ fl_stance), 0.25, 0.0)
        reward += torch.where(~(self.fr_contact ^ fr_stance), 0.25, 0.0)
        reward += torch.where(~(self.bl_contact ^ bl_stance), 0.25, 0.0)
        reward += torch.where(~(self.br_contact ^ br_stance), 0.25, 0.0)
        
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        reward = reward * (cmd_norm >= 0.1)
        return reward

    def _reward_phase_match_foot_lift(self):
        """Reward proper foot lifting based on gait phase.
            NOTE:
            I don't think using last_contact_heights is a good idea.
            On stairs, the feet will constantly be at different heights
            and even if I clip the deltas to a zero mininum, it means
            that sliding the feet on a slope is going to still incur a reward...

            What we should be doing is this: 
            1. drawing out a 'plane' that is like 0.15m below the robot's base 
            (or whatever the actual dist. between the base and feet are at 
            NEUTRAL STANCE).
             
            2. Every physics step we take the robot's orientation quaternion
            and rotate the plane to match the robot's orientation. Then, we
            calculate our deltas as self.feet_pos - plane. 
            
           OR:
           We know the that the the sin of each leg > 0.0 means the leg should
           be in the air. Then we calculate the derivative of the sine wave, then
           calculate the dz (or just velocity in the z-axis) of the foot.

           If the foot is properly following the phase, it should be lifting
           when the sin dt is positive and moving down when the dt is negative.

           So then, the reward can be...
        """
         # Sine wave foot height trajectories
        fl = torch.sin(2*np.pi*self.phase_fl) # less than 0.0 means stance
        fr = torch.sin(2*np.pi*self.phase_fr)
        bl = torch.sin(2*np.pi*self.phase_bl)
        br = torch.sin(2*np.pi*self.phase_br)

        # Reward (or penalize) each foot for height (TODO!!!)
        fl_delta = self.feet_pos[:, 0, 2] - self.last_contact_heights[:, 0]  # FL
        fr_delta = self.feet_pos[:, 1, 2] - self.last_contact_heights[:, 1]  # FR
        bl_delta = self.feet_pos[:, 2, 2] - self.last_contact_heights[:, 2]  # BL
        br_delta = self.feet_pos[:, 3, 2] - self.last_contact_heights[:, 3]  # BR




