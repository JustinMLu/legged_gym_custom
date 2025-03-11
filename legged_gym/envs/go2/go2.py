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
        # Build noise vector with linear acceleration instead of linear velocity
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_accel * noise_level * self.obs_scales.lin_accel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        noise_vec[48:52] = 0.0 # phase observations
        
        # Add heightmap noise (if heightmap used)
        if self.cfg.terrain.measure_heights:
            noise_vec[52:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec


    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]


    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()


    def update_feet_states(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_states = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_states[:, :, :3]
        self.feet_vel = self.feet_states[:, :, 7:10]
    

    def _post_physics_step_callback(self):
        
        # Update feet states, positions, velociites
        self.update_feet_states()

        # TODO PUT ALL PHASE VARIABLES IN CONFIG!!!
        period = 0.66 # Complete cycle duration (seconds)
        fr_offset = 0.0
        bl_offset = 0.0
        fl_offset = 0.5
        br_offset = 0.5

        self.phase = (self.episode_length_buf * self.dt) % period / period 
        self.phase_fr = (self.phase + fr_offset) % 1
        self.phase_bl = (self.phase + bl_offset) % 1
        self.phase_fl = (self.phase + fl_offset) % 1
        self.phase_br = (self.phase + br_offset) % 1

        return super()._post_physics_step_callback()
    

    def compute_observations(self):
        """ Computes observations for the robot. Overloaded to use 
            linear acceleration instead of linear velocity as an observation.
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

        
        # Combine into phase features (using sin/cos provides continuity at cycle boundaries)
        # phase_features = torch.cat([sin_phase_fr, cos_phase_fr, sin_phase_fl, cos_phase_fl], dim=1)
        phase_features = torch.cat([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dim=1)

        # np.set_printoptions(precision=3)
        # print(phase_features[0])

        # Construct observations       
        base_obs = torch.cat((      self.base_ang_vel  * self.obs_scales.ang_vel,                       # (3,)
                                    self.projected_gravity,                                             # (3,)
                                    self.commands[:, :3] * self.commands_scale,                         # (3,)  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # (12,) for quadruped
                                    self.dof_vel * self.obs_scales.dof_vel,                             # (12,)
                                    self.actions                                                        # (12,) last actions
                                    ),dim=-1)                                                            # total: (45,)
        
        # Add phase info to observations
        self.obs_buf = torch.cat([base_obs, phase_features], dim=1) # total: (53,)

        # add perceptive inputs (height map) if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    # =========================== NEW REWARD FUNCTIONS ===========================
    def _reward_delta_torques(self): # Extreme Parkour -1.0e-7
        """ Penalize changes in torques
        """
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_hip_pos(self): # Extreme Parkour -0.5
        """ Penalize DOF hip positions away from default"""
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
    
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

        # "Duty factor" (% of each legs cycle on the ground) 
        # TODO PUT THIS IN CONFIG!!!
        stance_threshold = 0.5

        fl_stance = self.phase_fl < stance_threshold
        fr_stance = self.phase_fr < stance_threshold
        bl_stance = self.phase_bl < stance_threshold
        br_stance = self.phase_br < stance_threshold
        
        # Check actual foot contacts (measured from contact forces)
        # Threshold force (1.0) determines what counts as "contact"
        fl_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        fr_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        bl_contact = self.contact_forces[:, self.feet_indices[2], 2] > 1.0
        br_contact = self.contact_forces[:, self.feet_indices[3], 2] > 1.0
        
        # Reward matching contacts (true when contact matches expected stance)
        # Reward when both true or both false
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward += torch.where(~(fl_contact ^ fl_stance), 0.25, 0.0)
        reward += torch.where(~(fr_contact ^ fr_stance), 0.25, 0.0)
        reward += torch.where(~(bl_contact ^ bl_stance), 0.25, 0.0)
        reward += torch.where(~(br_contact ^ br_stance), 0.25, 0.0)

        return reward

    def _reward_foot_swing_height(self):
        """Reward appropriate foot height during swing phase.
        Works on any terrain by only considering feet not in contact.
        """
        # Target height for feet during swing
        target_height = 0.10
        
        # Detect feet in contact (use 3D force magnitude for more robust detection)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
        # Calculate error for feet not in contact (in swing phase)
        # The z-coordinate of the foot directly gives its height in world space
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_height) * ~contact
        
        # Return negative sum of errors
        return -torch.sum(pos_error, dim=1)
    
