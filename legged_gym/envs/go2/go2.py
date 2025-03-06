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
    

    def compute_observations(self):
        """ Computes observations for the robot. Overloaded to use 
            linear acceleration instead of linear velocity as an observation.
        """
        # ============ Linear Acceleration Ver. ================
        self.obs_buf = torch.cat((((self.base_lin_vel - self.last_base_lin_vel) / self.dt) * self.obs_scales.lin_accel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                       # (3,)
                                    self.projected_gravity,                                             # (3,)
                                    self.commands[:, :3] * self.commands_scale,                         # (3,)  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # (12,) for quadruped
                                    self.dof_vel * self.obs_scales.dof_vel,                             # (12,)
                                    self.actions                                                        # (12,) last actions
                                    ),dim=-1)                                                            # total: (48,)
        

        # add perceptive inputs (height map) if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


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
        
        # Add heightmap noise (if heightmap used)
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec


    # def _init_foot(self):
    #     self.feet_num = len(self.feet_indices)
    #     print("FEET INDICES: ", self.feet_indices)
    #     print("NUMBER OF FEET: ", self.feet_num)
    #     rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
    #     self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
    #     self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
    #     self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        
    #     self.feet_pos = self.feet_state[:, :, :3]
    #     print("FEET POSITIONS: ", self.feet_pos)
    #     self.feet_vel = self.feet_state[:, :, 7:10]
    #     print("FEET VELOCITIES: ", self.feet_vel)

    def _init_buffers(self):
        super()._init_buffers()
        # self._init_foot()

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