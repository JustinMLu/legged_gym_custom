from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import torch
import numpy as np
import os


def quaternion_to_euler(quat_angle):
        """
        Converts a quaternion into euler angles (roll, pitch, yaw).
        Roll: rotation around x in radians (counterclockwise)
        Pitch: rotation around y in radians (counterclockwise)
        Yaw: rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


class Go2Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.debug_viz = True
        

    def _create_envs(self):
        super()._create_envs()
        
        # Get hip link indices
        hip_names = [s for s in self.body_names if "hip" in s]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

        # Get thigh link indices
        thigh_names = [s for s in self.body_names if "thigh" in s]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])

        # Get calf link indices
        calf_names = [s for s in self.body_names if "calf" in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], calf_names[i])

        # Get hip joint indices
        hip_joint_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        self.hip_joint_indices = torch.zeros(len(hip_joint_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_joint_names):
            self.hip_joint_indices[i] = self.dof_names.index(name)

        # Get thigh joint indices
        thigh_joint_names = ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]
        self.thigh_joint_indices = torch.zeros(len(thigh_joint_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_joint_names):
            self.thigh_joint_indices[i] = self.dof_names.index(name)

        # Get calf joint indices
        calf_joint_names = ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
        self.calf_joint_indices = torch.zeros(len(calf_joint_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_joint_names):
            self.calf_joint_indices[i] = self.dof_names.index(name)


    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the max. tracking reward, increase cmd range
        mean_lin_vel_reward = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
        delta = self.cfg.commands.vel_increment
        
        # Linear velocity (x-axis)
        if mean_lin_vel_reward > 0.8 * self.reward_scales["tracking_lin_vel"]:

            # Increase lower range
            if self.cfg.commands.max_reverse_vel < 0.0:

                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - delta, 
                                                              self.cfg.commands.max_reverse_vel, # This is a negative number
                                                              0.)
            else: # Positive reverse velocity case (usually for sprinty guys)
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - delta, 
                                                              self.cfg.commands.max_reverse_vel, 
                                                              self.command_ranges["lin_vel_x"][0] - delta
                                                              )
            # Increase upper range
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + delta, 
                                                          0., 
                                                          self.cfg.commands.max_forward_vel)

            
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        """
        # Build noise vector
        noise_vec = torch.zeros(self.cfg.env.num_proprio, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # ==========================================================================
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel    # angular vel (3)
        noise_vec[3:5] = noise_scales.imu * noise_level                                 # roll, pitch (2)
        noise_vec[5:8] = 0.                                                             # commanded vel (3)
        noise_vec[8:9] = 0.                                                             # jump vel. commands (1)
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # dof pos (12)
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dof vel (12)
        noise_vec[33:45] = 0.                                                           # prev. actions (12)
        noise_vec[45:53] = 0.                                                           # phase obs (8)
        # ==========================================================================
        return noise_vec


    def _init_custom_buffers(self):
        """ Initialize some custom buffers that are special to my Go2 implementation.
        """
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
        # Jump flags - not used in obs anymore
        self.jump_flags = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _init_buffers(self):
        """ Initializes the buffers used to store the simulation state and observational data.
            Overloaded to also initialize some custom buffers.
        """
        super()._init_buffers()
        self._init_custom_buffers()

    def check_termination(self):
        """ Check if environments need to be reset
        """

        # Terminate if certain links are contacted
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        
        # Terminate timed out robots
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        # Terminate robots that flipped over
        self.upside_down_buf = self.projected_gravity[:, 2] > 0. # past 90 degrees
        self.reset_buf |= self.upside_down_buf

        # If parkour enabled, terminate robots that fell down a hole
        if self.cfg.terrain.parkour:
            self.fell_into_hole_buf = self.root_states[:, 2] < -1.0
            self.reset_buf |= self.fell_into_hole_buf


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids)
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # update terrain curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0. 
        self.obs_history_buf[env_ids, :, :] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = 0.
        self.last_contact_heights[env_ids] = 0.
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["min_command_x"] = self.command_ranges["lin_vel_x"][0]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
            
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def update_feet_states(self):
        """ Updates the positions and velocities of the feet.
            Also updates the period and phase of the gait based on commanded velocity.
            Gait phase stuff is put here so it updates every physics step.
        """
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
        mask = torch.norm(self.commands[:, :3], dim=1) < 0.2 # considers all 3 commands 
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


    def print_debug_info(self):
        """ Prints some debug information to the console.
        """
        # # Print base XYZ
        base_xyz = self.root_states[:, 0:3]
        print(f"x: {base_xyz[:, 0].item():.3f} m | y: , {base_xyz[:, 1].item():.3f}, m | z: {base_xyz[:, 2].item():.3f} m")

        # Print measured heights as grid
        num_x = len(self.cfg.terrain.measured_points_x)
        num_y = len(self.cfg.terrain.measured_points_y)
        reshaped = self.measured_heights.reshape(-1, num_x, num_y) # (num_envs, 12, 11)
        print(reshaped)


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        # self.print_debug_info()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Update feet states, pos, vel
        self.update_feet_states()

        # Constantly transform IMU quat to rpy
        self.roll, self.pitch, self.yaw = quaternion_to_euler(self.base_quat)

        # run post physics step callback
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)     # calls resample_commands ...
        self.compute_observations()

        # Update buffers that store 'previous' data
        self.last_actions[:] = self.actions[:]              # Update prev. actions
        self.last_dof_vel[:] = self.dof_vel[:]              # Update prev. dof velocity
        self.last_root_vel[:] = self.root_states[:, 7:13]   # Update prev. root velocity
        self.last_base_lin_vel[:] = self.base_lin_vel[:]    # Update prev. base linear velocity
        self.last_torques[:] = self.torques[:]              # Update prev. torques

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

        # Resample commands
        self._resample_commands(env_ids)

        # Apply heading command if enabled
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_error = wrap_to_pi(self.commands[:, 3] - heading) * self.cfg.commands.heading_error_gain
            self.commands[:, 2] = torch.clip(heading_error, -1., 1.)

        # Update measured heights buffer
        self.measured_heights = self._get_heights()

        # bully robots
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    

    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # if empty env_ids provided, do nothing
        if len(env_ids)==0:
            return

        # User override
        if len(self.cfg.commands.user_command) > 0:
            self.commands[env_ids, :] = torch.as_tensor(self.cfg.commands.user_command, device=self.device).unsqueeze(0)
            return
        
        # Resample linear velocity commands
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], 
                                                     self.command_ranges["lin_vel_x"][1], 
                                                     (len(env_ids), 1), 
                                                     device=self.device).squeeze(1)
        
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], 
                                                     self.command_ranges["lin_vel_y"][1], 
                                                     (len(env_ids), 1), 
                                                     device=self.device).squeeze(1)
        
        # Resample angular commands
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], 
                                                         self.command_ranges["heading"][1], 
                                                         (len(env_ids), 1), 
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], 
                                                         self.command_ranges["ang_vel_yaw"][1], 
                                                         (len(env_ids), 1), 
                                                         device=self.device).squeeze(1)
            
        # Set small XY linear velocity commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
   

        # Randomly zero out commands
        if self.cfg.commands.zero_command:
            zero_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.zero_command_prob
            idx = env_ids[zero_mask]        # envs to zero out
            self.commands[idx, 0:3] *= 0.0   # zero out vx, vy, w
            
            # "Zero out" heading by setting tgt heading to cur heading
            if self.cfg.commands.heading_command:
                forward = quat_apply(self.base_quat[idx], self.forward_vec[idx])
                cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[idx, 3] = cur_heading
                

    def compute_observations(self):
        """ Computes observations for the robot. Overloaded to include unique observations for Go2.
        """
        # Construct phase features
        sin_phase_fl = torch.sin(2 * np.pi * self.phase_fl); cos_phase_fl = torch.cos(2 * np.pi * self.phase_fl)
        sin_phase_fr = torch.sin(2 * np.pi * self.phase_fr); cos_phase_fr = torch.cos(2 * np.pi * self.phase_fr)
        sin_phase_bl = torch.sin(2 * np.pi * self.phase_bl); cos_phase_bl = torch.cos(2 * np.pi * self.phase_bl)
        sin_phase_br = torch.sin(2 * np.pi * self.phase_br); cos_phase_br = torch.cos(2 * np.pi * self.phase_br)

        phase_features = torch.stack([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dim=1)

        # Construct IMU obs
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)

        # Deal with parkour jump zone
        if self.cfg.terrain.parkour:

            min_outlier_threshold = 8
            height_threshold = 0.1

            num_outliers = torch.sum(torch.abs(self.measured_heights) > height_threshold, dim=1)
            jump_zone_mask = (num_outliers >= min_outlier_threshold)
            self.jump_flags = jump_zone_mask.unsqueeze(1).float()


            # # Get local robot and obstacle x coordinates
            # robot_x = (self.root_states[:, 0] - self.env_origins[:, 0]).unsqueeze(1)
            # obstacle_x = torch.tensor(self.cfg.terrain.obstacle_x_positions, device=self.device)

            # # Trigger jump flag
            # in_jump_zone = (robot_x >= (obstacle_x - 1.2)) & (robot_x <= obstacle_x + 0.2)
            # jump_zone_mask = in_jump_zone.any(dim=1)
            # jump_idx = torch.nonzero(jump_zone_mask, as_tuple=False).squeeze(1)
            # self.jump_flags = jump_zone_mask.unsqueeze(1).float()


            # =================================== DEBUG PRINT ===================================
            # if torch.any(self.jump_flags):                       # at least one robot in-zone
            #     in_zone_env_ids = torch.nonzero(self.jump_flags[:, 0], as_tuple=False).flatten()

            #     # NOTE: .item() requires moving the tensor to CPU first
            #     for env_id in in_zone_env_ids.cpu().tolist():
            #         print(f"[JUMP-DBG]  robot env {env_id:1d} inside jump zone ")
            # ===================================================================================

        # CUR OBS    
        cur_obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,                     # (3,)
                                imu_obs,                                                           # (2,)      
                                self.commands[:, :3] * self.commands_scale,                        # (3,)
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,   # (12,) for quadruped
                                self.dof_vel * self.obs_scales.dof_vel,                            # (12,)
                                self.actions,                                                      # (12,) last actions
                                ),dim=-1)                                                          
        
        # PHASE FEATURES
        cur_obs_buf = torch.cat([cur_obs_buf, phase_features], dim=1) # add 8
 
        # NOISE
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec
        
        # HISTORY OBS (concatenate)
        self.obs_buf = torch.cat([
            self.obs_history_buf.view(self.num_envs, -1),  # Flattened history
            cur_obs_buf                                    # Current observation
        ], dim=-1)

        # PRIVILEGED OBS
        self.privileged_obs_buf = torch.cat((self.privileged_mass_params,     # 4
                                             self.privileged_friction_coeffs, # 1
                                             self.kp_kd_multipliers[0] - 1,      # 12
                                             self.kp_kd_multipliers[1] - 1,      # 12
                                             ), dim=-1)
        
        # ESTIMATED OBS
        self.estimated_obs_buf = self.base_lin_vel * self.obs_scales.lin_vel
        
        # SCAN OBS
        self.scan_obs_buf = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
        
        # CRITIC OBS
        self.critic_obs_buf = torch.cat((
            self.obs_buf.clone().detach(),
            self.privileged_obs_buf.clone().detach(),
            self.estimated_obs_buf.clone().detach(),
            self.scan_obs_buf.clone().detach()
        ), dim=-1)
        
        # Update the history buffer   
        self.obs_history_buf = torch.where((
            self.episode_length_buf <= 1)[:, None, None],
            torch.stack([cur_obs_buf] * (self.cfg.env.history_buffer_length), dim=1),
            torch.cat([self.obs_history_buf[:, 1:], cur_obs_buf.unsqueeze(1)], dim=1)
        )


    # ==================================== EXTREME  PARKOUR ====================================
    def _reward_delta_torques(self):
        """ Penalize changes in torques
        """
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    

    def _reward_dof_error(self):
        """ Penalize DOF positions away from default
        """
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    

    def _reward_zero_cmd_dof_error(self):
        """ Penalize DOF positions away from default 
        """
        zero_mask = (torch.norm(self.commands[:, :3], dim=1) < 0.2).float()
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error * zero_mask
    

    def _reward_hip_pos(self):
        """ Penalize DOF hip positions away from default"""
        default_hip_pos = self.default_dof_pos[:, self.hip_joint_indices]
        cur_hip_pos = self.dof_pos[:, self.hip_joint_indices]
        return torch.sum(torch.square(cur_hip_pos - default_hip_pos), dim=1)
    

    def _reward_thigh_pos(self):
        """ Penalize DOF thigh positions away from default"""
        default_thigh_pos = self.default_dof_pos[:, self.thigh_joint_indices]
        cur_thigh_pos = self.dof_pos[:, self.thigh_joint_indices]
        return torch.sum(torch.square(cur_thigh_pos - default_thigh_pos), dim=1)
    

    def _reward_calf_pos(self):
        """ Penalize DOF calf positions away from default"""
        default_calf_pos = self.default_dof_pos[:, self.calf_joint_indices]
        cur_calf_pos = self.dof_pos[:, self.calf_joint_indices]
        return torch.sum(torch.square(cur_calf_pos - default_calf_pos), dim=1)
    

    # ================================== PHASE GAIT BOOTSTRAP ==================================
    def _reward_phase_contact_match(self):
        """ Reward proper foot contact based on gait phase.
            For a trot gait:
            - FR and BL feet should contact when phase < 0.5
            - FL and BR feet should contact when phase >= 0.5
        """
        
        # 1 is 100% on ground, 0 is 50% on ground, -1 is 0% on ground
        stance_threshold = 2.0 * self.cfg.rewards.percent_time_on_ground - 1.0

        # Check if feet are supposed to be in stance
        fl_stance = torch.sin(2*np.pi*self.phase_fl) <= stance_threshold
        fr_stance = torch.sin(2*np.pi*self.phase_fr) <= stance_threshold
        bl_stance = torch.sin(2*np.pi*self.phase_bl) <= stance_threshold
        br_stance = torch.sin(2*np.pi*self.phase_br) <= stance_threshold

        # Reward / penalty for each foot
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward += torch.where(~(self.fl_contact ^ fl_stance), 0.25, -0.0)
        reward += torch.where(~(self.fr_contact ^ fr_stance), 0.25, -0.0)
        reward += torch.where(~(self.bl_contact ^ bl_stance), 0.25, -0.0)
        reward += torch.where(~(self.br_contact ^ br_stance), 0.25, -0.0)

        return reward


    def _reward_phase_foot_lifting(self):
        """ Reward proper foot lifting based on gait phase.
            Uses same general logic as _reward_phase_contact_match()
        """
        # 1 is 100% on ground, 0 is 50% on ground, -1 is 0% on ground
        stance_threshold = 2.0 * self.cfg.rewards.percent_time_on_ground - 1.0

        # Check if feet are supposed to be in stance
        fl_stance = torch.sin(2*np.pi*self.phase_fl) <= stance_threshold
        fr_stance = torch.sin(2*np.pi*self.phase_fr) <= stance_threshold
        bl_stance = torch.sin(2*np.pi*self.phase_bl) <= stance_threshold
        br_stance = torch.sin(2*np.pi*self.phase_br) <= stance_threshold

        # Get foot heights
        foot_heights = torch.stack([
            self.feet_pos[:, 0, 2] - self.last_contact_heights[:, 0],  # FL
            self.feet_pos[:, 1, 2] - self.last_contact_heights[:, 1],  # FR
            self.feet_pos[:, 2, 2] - self.last_contact_heights[:, 2],  # BL
            self.feet_pos[:, 3, 2] - self.last_contact_heights[:, 3]   # BR
        ], dim=1)

        # Clamp foot heights
        foot_heights = torch.clamp(foot_heights, min=0.0, max=self.cfg.rewards.max_foot_height)
        
        # Normalize & apply swing mask
        swing_masks = torch.stack([~fl_stance, ~fr_stance, ~bl_stance, ~br_stance], dim=1)
        normalized_heights = (foot_heights / self.cfg.rewards.max_foot_height)
        height_rewards = torch.where(swing_masks, normalized_heights, -normalized_heights)
    
        # Compute reward, normalize by # of feet on ground
        reward = torch.sum(height_rewards, dim=1) / 2.0 
        return reward


    def _reward_stumble_calves(self):
        """ Penalize calves hitting vertical surfaces. Uses norm of the X and Y contact forces
        """
        return torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.calf_indices, 2]), dim=1)
    

    # ==================================== CHEETAH  REWARDS ====================================
    def _reward_calf_collision(self):
        """ Penalize calves making ANY contact with surfaces (not just vertical ones)
            Hopefully will prevent calf strikes. Does it work? You decide!
        """
        threshold = 0.1  # Contact force threshold
        return torch.sum(1.0*(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=-1) > threshold), dim=1)


    def _reward_tracking_pitch(self):
        """ Rewards close-to-target pitch angles of the robot base.
            Returns values from 0 to 1, where 1 means perfect pitch tracking.
        """
        pitch_deg = self.pitch * (180.0 / np.pi)
        pitch_error = torch.square(pitch_deg - self.cfg.rewards.pitch_deg_target)
        return torch.exp(-pitch_error / self.cfg.rewards.tracking_sigma)
        

    def _reward_tracking_roll(self):
        """ Rewards close-to-target roll angles of the robot base.
            Returns values from 0 to 1, where 1 means perfect roll tracking.
        """
        roll_deg = self.roll * (180.0 / np.pi)
        roll_error = torch.square(roll_deg - self.cfg.rewards.roll_deg_target)
        return torch.exp(-roll_error / self.cfg.rewards.tracking_sigma)
    
    
    # ========================================= JUMPER =========================================
    def _reward_thigh_symmetry(self):
        """ Penalize DOF left-right thigh differences. Ordering of 
            joint_indices buffer is [0] = FL, [1] = FR, [2] = RL, [3] = RR
        """
        left_thigh_dof_pos = self.dof_pos[:, self.thigh_joint_indices[[0, 2]]]  # FL and RL
        right_thigh_dof_pos = self.dof_pos[:, self.thigh_joint_indices[[1, 3]]]  # FR and RR
        return torch.sum(torch.abs(left_thigh_dof_pos - right_thigh_dof_pos), dim=1)


    def _reward_calf_symmetry(self):
        """ Penalize DOF left-right calf differences. Ordering of
            joint_indices buffer is [0] = FL, [1] = FR, [2] = RL, [3] = RR
        """
        left_calf_dof_pos = self.dof_pos[:, self.calf_joint_indices[[0, 2]]]
        right_calf_dof_pos = self.dof_pos[:, self.calf_joint_indices[[1, 3]]]
        return torch.sum(torch.abs(left_calf_dof_pos - right_calf_dof_pos), dim=1)


    def _reward_heading_alignment(self):
        """ Penalize the robot for deviating from a forward (0 rad) heading,
            computed from its base quaternion using the forward vector.
        """
        # Compute heading from the forward vec
        fwd = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(fwd[:, 1], fwd[:, 0])
        
        # Compute desired heading
        desired_heading = wrap_to_pi(self.commands[:, 3]) if self.cfg.commands.heading_command \
                  else torch.zeros_like(heading)

        # Wrap the heading to [-pi, pi]
        angle_error = wrap_to_pi(desired_heading - heading)

        # Debug prints that only work when we have 1 robot
        # print(f"Current heading: {heading.item():.3f}")
        # print(f"Desired heading: {desired_heading.item():.3f}")
        # print(f"Actual reward: {(-4 * torch.square(angle_error)).item():.3f}")
        return torch.square(angle_error)
    

    def _reward_fwd_jump_vel(self):
        """ LITERALLY JUST REWARDS FORWARD VELOCITY
        """

        world_lin_vel = self.root_states[:, 7:10]  # [vx, vy, vz]
        fwd_rew = torch.clamp(world_lin_vel[:, 0], min=0.0)

        jump_mask = (self.jump_flags[:, 0] > 0.0).float()
        not_zero_mask = (torch.norm(self.commands[:, :3], dim=1) >= 0.2).float()
        return fwd_rew * jump_mask * not_zero_mask
    

    def _reward_up_jump_vel(self):
        """ LITERALLY JUST REWARDS UPWARD VELOCITY
        """
        world_lin_vel = self.root_states[:, 7:10]  # [vx, vy, vz]
        up_rew = torch.clamp(world_lin_vel[:, 2], min=0.0)

        jump_mask = (self.jump_flags[:, 0] > 0.0).float()
        not_zero_mask = (torch.norm(self.commands[:, :3], dim=1) >= 0.2).float()
        return up_rew * jump_mask * not_zero_mask
    

    def _reward_reverse_penalty(self):
        """ Penalizes reverse velocity. Requires a NEGATIVE coefficient
        """
        world_lin_vel = self.root_states[:, 7:10] # [vx, vy, vz]
        reverse_vel = torch.clamp(world_lin_vel[:, 0], max=0.0) # [-inf, 0]
        return -reverse_vel # negative sign for coeff consistency only


    def _reward_jump_height(self):
        """ Reward base height above 0.42 z-value. This was empirically determined
            for the Go2. We CANNOT use self.measured_heights because of the huge
            gaps in the terrain when doing parkour.
        """
        extra = torch.clamp(self.root_states[:, 2] - 0.42, min=0.0)
    
        jump_mask = (self.jump_flags[:, 0] > 0.0).float()
        not_zero_mask = (torch.norm(self.commands[:, :3], dim=1) >= 0.2).float()
        return torch.square(extra) * jump_mask * not_zero_mask


    # Function that I moved into Go2.py, planning to move it back 
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