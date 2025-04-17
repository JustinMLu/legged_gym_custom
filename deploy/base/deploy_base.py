import numpy as np
import torch
from deploy.base.config_parser import ConfigParser

def quaternion_to_euler(quat_angle):
        """ Converts a quaternion into euler angles (roll, pitch, yaw).
            - Roll: rotation around x in radians (counterclockwise)
            - Pitch: rotation around y in radians (counterclockwise)
            - Yaw: rotation around z in radians (counterclockwise)
        """
        w = quat_angle[0]; x = quat_angle[1]; y = quat_angle[2]; z = quat_angle[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class BaseController:
    def __init__(self, cfg: ConfigParser) -> None:

        # Get config, load policy and encoder
        self.cfg = cfg
        self.policy = torch.jit.load(cfg.policy_path)
        self.encoder = torch.jit.load(cfg.encoder_path)
        self.estimator = torch.jit.load(cfg.estimator_path)

        # Initialize essential buffers
        self.qj = np.zeros(cfg.num_actions, dtype=np.float32)    # joint pos.
        self.dqj = np.zeros(cfg.num_actions, dtype=np.float32)   # joint vel.
        self.ang_vel = np.zeros(3, dtype=np.float32)             # angular vel.
        self.base_quat = np.zeros(4, dtype=np.float32)           # base orientation
        self.actions = np.zeros(cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = cfg.default_angles.copy()
        self.obs = np.zeros(cfg.num_obs, dtype=np.float32)
        self.obs_history = np.zeros((cfg.buffer_length, cfg.num_proprio), dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.first_step_ever = True

        # Initialize projected gravity (not really needed, immediately overwritten on first use)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.projected_gravity[2] = -1.0

        # Command smoothing - used for both controllers
        self.smoothed_cmd = np.zeros(3, dtype=np.float32)
        

    def get_gravity_orientation(self, quaternion):
        """ Returns gravity orientation in the frame specified by the quaternion.

        Args:
            quaternion (np.array): A quaternion representing the orientation of the
            base of the robot w.r.t. the world frame.

        Returns:
            np.array: A 3D vector representing the MuJoCo unit gravity vector ([0, 0, -1])
            projected onto the base frame.
        """
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation
    

    def get_smoothed_command(self, raw_cmd, smoothing_factor):
        """ Smooths the controller command by gradually blending the new commands with the previous.
            Lower values (0.01 - 0.05) give smoother but slower responses, while higher values (0.1 - 0.2)
            give faster but less smooth responses.
            Args:
                raw_cmd (np.array): The controller command [vx, vy, wz].
                smoothing_factor (float): The factor by which to smooth the command.
        """
        self.smoothed_cmd = self.smoothed_cmd + smoothing_factor * (raw_cmd - self.smoothed_cmd)
        return self.smoothed_cmd


    def refresh_robot_states(self):
        """ Retrieve the latest robot state (joints, velocities, orientation, etc.) from 
            the environment and store it in this controller's internal buffers. 
        
            Should update the following data:
                - qj (joint pos.)
                - dqj (joint vel.)
                - ang_vel (in the local frame)
                - base_quat (base orientation quaternion)
        """
        raise NotImplementedError("refresh_robot_states() not implemented")


    def step(self, elapsed_time_s):
        """ Execute one iteration of the control loop, performing the following steps:
            1) Refresh robot states
            2) Build observation
            3) Run policy for actions
            4) Update actuators/buffers

            Args:
                elapsed_time_s (float): The *real* time elapsed in seconds.
        """
        # Refresh robot states
        self.refresh_robot_states()
        self.projected_gravity = self.get_gravity_orientation(self.base_quat)
        self.roll, self.pitch, self.yaw = quaternion_to_euler(self.base_quat)
        
        # Calculate per-leg phase variables
        phase = (elapsed_time_s % self.cfg.period) / self.cfg.period
        phase_fr = (phase + self.cfg.fr_offset) % 1
        phase_bl = (phase + self.cfg.bl_offset) % 1
        phase_fl = (phase + self.cfg.fl_offset) % 1
        phase_br = (phase + self.cfg.br_offset) % 1

        # Zero out phase variables if small command
        cmd_norm = np.linalg.norm(self.cmd[:3])
        if cmd_norm < 0.2:
            phase_fr *= 0.0
            phase_bl *= 0.0
            phase_fl *= 0.0
            phase_br *= 0.0

        # Calculate sine and cosine of phases for smooth transitions
        sin_phase_fl = np.sin(2 * np.pi * phase_fl); cos_phase_fl = np.cos(2 * np.pi * phase_fl)
        sin_phase_fr = np.sin(2 * np.pi * phase_fr); cos_phase_fr = np.cos(2 * np.pi * phase_fr)
        sin_phase_bl = np.sin(2 * np.pi * phase_bl); cos_phase_bl = np.cos(2 * np.pi * phase_bl)
        sin_phase_br = np.sin(2 * np.pi * phase_br); cos_phase_br = np.cos(2 * np.pi * phase_br)

        # Construct phase features
        phase_features = np.array([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dtype=np.float32)
        
        # These just help legibility
        num_a = self.cfg.num_actions

        # IMU readings
        self.pitch += (self.cfg.pitch_offset) * (np.pi / 180) # Degree offset
        self.roll += (self.cfg.roll_offset) * (np.pi / 180) # Degree offset
        imu_obs = np.stack([self.roll, self.pitch])

        # Construct observation
        cur_obs = np.zeros(self.cfg.num_proprio, dtype=np.float32)
        cur_obs[:3] = self.ang_vel * self.cfg.ang_vel_scale
        cur_obs[3:5] = imu_obs
        cur_obs[5:8] = self.cmd * self.cfg.cmd_scale * self.cfg.rc_scale # controller
        cur_obs[8 : 8+num_a] = (self.qj - self.cfg.default_angles) * self.cfg.dof_pos_scale 
        cur_obs[8+num_a : 8+2*num_a] = self.dqj * self.cfg.dof_vel_scale
        cur_obs[8+2*num_a : 8+3*num_a] = self.actions
        cur_obs[8+3*num_a:8+3*num_a+8] = phase_features

        # Update observation buffer
        self.obs[:] = np.concatenate([self.obs_history.flatten(), cur_obs])
        
        # Update the history buffer
        if self.first_step_ever:
            self.first_step_ever = False
            self.obs_history = np.tile(cur_obs, (self.cfg.buffer_length, 1))
        else:
            self.obs_history = np.roll(self.obs_history, -1, axis=0)
            self.obs_history[-1] = cur_obs

        # Convert observations to tensor and clip
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = torch.clip(obs_tensor, -self.cfg.clip_obs, self.cfg.clip_obs)
        
        # Use adaptation module
        hist_len = self.cfg.buffer_length * self.cfg.num_proprio
        hist_tensor = obs_tensor[:, :hist_len].reshape(1, self.cfg.buffer_length, self.cfg.num_proprio)
        latent = self.encoder(hist_tensor)

        # Use estimator 
        estimated_vel = self.estimator(obs_tensor)

        # Concatenate
        actor_input = torch.cat((obs_tensor, latent, estimated_vel), dim=-1)

        # Get actions from policy network, clip
        self.actions = self.policy(actor_input)
        self.actions = torch.clip(self.actions, 
                                 -self.cfg.clip_actions, 
                                  self.cfg.clip_actions).detach().numpy().squeeze()
        
        # Update target dof pos
        self.target_dof_pos = self.actions * self.cfg.action_scale + self.cfg.default_angles