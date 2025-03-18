from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            # Paths
            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            
            # Protocols
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            # No idea what this is (holdout from G1)
            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            # Timing
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]
            self.control_dt = self.simulation_dt * self.control_decimation

            # Motor-related
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)


            # Scales
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            # Clipping
            self.clip_obs = config["clip_observations"]
            self.clip_actions = config["clip_actions"]
            
            # Actions & Observations
            self.num_actions = config["num_actions"]
            self.num_proprio = config["num_proprio"]
            self.enable_history = config["enable_history"]
            self.buffer_length = config["buffer_length"]
            self.num_obs = self.num_proprio+(self.num_proprio*self.buffer_length) if self.enable_history else self.num_proprio
            self.rc_scale = np.array(config["rc_scale"], dtype=np.float32)

            # Phase
            self.period = config["period"]
            self.fr_offset = config["fr_offset"]
            self.bl_offset = config["bl_offset"]
            self.fl_offset = config["fl_offset"]
            self.br_offset = config["br_offset"]