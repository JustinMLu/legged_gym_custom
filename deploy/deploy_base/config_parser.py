from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml

class ConfigParser:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

            # ====================== REAL ONLY ======================
            self.msg_type = cfg["msg_type"]
            self.lowcmd_topic = cfg["lowcmd_topic"]
            self.lowstate_topic = cfg["lowstate_topic"]
            self.leg2motor_idx = cfg["leg_joint2motor_idx"]
            self.weak_motor = [] # Deprecated but required for lowcmd
            if "weak_motor" in cfg:
                self.weak_motor = cfg["weak_motor"]
    
            # ===================== SIM ONLY ========================
            self.xml_path = cfg["xml_path"].replace(
                "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
  

            # ===================== COMMON ==========================
            # Policy Path
            self.policy_path = cfg["policy_path"].replace(
                "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            
            # Timing
            self.simulation_dt = cfg["simulation_dt"]
            self.control_decimation = cfg["control_decimation"]
            self.control_dt = self.simulation_dt * self.control_decimation

            # Joint-related
            self.kps = cfg["kps"]
            self.kds = cfg["kds"]
            self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)

            # Scaling
            self.ang_vel_scale = cfg["ang_vel_scale"]
            self.dof_pos_scale = cfg["dof_pos_scale"]
            self.dof_vel_scale = cfg["dof_vel_scale"]
            self.action_scale = cfg["action_scale"]
            self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
            self.rc_scale = np.array(cfg["rc_scale"], dtype=np.float32)

            # Clipping
            self.clip_obs = cfg["clip_observations"]
            self.clip_actions = cfg["clip_actions"]

            # Actions & Observations
            self.num_actions = cfg["num_actions"]
            self.num_proprio = cfg["num_proprio"]
            self.enable_history = cfg["enable_history"]
            self.buffer_length = cfg["buffer_length"]
            self.num_obs = self.num_proprio+(self.num_proprio*self.buffer_length) if self.enable_history else self.num_proprio

            # Phase features
            self.period = cfg["period"]
            self.fr_offset = cfg["fr_offset"]
            self.bl_offset = cfg["bl_offset"]
            self.fl_offset = cfg["fl_offset"]
            self.br_offset = cfg["br_offset"]
            # =======================================================