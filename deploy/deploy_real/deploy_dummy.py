from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

import mujoco.viewer
import mujoco

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient



class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)        # joint pos.
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)       # joint vel.
        self.actions = np.zeros(config.num_actions, dtype=np.float32)   # actions
        self.target_dof_pos = config.default_angles.copy()              # target joint positions    
        self.obs = np.zeros(config.num_obs, dtype=np.float32)           # observation
        self.cmd = np.array([0.0, 0.0, 0.0])                            # command (x, y, yaw)
        self.counter = 0


        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # From sdk2py
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
            

        # wait for the subscriber to receive data
        self.wait_for_low_state()
        
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("===== MOVING TO DEFAULT POSITION... =====")
        # move time 2s
        total_time = 4
        num_step = int(total_time / self.config.control_dt) # 0.02 for go2 (0.005 sim dt * 4 ctrl decimation)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_pos = self.config.default_angles # already np array
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j] # Kp obtained from cfg
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j] # Kd obtained from cfg
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt) # should be correct

    def default_pos_state(self):
        print("default_pos_state() invoked...")
        print("Waiting for the Button A signal...")
        
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]

                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1

        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq


        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # Prepare observation quantities
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        
        # Get projected gravity
        projected_gravity = get_gravity_orientation(quat)

        # Prepare phase features (*MATCH*)
        real_time_s = self.counter * self.config.control_dt # VERIFY THIS IS SECONDS!!!

        phase = (real_time_s % self.config.period) / self.config.period
        phase_fr = (phase + self.config.fr_offset) % 1
        phase_bl = (phase + self.config.bl_offset) % 1
        phase_fl = (phase + self.config.fl_offset) % 1
        phase_br = (phase + self.config.br_offset) % 1

        sin_phase_fl = np.sin(2 * np.pi * phase_fl)
        cos_phase_fl = np.cos(2 * np.pi * phase_fl)
        sin_phase_fr = np.sin(2 * np.pi * phase_fr)
        cos_phase_fr = np.cos(2 * np.pi * phase_fr)
        sin_phase_bl = np.sin(2 * np.pi * phase_bl)
        cos_phase_bl = np.cos(2 * np.pi * phase_bl)
        sin_phase_br = np.sin(2 * np.pi * phase_br)
        cos_phase_br = np.cos(2 * np.pi * phase_br)

        # Construct phase features vector (*MATCH*)
        phase_features = np.array([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dtype=np.float32)

        # Get commands from remote controller
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # Construct observation tensor
        num_actions = self.config.num_actions

        self.obs[:3] = ang_vel * self.config.ang_vel_scale
        self.obs[3:6] = projected_gravity
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd # scale to controller
        self.obs[9 : 9+num_actions] = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale 
        self.obs[9+num_actions : 9+2*num_actions] = dqj_obs * self.config.dof_vel_scale
        self.obs[9+2*num_actions : 9+3*num_actions] = self.actions
        self.obs[9+3*num_actions:9+3*num_actions+8] = phase_features

        # Convert to tensor
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)

        # Get actions from policy network
        self.actions = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.actions * self.config.action_scale + self.config.default_angles

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Send command
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface", default="eno1")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2_real.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    # DEBUG

    while True:
        try:
            controller.run() # UNCOMMENT LATER
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

# IPV4: 192.168.123.222
# Netmask: 255.255.255.0
# Ping the Go2 at: 192.168.123.161