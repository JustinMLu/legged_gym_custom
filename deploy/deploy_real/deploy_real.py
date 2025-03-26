from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

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

from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_



class Controller:
    def __init__(self, config: Config) -> None:
        
        # Initialize config and controller objects
        self.cfg = config
        self.remote_controller = RemoteController()

        # Initialize policy network by loading it
        self.policy = torch.jit.load(config.policy_path)
        
        # Initialize essential buffers
        self.qj = np.zeros(config.num_actions, dtype=np.float32)        # joint pos.
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)       # joint vel.
        self.actions = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()   
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.obs_history = np.zeros((config.buffer_length, config.num_proprio), dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])
        
        # Timing
        self.counter = 0
        self.real_time_s = 0.0 # total accumulated real time
        self.first_step_ever = True
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.projected_gravity[2] = -1.0

        # Low command stuff
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # Turn off LiDAR
        self.lidar_toggle_publisher_ = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_toggle_publisher_.Init()
        self.lidar_off_cmd = std_msgs_msg_dds__String_()
        self.lidar_off_cmd.data = "OFF"
        self.lidar_toggle_publisher_.Write(self.lidar_off_cmd)

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
        
        # Initialize low command
        init_cmd_go(self.low_cmd, weak_motor=self.cfg.weak_motor)


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
            time.sleep(self.cfg.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)

    def move_to_default_pos(self):
        print("===== MOVING TO DEFAULT POSITION... =====")
        # move time 2s
        total_time = 5
        num_step = int(total_time / self.cfg.control_dt)
        
        dof_idx = self.cfg.leg_joint2motor_idx
        kps = self.cfg.kps
        kds = self.cfg.kds
        default_pos = self.cfg.default_angles # already np array
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
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt) # should be correct

    def default_pos_state(self):
        print("default_pos_state() invoked...")
        print("Waiting for the Button A signal...")
        
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.cfg.leg_joint2motor_idx)):
                motor_idx = self.cfg.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.cfg.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.cfg.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.cfg.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.cfg.control_dt)
    
    def update_projected_gravity(self):
        quat = self.low_state.imu_state.quaternion
        self.projected_gravity = get_gravity_orientation(quat)


    def run(self):
        self.counter += 1
        self.real_time_s = self.counter * self.cfg.control_dt

        # Get commands from remote controller
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        
        # Get the current joint position and velocity
        for i in range(len(self.cfg.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.cfg.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.cfg.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # Prepare observation quantities
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        
        # Get projected gravity
        self.projected_gravity = get_gravity_orientation(quat)

        # Calculate gait period
        period = 0.66
       
        # Prepare phase features
        phase = (self.real_time_s % period) / period
        phase_fr = (phase + self.cfg.fr_offset) % 1
        phase_bl = (phase + self.cfg.bl_offset) % 1
        phase_fl = (phase + self.cfg.fl_offset) % 1
        phase_br = (phase + self.cfg.br_offset) % 1

        # Zero out all of them if small command
        cmd_norm = np.linalg.norm(self.cmd[:3])
        if cmd_norm < 0.15:
            phase_fr *= 0.0
            phase_bl *= 0.0
            phase_fl *= 0.0
            phase_br *= 0.0

        sin_phase_fl = np.sin(2 * np.pi * phase_fl)
        cos_phase_fl = np.cos(2 * np.pi * phase_fl)
        sin_phase_fr = np.sin(2 * np.pi * phase_fr)
        cos_phase_fr = np.cos(2 * np.pi * phase_fr)
        sin_phase_bl = np.sin(2 * np.pi * phase_bl)
        cos_phase_bl = np.cos(2 * np.pi * phase_bl)
        sin_phase_br = np.sin(2 * np.pi * phase_br)
        cos_phase_br = np.cos(2 * np.pi * phase_br)

        # Construct phase features 
        phase_features = np.array([
            sin_phase_fr, cos_phase_fr, 
            sin_phase_fl, cos_phase_fl,
            sin_phase_bl, cos_phase_bl,
            sin_phase_br, cos_phase_br
        ], dtype=np.float32)

        # Create observation list
        num_actions = self.cfg.num_actions
        cur_obs = np.zeros(self.cfg.num_proprio, dtype=np.float32)
        cur_obs[:3] = ang_vel * self.cfg.ang_vel_scale
        cur_obs[3:6] = self.projected_gravity
        cur_obs[6:9] = self.cmd * self.cfg.cmd_scale * self.cfg.rc_scale # controller
        cur_obs[9 : 9+num_actions] = (qj_obs - self.cfg.default_angles) * self.cfg.dof_pos_scale 
        cur_obs[9+num_actions : 9+2*num_actions] = dqj_obs * self.cfg.dof_vel_scale
        cur_obs[9+2*num_actions : 9+3*num_actions] = self.actions
        cur_obs[9+3*num_actions:9+3*num_actions+8] = phase_features

        # Concatenate obs history if enabled
        if self.cfg.enable_history:
            self.obs[:] = np.concatenate([self.obs_history.flatten(), cur_obs])
            # Then, add current observation to history
            if self.first_step_ever:
                self.first_step_ever = False
                self.obs_history = np.tile(cur_obs, (self.cfg.buffer_length, 1))  # (4x1, 1x53)
            else:
                self.obs_history = np.roll(self.obs_history, -1, axis=0)
                self.obs_history[-1] = cur_obs
        else:
            self.obs[:] = cur_obs

        # Convert to tensor, clip
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = torch.clip(obs_tensor, -self.cfg.clip_obs, self.cfg.clip_obs)

        # Get actions from policy network, clip
        self.actions = self.policy(obs_tensor)
        self.actions = torch.clip(self.actions, 
                                  -self.cfg.clip_actions, 
                                  self.cfg.clip_actions).detach().numpy().squeeze()
        
        # Update target dof positions
        self.target_dof_pos = self.actions * self.cfg.action_scale + self.cfg.default_angles


        # Build low cmd
        for i in range(len(self.cfg.leg_joint2motor_idx)):
            motor_idx = self.cfg.leg_joint2motor_idx[i]
            
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.cfg.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.cfg.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Send command
        self.send_cmd(self.low_cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface", default="eno1")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    args = parser.parse_args()

    # Load config from path
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/unified_configs/{args.config}"
    cfg_object = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(cfg_object)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    print("Running...")
    sleep_mode = False
    while True:
        try:

            # Enter sleep mode if robot flipped over
            if controller.projected_gravity[2] > 0.:
                print("Warning: Robot is upside down!")
                sleep_mode = True

            # Enter sleep mode if down on Dpad is pressed
            if controller.remote_controller.button[KeyMap.down] == 1:
                sleep_mode = True

            # If sleep mode, send damping command and await reawakening
            if sleep_mode:
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                time.sleep(0.01)
                
                controller.update_projected_gravity()
                if controller.remote_controller.button[KeyMap.up] == 1:
                    sleep_mode = False
                    print("Exiting sleep mode...")

            else:
                # Start time of current run step
                step_start = time.perf_counter()

                # Run the controller
                controller.run()

                # Timekeep
                time_elapsed_during_step = time.perf_counter() - step_start
                time_until_next_step = controller.cfg.control_dt - time_elapsed_during_step
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


            # Press the select key break out of the loop
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

# ssh unitree@192.168.123.18