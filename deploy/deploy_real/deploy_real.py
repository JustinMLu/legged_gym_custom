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

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

from deploy.deploy_base.deploy_base import BaseController, ConfigParser
from command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from remote_controller import RemoteController, KeyMap


class RobotController(BaseController):
    def __init__(self, cfg: ConfigParser) -> None:
        super().__init__(cfg)
        
        # Child-specific initializations
        self.counter = 0
        self.real_time_s = 0.0 # total accumulated real time
        self.remote_controller = RemoteController()
        
        # Low command stuff
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher_ = ChannelPublisher(cfg.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(cfg.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # Turn off LiDAR
        self.lidar_toggle_publisher_ = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_toggle_publisher_.Init()
        self.lidar_off_cmd = std_msgs_msg_dds__String_()
        self.lidar_off_cmd.data = "OFF"
        self.lidar_toggle_publisher_.Write(self.lidar_off_cmd)

        # Sport client mode handler
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
            
        # Wait for the subscriber to receive data
        self.wait_for_low_state()
        
        # Initialize low command
        init_cmd_go(self.low_cmd, weak_motor=self.cfg.weak_motor)

    def refresh_robot_states(self):
        """ Retrieve the latest robot state (joints, velocities, orientation, etc.) from 
            the environment and store it in this controller's internal buffers. 
        
            Should update the following data:
                - qj (joint pos.)
                - dqj (joint vel.)
                - ang_vel (in the local frame)
                - base_quat (base orientation quaternion)
        """
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        for i in range(len(self.cfg.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.cfg.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.cfg.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        self.base_quat = self.low_state.imu_state.quaternion
        self.ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

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
        move_time = 3.0 # seconds
        num_step = int(move_time / self.cfg.control_dt)
        
        print("Moving to default position...")
        print("move_time: ", move_time, "seconds")
        
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
            time.sleep(self.cfg.control_dt)

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
        """ Update the projected gravity separately. This is useful when the robot is in sleep mode. """
        quat = self.low_state.imu_state.quaternion
        self.projected_gravity = self.get_gravity_orientation(quat)

    def run(self):
        # Update time
        self.counter += 1
        self.real_time_s = self.counter * self.cfg.control_dt
        
        self.step(self.real_time_s)

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
    parser.add_argument("net", type=str, nargs='?', default="eno1", help="Specify the network interface ('eth0', 'eno1', etc.)")
    parser.add_argument("config_file", type=str, help="Specify the name of the YAML config file to use.")
    args = parser.parse_args()

    # Load config from path
    file_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/configs/{args.config_file}"
    
    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    # Initialize config parser and controller
    cfg = ConfigParser(file_path)
    controller = RobotController(cfg)

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

            # Enter sleep mode if key pressed
            if controller.remote_controller.button[KeyMap.down] == 1:
                sleep_mode = True

            # If sleep, damp and await reawakening
            if sleep_mode:
                create_damping_cmd(controller.low_cmd)
                controller.send_cmd(controller.low_cmd)
                time.sleep(controller.cfg.control_dt)
                
                controller.update_projected_gravity() # Keep updating IMU data

                if controller.remote_controller.button[KeyMap.up] == 1:
                    print("Exiting sleep mode...")
                    sleep_mode = False
                    controller.move_to_default_pos()
                    time.sleep(1.0) # Wait for the robot to stabilize

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