import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import pdb
from deploy.base.deploy_base import BaseController, ConfigParser
from gamepad_reader import Gamepad
import re

class MujocoController(BaseController):

    def __init__(self, cfg: ConfigParser) -> None:
        super().__init__(cfg)

        self.gamepad = Gamepad(cfg.rc_scale[0], 
                               cfg.rc_scale[1], 
                               cfg.rc_scale[2])

        # Init Mujoco here so refresh_robot_states() can access mj_data
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.simulation_dt
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # ======================= Fake scan observation setup =======================
        # Stuff for fake obs
        self.scan_idx = 0
        self.phase_sync_point = -1
        self.fake_scan_obs = []
        self.mode = 'NORMAL'

        with open('SCAN_v12_ft_iii.txt') as f:
            text = f.read()

        # Split on blank lines (two or more newlines)
        blocks = re.split(r'\n\s*\n', text.strip())

        # Parse lists into fake_scan_obs
        for blk in blocks:
            content = blk.strip().lstrip('[').rstrip(']')
            nums = [float(x) for x in content.split()]
            self.fake_scan_obs.append(nums)
        print("Parsed fake scan observations of length: ", len(self.fake_scan_obs))

        # Get our sync point and REMOVE IT FROM THE LIST
        self.phase_sync_point = self.fake_scan_obs[0][0]
        self.fake_scan_obs = self.fake_scan_obs[1:]
        print("Phase sync point: ", self.phase_sync_point)
        # ===========================================================================


    def _get_scan_obs(self) -> torch.Tensor:
        """ Return (1, num_scan_obs) tensor for the scan-encoder.
        """
        # Always zero unless in REPLAY mode
        scan_tensor = torch.zeros((1, self.cfg.num_scan_obs), dtype=torch.float32)
        
        # Switch to WAITING mode if RB is pressed
        if self.gamepad._rb_pressed and self.mode == 'NORMAL':
            self.mode = 'WAITING'
        
        # Switch to REPLAY if WAITING and phases are synced
        if self.mode == 'WAITING' and np.abs(self.phase - self.phase_sync_point) < 0.005:
            self.mode = 'REPLAY'
            print("Replay mode activated")

        # During REPLAY, actually modify scan_tensor
        if self.mode == 'REPLAY':
            scan = self.fake_scan_obs[self.scan_idx]
            scan_tensor = torch.tensor(scan, dtype=torch.float32).view(1, -1)
            self.scan_idx += 1
            print("Scan idx: ", self.scan_idx)

            # If we reach the end of the replay buffer, switch back to NORMAL
            if self.scan_idx == len(self.fake_scan_obs) - 1:
                self.mode = 'NORMAL'
                print("Replay mode deactivated")
                self.scan_idx = 0
        
        return scan_tensor


    def _refresh_robot_states(self):
        """ Retrieve the latest robot state (joints, velocities, orientation, etc.) from 
            the environment and store it in this controller's internal buffers. 
        
            Should update the following data:
                - qj (joint pos.)
                - dqj (joint vel.)
                - ang_vel (in the local frame)
                - base_quat (base orientation quaternion)
        """

        # Input smoothed commands
        smoothed_cmd = self.get_smoothed_command([self.gamepad.vx, self.gamepad.vy, self.gamepad.wz], 0.15)
        self.cmd[0] = smoothed_cmd[0]
        self.cmd[1] = smoothed_cmd[1]
        self.cmd[2] = smoothed_cmd[2]
        
        # Refresh robot states
        self.qj = self.mj_data.qpos[7:]          # joint pos
        self.dqj = self.mj_data.qvel[6:]         # joint vel
        self.ang_vel = self.mj_data.qvel[3:6]    # angular vel (local frame)
        self.base_quat = self.mj_data.qpos[3:7]  # base rot. in quaternion


    def compute_torques(self, target_q, q, kp, target_dq, dq, kd):
        """ Calculates torques from position commands using position PD control.
        """
        return kp*(target_q-q) + kd*(target_dq-dq)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Specify the name of the YAML config file to use.")
    args = parser.parse_args()

    # Load config from path
    file_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/configs/{args.config_file}"

    # Initialize config parser and controller
    cfg = ConfigParser(file_path)
    controller = MujocoController(cfg)

    # Intialize time counters
    sim_time_s = 0.0
    decimation_counter = 0.0

    # Initialize Mujoco viewer
    viewer = mujoco.viewer.launch_passive(controller.mj_model, controller.mj_data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    while viewer.is_running():
        
        # Start time of current step
        step_start = time.perf_counter()
        
        # Step simulation
        mujoco.mj_step(controller.mj_model, controller.mj_data)
        
        # Update time counters
        sim_time_s += cfg.simulation_dt
        decimation_counter += 1

        # DEBUG: print stuff!
        if decimation_counter % 50 == 0:
            print(f"Pitch: {controller.pitch * (180 / np.pi):.3f} deg", 
                  f"Roll: {controller.roll * (180 / np.pi):.3f} deg",
                  f"| Base height: {controller.mj_data.qpos[2]:.3f} m")
            # print("Mean join torques: ", np.mean(np.abs(controller.mj_data.ctrl[:])))


        # Apply control signal every (control_decimation) steps
        if decimation_counter % cfg.control_decimation == 0:
            controller.step(sim_time_s)

        # Joint torque PD control outside of control decimation 
        tau = controller.compute_torques(target_q=controller.target_dof_pos, 
                                         q=controller.qj, 
                                         kp=cfg.kps, 
                                         target_dq=np.zeros_like(cfg.kds), 
                                         dq=controller.dqj, 
                                         kd=cfg.kds)
        controller.mj_data.ctrl[:] = tau


        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Modified timekeeping
        time_elapsed_during_step = time.perf_counter() - step_start # wall-time elapsed in this step
        time_until_next_step = controller.mj_model.opt.timestep - time_elapsed_during_step
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        





