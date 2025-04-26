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


        self._wave_active = False
        self._wave_step = 0
        self._wave_len = 11

    def _get_scan_obs(self) -> torch.Tensor:
        """
        Return (1, num_scan_obs) tensor for the scan-encoder.

        If RB is pressed, spoof a wall that the policy learned to jump over: 
        three rows (≈0.9-1.2 m ahead) are set to -1.  
        Otherwise we return all-zeros (flat ground).
        """
        NX, NY = 12, 11
        scan = torch.zeros((1, self.cfg.num_scan_obs), dtype=torch.float32)
        
        if self.gamepad._rb_pressed and not self._wave_active:
            self._wave_active = True
            self._wave_step   = 0

        if self._wave_active:
            active_row = NY - 1 - self._wave_step
            if active_row >= 0:
                start = active_row * NX
                end   = (active_row + 1) * NX       # row-major slice
                scan[0, start:end] = -0.5           # -1 is 30cm wall
                self._wave_step += 1
            else:
                # finished: reset state-machine
                self._wave_active = False

            if self._wave_step >= self._wave_len:
                self._wave_active = False

        return scan


        # # 132 cells = 12 columns (x-axis, forward)  ×  11 rows (y-axis, left⇄right)
        # NX, NY = 12, 11                        
        # scan = torch.zeros((1, self.cfg.num_scan_obs), dtype=torch.float32)

        # if self.gamepad._rb_pressed:      # your Gamepad class already exposes this
        #     front_rows = [8, 9, 10]            # rows in front of the robot
        #     for r in front_rows:
        #         start = r * NX                # row-major flattening
        #         end   = (r + 1) * NY
        #         scan[0, start:end] = -5.0      # –1 ⇒ 30 cm obstacle in training data

        # return scan


    def refresh_robot_states(self):
        """ Retrieve the latest robot state (joints, velocities, orientation, etc.) from 
            the environment and store it in this controller's internal buffers. 
        
            Should update the following data:
                - qj (joint pos.)
                - dqj (joint vel.)
                - ang_vel (in the local frame)
                - base_quat (base orientation quaternion)
        """

        # Input smoothed commands
        smoothed_cmd = self.get_smoothed_command([self.gamepad.vx, self.gamepad.vy, self.gamepad.wz], 0.025)
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
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
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
        





