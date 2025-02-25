import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import pdb

# ========================================================
def get_gravity_orientation(quaternion):
    """BRO WHAT FORMULA IS THIS?"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def get_pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return kp*(target_q-q) + kd*(target_dq-dq)
# ========================================================

if __name__ == "__main__":
    # (For debug) round print statements
    np.set_printoptions(formatter={'float': lambda x: f"{x:.3g}"})

    # Get name of YAML config file from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Specify the name of the YAML config file to use.")
    args = parser.parse_args()
    config_file = args.config_file

    # Load config file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Load arguments
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        run_forever = config["run_forever"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kp_gains = np.array(config["kp_gains"], dtype=np.float32) 
        kd_gains = np.array(config["kd_gains"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["command"], dtype=np.float32)

    # Initialize some non-scalar data structures
    actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    """
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata
    """
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy file
    policy = torch.jit.load(policy_path)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Start simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        # Close the viewer automatically after simulation_duration wall-seconds.
        while viewer.is_running() and (time.time() - start < simulation_duration or run_forever):
            step_start = time.time()

            # print("d.qpos len: ", len(d.qpos)) # 19
            # print("d.qvel len: ", len(d.qvel)) # 18

            # qj_pos = d.sensordata[0:12] # d.qpos[7:]
            # qj_vel = d.sensordata[12:24] # d.qvel[6:]

            # Get current q
            qj_pos = d.qpos[7:] # 19 --> 12
            qj_vel = d.qvel[6:] # 18 --> 12

            # Joint torque PD control
            tau = get_pd_control(target_dof_pos, qj_pos, kp_gains, np.zeros_like(kd_gains), qj_vel, kd_gains)
            d.ctrl[:] = tau
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Apply control signal every (control_decimation) steps
            counter += 1
            if counter % control_decimation == 0:

                # Create observation
                qj = qj_pos
                dqj = qj_vel
                lin_vel = d.qvel[:3]                        # linear vel. in the world frame
                ang_vel = d.qvel[3:6]                       # angular vel. in the world frame
                base_rot_quat = d.qpos[3:7]                 # rotation of base in quaternion

                # ========== rotation math ==========
                temp = np.zeros(9)   
                mujoco.mju_quat2Mat(temp, base_rot_quat)
                base_rot_mat = temp.reshape(3, 3)           # rotation of base in matrix form
                base_lin_vel = base_rot_mat.T @ lin_vel     # linear vel. in the body frame
                # ===================================

                # Match observation format of Isaac Gym
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale
                projected_gravity = get_gravity_orientation(base_rot_quat)
                
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                obs[:3] = base_lin_vel
                obs[3:6] = ang_vel
                obs[6:9] = projected_gravity
                obs[9:12] = cmd * cmd_scale                 # overflow unless multiplied here
                obs[12 : 12+num_actions] = qj
                obs[12+num_actions : 12+2*num_actions] = dqj
                obs[12+2*num_actions : 12+3*num_actions] = actions

                # Convert to tensor
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # Get actions from policy
                actions = policy(obs_tensor).detach().numpy().squeeze()

                # Transform action to target_dof_pos
                target_dof_pos = actions * action_scale + default_angles


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
