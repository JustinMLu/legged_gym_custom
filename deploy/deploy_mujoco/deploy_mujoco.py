import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

# ========================================================

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# ========================================================

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Path to saved PyTorch .pt policy file
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        
        # Path to mujoco xml file (so scene.xml which includes go2.xml)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
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
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Debug
    import pdb
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # qj_pos = d.sensordata[0:12] # d.qpos[7:]
            # qj_vel = d.sensordata[12:24] # d.qvel[6:]
            qj_pos = d.qpos[7:]
            qj_vel = d.qvel[6:]

            # Torque PDF control
            tau = pd_control(target_dof_pos, qj_pos, kp_gains, np.zeros_like(kd_gains), qj_vel, kd_gains)
            d.ctrl[:] = tau
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = qj_pos
                dqj = qj_vel
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                vel_world = d.qvel[:3]
                rot_vec = np.zeros(9)
                mujoco.mju_quat2Mat(rot_vec, quat)
                rot_mat = rot_vec.reshape(3, 3)
                vel_body = rot_mat.T @ vel_world

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # # Originally for g1
                # obs[:3] = omega
                # obs[3:6] = gravity_orientation
                # obs[6:9] = cmd * cmd_scale
                # obs[9 : 9 + num_actions] = qj
                # obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                # obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs[:3] = vel_body
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12 : 12 + num_actions] = qj
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action
                              
                # print(obs)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
