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
# ========================================================
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
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/unified_configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Paths
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        
        # Timing
        simulation_duration = config["simulation_duration"]
        run_forever = config["run_forever"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
        # Motor-related
        kps = np.array(config["kps"], dtype=np.float32) 
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        
        # Scales
        lin_accel_scale = config["lin_accel_scale"] # (Deprecated)
        lin_vel_scale = config["lin_vel_scale"] # (Deprecated)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        
        # Actions & Observations
        num_actions = config["num_actions"]
        num_proprio = config["num_proprio"]
        enable_history = config["enable_history"]
        buffer_length = config["buffer_length"]
        num_obs = (num_proprio * buffer_length if enable_history else num_proprio)

        # Phase
        period = config["period"]
        fr_offset = config["fr_offset"]
        bl_offset = config["bl_offset"]
        fl_offset = config["fl_offset"]
        br_offset = config["br_offset"]

        # User command
        cmd = np.array(config["command"], dtype=np.float32)


    # Initialize non-optional (essential) buffers
    actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros((buffer_length, num_proprio), dtype=np.float32)

    """
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata
    """

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy file
    policy = torch.jit.load(policy_path)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Initialize phase-related variables
    sim_time_s = 0.0

    # Start simulation
    counter = 0

    # Initialize Mujoco Viewer
    viewer = mujoco.viewer.launch_passive(m, d)

    # Have camera track the robot base
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 0

    # Set some visualization flags
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    start = time.time()

    # Close the viewer automatically after simulation_duration wall-seconds.
    while viewer.is_running() and (time.time() - start < simulation_duration or run_forever):
        step_start = time.time()

        # Get current q
        qj_pos = d.qpos[7:] # 19 --> 12
        qj_vel = d.qvel[6:] # 18 --> 12

        # Joint torque PD control
        tau = get_pd_control(target_dof_pos, qj_pos, kps, np.zeros_like(kds), qj_vel, kds)
        d.ctrl[:] = tau
        
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Update simulation time
        sim_time_s += simulation_dt

        # Apply control signal every (control_decimation) steps
        counter += 1
        if counter % control_decimation == 0:

            # Prepare observation quantities
            qj = qj_pos
            dqj = qj_vel
            lin_vel = d.qvel[:3]                        # linear vel. in the world frame (Deprecated)
            ang_vel = d.qvel[3:6]                       # angular vel. in the world frame
            lin_accel = d.qacc[:3]                      # linear accel. in the world frame (Deprecated)

            # ========== rotation math ==========
            base_rot_quat = d.qpos[3:7]                 # rotation of base in quaternion
            temp = np.zeros(9)   
            mujoco.mju_quat2Mat(temp, base_rot_quat)
            base_rot_mat = temp.reshape(3, 3)           # rotation of base in matrix form
            base_lin_vel = base_rot_mat.T @ lin_vel     # linear vel. in the body frame (Deprecated)
            base_lin_accel = base_rot_mat.T @ lin_accel # linear accel. in the body frame (Deprecated)
            # ===================================

            # Get projected gravity
            projected_gravity = get_gravity_orientation(base_rot_quat)
            
            # Get sensor data for accelerometer (linear accel)
            # accel_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
            # sensor_lin_accel = d.sensordata[accel_sensor_id:accel_sensor_id+3]
            
            # Prepare phase features (*MATCH*)
            phase = (sim_time_s % period) / period
            phase_fr = (phase + fr_offset) % 1
            phase_bl = (phase + bl_offset) % 1
            phase_fl = (phase + fl_offset) % 1
            phase_br = (phase + br_offset) % 1

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


            # Create observation tensor
            obs[:3] = ang_vel * ang_vel_scale
            obs[3:6] = projected_gravity
            obs[6:9] = cmd * cmd_scale 
            obs[9 : 9+num_actions] = (qj - default_angles) * dof_pos_scale
            obs[9+num_actions : 9+2*num_actions] = dqj * dof_vel_scale
            obs[9+2*num_actions : 9+3*num_actions] = actions
            obs[9+3*num_actions:9+3*num_actions+8] = phase_features

            # Add history to observation tensor
            if enable_history:
                cur_obs = obs[:9+3*num_actions+8] # slice to exclude pre-allocated history indices

                if counter == control_decimation:  # First control step
                    obs_history = np.zeros((buffer_length-1, num_proprio), dtype=np.float32)
                    for i in range(buffer_length-1):
                        obs_history[i] = cur_obs
                else:
                    obs_history = np.roll(obs_history, 1, axis=0)
                    obs_history[-1] = cur_obs
                
                obs[:(buffer_length-1)*num_proprio] = obs_history.flatten()
                obs[(buffer_length-1)*num_proprio:] = cur_obs 

            # Convert to tensor
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            
            # Get actions from policy network
            actions = policy(obs_tensor).detach().numpy().squeeze()

            # Transform action to target_dof_pos
            target_dof_pos = actions * action_scale + default_angles


        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
