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
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # Clipping
        clip_obs = config["clip_observations"]
        clip_actions = config["clip_actions"]
        
        # Actions & Observations
        num_actions = config["num_actions"]
        num_proprio = config["num_proprio"]
        enable_history = config["enable_history"]
        buffer_length = config["buffer_length"]
        num_obs = num_proprio+(num_proprio*buffer_length) if enable_history else num_proprio

        # Phase
        # period = config["period"]
        fr_offset = config["fr_offset"]
        bl_offset = config["bl_offset"]
        fl_offset = config["fl_offset"]
        br_offset = config["br_offset"]

        # User command
        cmd = np.array(config["command"], dtype=np.float32)


    # Initialize essential buffers
    actions = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros((buffer_length, num_proprio), dtype=np.float32)

    """
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
    https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata
    """

    # Load robot model
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = simulation_dt

    # Load policy file
    policy = torch.jit.load(policy_path)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    # Initialize phase-related variables
    sim_time_s = 0.0

    # Start simulation
    counter = 0
    first_step_ever = True

    # Initialize Mujoco Viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Have camera track the robot base
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 0

    # Set some visualization flags
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    
    start = time.perf_counter()

    # Close the viewer automatically after simulation_duration wall-seconds.
    while viewer.is_running() and ((time.perf_counter()-start) < simulation_duration or run_forever):
        
        # Start time of current step
        step_start = time.perf_counter()

        # Get current q
        qj_pos = mj_data.qpos[7:] # 19 --> 12
        qj_vel = mj_data.qvel[6:] # 18 --> 12

        # Joint torque PD control
        tau = get_pd_control(target_dof_pos, qj_pos, kps, np.zeros_like(kds), qj_vel, kds)
        mj_data.ctrl[:] = tau
        
        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Update simulation time
        sim_time_s += simulation_dt

        # Apply control signal every (control_decimation) steps
        counter += 1
        if counter % control_decimation == 0:

            # Prepare observation quantities
            qj = qj_pos                                         # joint positions 
            dqj = qj_vel                                        # joint velocities
            ang_vel = mj_data.qvel[3:6]                         # angular vel. (local frame)
            base_rot_quat = mj_data.qpos[3:7]                   # base rot. in quaternion
            base_rot_mat = np.zeros(9)   
            mujoco.mju_quat2Mat(base_rot_mat, base_rot_quat)
            base_rot_mat = base_rot_mat.reshape(3, 3)           # base rot. in matrix form

            # Get projected gravity
            projected_gravity = get_gravity_orientation(base_rot_quat)
            
            # Calculate gait period
            cmd_norm = np.linalg.norm(cmd[:2])                  # DOESN'T FACTOR IN ANG_VEL_YAW FOR NOW
            period = 1.0 / (1.0 + cmd_norm)                     
            period = (period * 2.0) * 0.66                      # Scale
            period = np.clip(period, a_min=0.25, a_max=1.0)     # Clamp result

            # Calculate per-leg phase
            phase = (sim_time_s % period) / period
            phase_fr = (phase + fr_offset) % 1
            phase_bl = (phase + bl_offset) % 1
            phase_fl = (phase + fl_offset) % 1
            phase_br = (phase + br_offset) % 1

            # Calculate sine and cosine of phases for smooth transitions
            sin_phase_fl = np.sin(2 * np.pi * phase_fl)
            cos_phase_fl = np.cos(2 * np.pi * phase_fl)
            sin_phase_fr = np.sin(2 * np.pi * phase_fr)
            cos_phase_fr = np.cos(2 * np.pi * phase_fr)
            sin_phase_bl = np.sin(2 * np.pi * phase_bl)
            cos_phase_bl = np.cos(2 * np.pi * phase_bl)
            sin_phase_br = np.sin(2 * np.pi * phase_br)
            cos_phase_br = np.cos(2 * np.pi * phase_br)
        
            # Construct phase features - zero out if small command
            if cmd_norm < 0.2:
                phase_features = np.array([
                    0.0, 0.0, 
                    0.0, 0.0, 
                    0.0, 0.0, 
                    0.0, 0.0
                ], dtype=np.float32)
            else:
                phase_features = np.array([
                    sin_phase_fr, cos_phase_fr, 
                    sin_phase_fl, cos_phase_fl,
                    sin_phase_bl, cos_phase_bl,
                    sin_phase_br, cos_phase_br
                ], dtype=np.float32)

            print(f"Base height: {mj_data.qpos[2]:.3f} meters")

            # Create observation list
            cur_obs = np.zeros(num_proprio, dtype=np.float32)
            cur_obs[:3] = ang_vel * ang_vel_scale
            cur_obs[3:6] = projected_gravity
            cur_obs[6:9] = cmd * cmd_scale 
            cur_obs[9 : 9+num_actions] = (qj - default_angles) * dof_pos_scale
            cur_obs[9+num_actions : 9+2*num_actions] = dqj * dof_vel_scale
            cur_obs[9+2*num_actions : 9+3*num_actions] = actions
            cur_obs[9+3*num_actions:9+3*num_actions+8] = phase_features

            # Concatenate obs history if enabled
            if enable_history:
                obs[:] = np.concatenate([obs_history.flatten(), cur_obs])
                # Then, add current observation to history
                if first_step_ever:
                    first_step_ever = False
                    obs_history = np.tile(cur_obs, (buffer_length, 1))  # (4x1, 1x53)
                else:
                    obs_history = np.roll(obs_history, -1, axis=0)
                    obs_history[-1] = cur_obs
            else:
                obs[:] = cur_obs
            
            # Convert to tensor, clip 
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            obs_tensor = torch.clip(obs_tensor, -clip_obs, clip_obs)
            
            # Get actions from policy network, clip
            actions = policy(obs_tensor)
            actions = torch.clip(actions, -clip_actions, clip_actions).detach().numpy().squeeze()

            # Transform action to target_dof_pos
            target_dof_pos = actions * action_scale + default_angles

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Modified timekeeping
        time_elapsed_during_step = time.perf_counter() - step_start # wall-time elapsed in this step
        time_until_next_step = mj_model.opt.timestep - time_elapsed_during_step
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
