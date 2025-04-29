from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from deploy.deploy_mujoco.gamepad_reader import Gamepad

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.commands.user_command = [0.0, 0.0, 0.0, 0.0] # this SHOULD stop the resampling?
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_center_of_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False             # ESSENTIAL OTHERWISE JOYSTICK WILL FIGHT YOU

    # Initialize gamepad
    gamepad = Gamepad(1.25, 0.0, 1.57)           # Manually have to calibrate with rc_scale :(

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load inference policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    inference_policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Export policy files as jit
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, ppo_runner.alg.estimator, path)
        
    # Logger 
    logger = Logger(env.dt)
    robot_index = 0                             # which robot is used for logging
    joint_index = 1                             # which joint is used for logging
    stop_state_log = 100                        # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1   # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # Initialize the obs variables
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    critic_obs = env.get_critic_observations()
    estimated_obs = env.get_estimated_observations()
    scan_obs = env.get_scan_observations()

    # Specify custom camera
    ISO_PITCH = np.deg2rad(25)        # Pitch (up/down) angle
    ISO_YAW   = np.deg2rad(0)        # Yaw (rotation) angle
    ISO_DIST  = 2.0                   # metres from the robot
    ISO_FOV   = 25.0                  # deg – optional, see below

    # Build camera direction vector
    cam_dir_vec   = np.array([
        -np.cos(ISO_PITCH) * np.cos(ISO_YAW),
        -np.cos(ISO_PITCH) * np.sin(ISO_YAW),
        np.sin(ISO_PITCH)
    ])

    i = 0
    while True:

        # Xbox gamepad control
        env.commands[:, 0] = gamepad.vx * env.cfg.normalization.obs_scales.lin_vel
        env.commands[:, 1] = gamepad.vy * env.cfg.normalization.obs_scales.lin_vel
        env.commands[:, 2] = gamepad.wz * env.cfg.normalization.obs_scales.ang_vel

        # Compute actions & step
        actions = inference_policy(obs.detach(), privileged_obs.detach(), estimated_obs.detach(), scan_obs.detach(), adaptation_mode=True) # use adaption module
        obs, privileged_obs, critic_obs, estimated_obs, scan_obs, rews, dones, infos = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 

        # Set custom camera
        robot_pos = env.root_states[0, :3].cpu().numpy()
        camera_pos = robot_pos + ISO_DIST * cam_dir_vec
        env.set_camera(camera_pos, robot_pos)


        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log and SHOW_PLOTS:
            logger.plot_states()

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        i += 1

if __name__ == '__main__':
    SHOW_PLOTS = False
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    play(args)
