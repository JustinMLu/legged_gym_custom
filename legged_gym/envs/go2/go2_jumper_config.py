from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Go2JumperCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_proprio = 52    
        num_scan_obs =  132
        num_estimated_obs = 3
        num_privileged_obs = 4+1+12+12
        history_buffer_length = 10
        num_actions = 12
        num_critic_obs = num_proprio+(num_proprio*history_buffer_length)+num_privileged_obs+num_estimated_obs+num_scan_obs
        num_observations = num_proprio+(num_proprio*history_buffer_length)

        # Phase features
        period = 0.35
        fr_offset = 0.0 
        bl_offset = 0.5
        fl_offset = 0.0
        br_offset = 0.5

    class terrain( LeggedRobotCfg.terrain ):
        # Scandots (132)
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 12
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]   # 11
        
        # General
        mesh_type = 'trimesh'
        measure_heights = True      # for go2 this just enables draw_debug_vis
        add_roughness_to_selected_terrain = True
    
        # Parkour
        num_rows = 1                # num. difficulties
        num_cols = 200              # max. terrain choices
        terrain_length = 28.
        terrain_width = 8.

        # ======================== Parkour Terrains ========================
        parkour = True
        hurdle_x_positions = [4, 7, 10, 13, 16, 19, 22, 25]
        hurdle_y_positions = [0.0] * len(hurdle_x_positions)
        hurdle_heights = [0.10, 0.15, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25]

        parkour_hurdle_kwargs = {
            "platform_len": 3.,
            "platform_height": 0.,
    
            "x_positions": hurdle_x_positions,
            "y_positions": hurdle_y_positions,  # (-) right, (+) left
            
            "half_valid_width": 4.0,
            "hurdle_heights": hurdle_heights,  
            "hurdle_thickness": 0.35,             

            "border_width": 0.25,
            "border_height": 1.0,
        }

        # Not used until I can figure out waypoint implementation
        parkour_hurdle_randomized_kwargs = {
            "platform_len": 3.,
            "platform_height": 0.,

            "x_range": [2.0, 5.0],          # These are the DELTAS BETWEEN HURDLES
            "y_range": [-8.1, -7.9],        # (-) right, (+) left

            "num_hurdles": 5,                   
            "hurdle_thickness": 0.35,             
            "hurdle_height_range": [0.15, 0.35],  
            "half_valid_width": [3.8, 3.9],

            "border_width": 0.25,
            "border_height": 1.0,
        }
        
        # ==================== Manual Terrain Selection ====================
        selected = False

        random_uniform_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.01,
            "max_height": 0.01,
            "step": 0.005,
            "downsampled_scale": 0.3
        }

        pyramid_sloped_kwargs = {
            "type": "terrain_utils.pyramid_sloped_terrain",
            "slope": 0.5,
            "platform_size": 3.,
        }

        discrete_obstacles_kwargs = {
            "type": "terrain_utils.discrete_obstacles_terrain",
            "max_height": 0.4,
            "min_size": 1.,
            "max_size": 2.,
            "num_rects": 20,
            "platform_size": 3.
        }

        wave_kwargs = {
            "type": "terrain_utils.wave_terrain",
            "num_waves": 1.,
            "amplitude": 0.7,
        }

        pyramid_stairs_kwargs = {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width": 0.25,
            "step_height": -0.165,
            "platform_size": 2.
        }

        stepping_stones_kwargs = {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_size": 0.6,
            "stone_distance": 0.4,
            "max_height": 0.4,
            "platform_size": 3.,
            "depth": -5.0,
            }

        terrain_kwargs = random_uniform_kwargs        
      
        # ============= Terrain Curriculum (TODO: Rework this) =============
        curriculum = False
        max_init_terrain_level = 1      # max. start level
        promote_threshold = 0.5         # [%] of terrain 
        demote_threshold = 0.4          # [%] of terrain
        terrain_default     = [0.20,    # smooth slope
                               0.20,    # rough slope
                               0.20,    # stairs up
                               0.20,    # stairs down
                               0.20,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # random uniform
        terrain_stairs      = [0.00,    # smooth slope
                               0.00,    # rough slope
                               0.75,    # stairs up
                               0.25,    # stairs down
                               0.00,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # random uniform
        terrain_proportions = terrain_default
        # ============= Terrain Curriculum (TODO: Rework this) =============
    
    class domain_rand:      
        randomize_friction = True
        friction_range = [0.3, 1.2]

        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]

        randomize_center_of_mass = True
        added_com_range = [-0.15, 0.15]

        randomize_kp_kd = True
        kp_kd_range = [0.8, 1.2]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class init_state( LeggedRobotCfg.init_state ):
        pos = [-12.25, 0.0, 0.42]    # Parkour [m]
        
        default_joint_angles = {
            'FL_hip_joint':  0.1, 'FL_thigh_joint': 0.8, 'FL_calf_joint': -1.5, 
            'FR_hip_joint': -0.1, 'FR_thigh_joint': 0.8, 'FR_calf_joint': -1.5,
            'RL_hip_joint':  0.1, 'RL_thigh_joint': 1.0, 'RL_calf_joint': -1.5, 
            'RR_hip_joint': -0.1, 'RR_thigh_joint': 1.0, 'RR_calf_joint': -1.5
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'          # Position control 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["base", "hip", "thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable (bitwise filter)

    class commands ( LeggedRobotCfg.commands ):
        

        # General
        resampling_time = 10.     # [seconds]
        zero_command = True      
        zero_command_prob = 0.10

        # Command curriculum
        curriculum = False
        max_forward_vel = 1.75    # [m/s], (+) is forward
        max_reverse_vel = 0.5     # [m/s], (+) is forward
        vel_increment = 0.10      # [m/s]

        # Ranges
        heading_command = True
        class ranges:
            lin_vel_x = [0.75, 1.5]     # [m/s]
            lin_vel_y = [0.0, 0.0]     # [m/s]
            ang_vel_yaw = [-0.0, 0.0]   # [rad/s]
            heading = [-0.2, 0.2]

    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 3.14
        
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0


    class noise( LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales( LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.1
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            imu = 0.02
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9       # [%]
        base_height_target = 0.29      # [m]

        pitch_deg_target = 0.0         # [deg]   (+) down, (-) up
        roll_deg_target = 0.0          # [deg]   (+) right, (-) left

        max_foot_height = 0.08         # [m]
        percent_time_on_ground = 0.50  # [%]
        max_contact_force = 100        # [N]

        class scales( LeggedRobotCfg.rewards.scales ):
            # =========================
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.5
            # ========================= 
            phase_contact_match = 1.0
            phase_foot_lifting = 1.0
            # =========================
            action_rate = -0.1         
            lin_vel_z = -1.0
            ang_vel_xy = -0.01
            torques = -0.00001
            dof_acc = -2.5e-7
            delta_torques = -1.0e-7
            # ========================= 
            orientation = -1.0
            stumble_feet = -1.0
            collision = -10.0
            calf_collision = -20.0
            # ========================= 
            dof_error = -0.05
            hip_pos = -0.5
            # =========================
            zero_cmd_dof_error = -0.4
            thigh_symmetry = -0.2
            calf_symmetry = -0.2
            # =========================
            minimum_base_height = -20.
            heading_alignment = -2.0    # Parkour only
            jump_velocity = 2.5         # Parkour only
            

class Go2JumperCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        latent_encoder_output_dim = 20
        scan_encoder_output_dim = 32
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed' # fixed or adaptive
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        dagger_update_freq = 20

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
       
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 50

        run_name = 'jumper_scandots_v2'
        experiment_name = 'go2_jumper'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
