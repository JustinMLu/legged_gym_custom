from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_proprio = 53
        num_scan_obs =  0   # 132 but not used yet
        num_estimated_obs = 3
        num_privileged_obs = 4+1+12+12
        history_buffer_length = 10
        num_actions = 12
        num_critic_obs = num_proprio+(num_proprio*history_buffer_length)+num_privileged_obs+num_estimated_obs+num_scan_obs
        num_observations = num_proprio+(num_proprio*history_buffer_length)

        # Phase features
        period = 0.40
        fr_offset = 0.0 
        bl_offset = 0.0
        fl_offset = 0.5
        br_offset = 0.5

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False      # reworked
        num_rows = 10                # num. difficulties
        num_cols = 20                # max. terrain choices
        terrain_length = 8.
        terrain_width = 8.
        add_roughness_to_selected_terrain = False

        # ======================== Parkour Terrains ========================
        parkour = False
        hurdle_x_positions = [5.0, 8.0, 11.0, 14.0, 17.0, 20.0]
        hurdle_y_positions = [-8.0, -8.0, -8.0, -8.0, -8.0, -8.0]
        hurdle_heights = [0.15, 0.15, 0.20, 0.25, 0.25, 0.30]

        parkour_hurdle_kwargs = {
            "platform_len": 3.,
            "platform_height": 0.,
    
            "x_positions": hurdle_x_positions,
            "y_positions": hurdle_y_positions,  # (-) right, (+) left
            
            "half_valid_width": 4.0, # hurdle width / 2
            "hurdle_heights": hurdle_heights,  
            "hurdle_thickness": 0.3,             

            "border_width": 0.25,
            "border_height": 1.0,
        }

        parkour_hurdle_randomized_kwargs = {
            "platform_len": 3.,
            "platform_height": 0.,

            "x_range": [3.0, 6.0],          # (-) backward, (+) forward
            "y_range": [-6.0, -5.9],        # (-) right, (+) left

            "num_hurdles": 3,                   
            "hurdle_thickness": 0.4,             
            "hurdle_height_range": [0.3, 0.35],  
            "half_valid_width": [3.9, 4.0], # hurdle width / 2

            "border_width": 0.25,
            "border_height": 1.0,
        }
        
        # ==================== Manual Terrain Selection ====================
        selected = False

        random_uniform_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.025,
            "max_height": 0.025,
            "step": 0.005,
            "downsampled_scale": 0.2
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

    
    class domain_rand:      
        randomize_friction = True
        friction_range = [0.3, 1.1]

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
        pos = [0.0, 0.0, 0.42]      # [x, y, z] (metres)
        # pos = [-10.0, 0.0, 0.42]    # Parkour
        
        default_joint_angles = {
            'FL_hip_joint':  0.1, 'FL_thigh_joint': 0.9, 'FL_calf_joint': -1.7, 
            'FR_hip_joint': -0.1, 'FR_thigh_joint': 0.9, 'FR_calf_joint': -1.7,
            'RL_hip_joint':  0.1, 'RL_thigh_joint': 1.1, 'RL_calf_joint': -1.7, 
            'RR_hip_joint': -0.1, 'RR_thigh_joint': 1.1, 'RR_calf_joint': -1.7
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
        penalize_contacts_on = ["base", "hip," "thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable (bitwise filter)

    class commands ( LeggedRobotCfg.commands ):
        
        # Command curriculum
        curriculum = True
        max_forward_vel = 1.5     # [m/s]
        max_reverse_vel = -1.0    # [m/s]
        vel_increment = 0.10      # [m/s]

        # General
        resampling_time = 10.     # [seconds]
        zero_command = True
        zero_command_prob = 0.10
        heading_command = False

        # Ranges
        class ranges:
            lin_vel_x = [-1.0, 1.0]     # [m/s]
            lin_vel_y = [-1.0, 1.0]     # [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [rad/s]
            # heading = [-3.14, 3.14]

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
        base_height_target = 0.25      # [m]

        pitch_deg_target = 0.0         # [deg]   (+) down, (-) up
        roll_deg_target = 0.0          # [deg]   (+) right, (-) left

        max_foot_height = 0.08         # [m]
        percent_time_on_ground = 0.50  # [%]
        max_contact_force = 100        # [N]

        class scales( LeggedRobotCfg.rewards.scales ):
            
            # =========================
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            # ========================= 
            phase_contact_match = 1.0
            phase_foot_lifting = 0.5
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
            # ========================= 
            dof_error = -0.04
            hip_pos = -0.5
            # =========================
            # z_vel_when_jumping = 2.0
            # heading_alignment = -5.0
            

class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 2e-4
        entropy_coef = 0.01
        value_loss_coef = 1.0
        dagger_update_freq = 20
        schedule = 'fixed' # fixed or adaptive

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'jumper_base_flat'
        experiment_name = 'go2'
        load_run = -1
        num_steps_per_env = 24
        max_iterations = 30000
        save_interval = 50
