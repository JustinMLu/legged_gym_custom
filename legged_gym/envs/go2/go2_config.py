from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        # Extremely important ones
        num_envs = 4096
        num_proprio = 53
        num_scan_obs =  0 # 132 # Not used yet
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

        # General parameters
        num_rows = 10               # num. difficulties     ->    (0/n, 1/n, 2/n ... (n-1)/n)
        num_cols = 20               # max. terrain choices  ->    affects terrain_proportions "accuracy"
        mesh_type = 'plane'
        measure_heights = False     # add a height measurement to the observations
        
        # Extreme parkour (132 SCANDOTS)
        # measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        # measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        
        # Manual terrain selection
        selected = False
        # ========================================================
        terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.07,
            "max_height": 0.07,
            "step": 0.005,
            "downsampled_scale": 0.2
        }

        terrain_kwargs = {
            "type": "terrain_utils.discrete_obstacles_terrain",
            "max_height": 0.4,
            "min_size": 1.,
            "max_size": 2.,
            "num_rects": 20,
            "platform_size": 3.
        }

        terrain_kwargs = {
            "type": "terrain_utils.pyramid_sloped_terrain",
            "slope": 1.0*0.5,
            "platform_size": 3.
        }

        terrain_kwargs = {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width":0.25,
            "step_height":-0.165,
            "platform_size":2.
        }
        # ========================================================
        # Terrain curriculum
        curriculum = False
        max_init_terrain_level = 1      # starting curriculum state
        promote_threshold = 0.5         # [%] of terrain traversed
        demote_threshold = 0.4          # [%] of terrain traversed

        terrain_default     = [0.15,    # smooth slope
                               0.15,    # rough slope
                               0.25,    # stairs up
                               0.25,    # stairs down
                               0.10,    # discrete terrain
                               0.00,    # stepping stones
                               0.10]    # random uniform NOTE: Turning this on artificially boosts terrain difficulty, as it doesn't get harder
        
        terrain_stairs      = [0.00,    # smooth slope
                               0.00,    # rough slope
                               0.75,    # stairs up
                               0.25,    # stairs down
                               0.00,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # random uniform
        
        terrain_proportions = terrain_stairs
        # ========================================================
    class domain_rand:      
        randomize_friction = True
        # friction_range = [0.1, 1.0] # Extreme Parkour
        friction_range = [0.2, 1.25]

        randomize_base_mass = True
        added_mass_range = [0., 3.]

        randomize_center_of_mass = True
        added_com_range = [-0.15, 0.15]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.0 # 25.0 for extra robust
    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]      # [x, y, z] (metres)
        
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
        penalize_contacts_on = ["base", "hip," "thigh", "calf", "Head"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable (bitwise filter)



    class commands ( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 10.
        zero_command_prob = 0.10 # Resampling probability
        
        # Command curriculum
        curriculum = True
        max_curriculum = 2.2 # [m/s]
        
        class ranges:
            # Default
            # lin_vel_x = [-1.0, 1.0]     # [m/s]
            # lin_vel_y = [-1.0, 1.0]     # [m/s]
            # ang_vel_yaw = [-1.0, 1.0]   # [rad/s]
            # heading = [-3.14, 3.14]

            # Cheetah
            lin_vel_x = [-0.2, 0.5]     # [m/s]
            lin_vel_y = [-0.0, 0.0]     # [m/s]
            ang_vel_yaw = [-0.5, 0.5]   # [rad/s]


    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 3.8
        
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
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.295
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.6
            # ======================
            lin_vel_z = -0.1
            ang_vel_xy = -0.01
            torques = -0.00001 * 0.5
            dof_acc = -2.5e-7
            collision = -10.0
            delta_torques = -1.0e-7
            dof_error = -0.04 
            hip_pos = -0.75
            orientation = -2.5            # -5.0 for super stable
            # ====================== 
            contact_phase_match = 1.0
            action_rate = -0.01           # decreased (-0.025 to -0.010) 
            # base_height = -20.0         
            # stumble_feet = -2.5         # renamed
            # stumble_calves = -2.5       # new


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
        run_name = 'cheetah'
        experiment_name = 'go2'
        load_run = -1
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 50
