from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Go2ParkourCfg( LeggedRobotCfg ):

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
        add_roughness_to_selected_terrain = False
    
        # Parkour
        num_rows = 10               # num. difficulties
        num_cols = 20               # max. terrain choices
        terrain_length = 28.
        terrain_width = 10.

        # ====================== PARKOUR TERRAIN ======================
        parkour = True
        curriculum = True # TURN OFF FOR FINETUNING
        
        promote_threshold = 0.60
        demote_threshold = 0.40
        terrain_proportions = [1.0, 0.0, 0.0] # [Gap, Box, Hurdles]
        max_init_terrain_level = 2

        # ====================== Jump Finetuning ======================
        # gap_heights = [-2.0, 0.10, -2.0,
        #                -2.0, 0.15, -2.0,
        #                -2.0, 0.20, -2.0,
        #                -2.0, 0.20, -2.0,
        #                -2.0, 0.25, -2.0,
        #                -2.0, 0.25, -2.0]
        
        # gap_lengths = [0.2, 0.2, 0.2]  * 6

        # obstacle_x_positions = [6.0, 6.2, 6.4,
        #                         10.0, 10.2, 10.4,
        #                         14.0, 14.2, 14.4,
        #                         18.0, 18.2, 18.4,
        #                         22.0, 22.2, 22.4,
        #                         26.0, 26.2, 26.4]
        
        # obstacle_y_positions = [0.0, 0.0, 0.0] * 6

        # parkour_kwargs = {
        #     "start_platform_length": 3.,
        #     "start_platform_height": 0.,
    
        #     "x_positions": obstacle_x_positions,
        #     "y_positions": obstacle_y_positions,  # (-) right, (+) left
            
        #     "half_valid_width": 5.0,
        #     "obstacle_heights": gap_heights,  
        #     "obstacle_lengths": gap_lengths,             

        #     "border_width": 0.50,
        #     "border_height": -2.0,
        # }
        # =============================================================

        # ======================== Gap Hurdles ========================
        x_start = 5.0
        dx = 3.5
        n = 7
        gap_heights = [-2.0] * n
        gap_lengths = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.2]

        obstacle_x_positions = list(np.arange(x_start,x_start+n*dx,dx))
        obstacle_y_positions = [0.0] * n

        parkour_kwargs = {
            "start_platform_length": 3.,
            "start_platform_height": 0.,
    
            "x_positions": obstacle_x_positions,
            "y_positions": obstacle_y_positions,  # (-) right, (+) left
            
            "half_valid_width": 5.0,
            "obstacle_heights": gap_heights,  
            "obstacle_lengths": gap_lengths,             

            "border_width": 0.50,
            "border_height": -2.0,
        }
        # =============================================================

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
        pos = [2.0, 0.0, 0.50]    # Parkour [m]
        
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
        terminate_after_contacts_on = ["base", "Head"]
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
        heading_error_gain = 0.5
        class ranges:
            lin_vel_x = [0.75, 1.5]    # [m/s]
            lin_vel_y = [0.0, 0.0]     # [m/s]
            ang_vel_yaw = [-0.25, 0.25]  # [rad/s]
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
        base_height_target = 0.25      # [m]

        pitch_deg_target = 0.0         # [deg]   (+) down, (-) up
        roll_deg_target = 0.0          # [deg]   (+) right, (-) left

        max_foot_height = 0.08         # [m]
        percent_time_on_ground = 0.50  # [%]
        max_contact_force = 100        # [N]

        class scales( LeggedRobotCfg.rewards.scales ):
            # =========================
            tracking_lin_vel = 2.25
            tracking_ang_vel = 2.25
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
            collision = -10.0
            orientation = -1.0
            stumble_feet = -1.0
            # ========================= 
            dof_error = -0.04
            zero_cmd_dof_error = -1.0
            hip_pos = -0.5
            thigh_pos = -0.5
            # =========================
            thigh_symmetry = -0.2
            calf_symmetry = -0.2
            # =========================
            heading_alignment = -4.0    # Parkour only
            reverse_penalty = -1.0      # ABSOLUTELY parkour only
            fwd_jump_vel = 1.25         # Jump & cmd mask
            up_jump_vel = 3.75          # Jump & cmd mask
            # jump_height = 1.0           # Jump & cmd mask
    
            

class Go2ParkourCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        
        # Actor-Critic
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 1.0
        
        # Latent encoders
        priv_encoder_hidden_dims=[64, 20]
        latent_encoder_output_dim = 20

        # Scan encoder
        scan_encoder_hidden_dims=[128, 64]
        scan_encoder_output_dim = 32

        # Estimator
        # estimator_hidden_dims = [128, 64]
        estimator_hidden_dims = [256, 128]
        use_history = True
        
        # Activation (all)
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid


    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        estimator_learning_rate = 1e-4
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
        max_iterations = 20000
        save_interval = 50

        run_name = 'parkour_v12'
        experiment_name = 'go2_parkour'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
