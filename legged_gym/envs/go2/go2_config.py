from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        enable_history = True
        buffer_length = 9 # number of previous obs to keep in buffer
        num_proprio = 53 
        num_observations = num_proprio+(num_proprio*buffer_length) if enable_history else num_proprio
        num_envs = 4096
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        num_rows = 20 # num. difficulties       ->    (0/n, 1/n, 2/n ... (n-1)/n)
        num_cols = 20 # max. terrain choices    ->    affects terrain_proportions "accuracy"

        mesh_type = 'trimesh'
        measure_heights = False     # changed so this only enables the buffer & noise
        max_init_terrain_level = 2  # starting curriculum state
        

        selected = False
        terrain_kwargs = {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width": 0.24,
            "step_height": -0.17,
            "platform_size": 2.0,
        }

        curriculum = True
        terrain_default     = [0.10,    # smooth slope
                               0.15,    # rough slope
                               0.30,    # stairs up
                               0.20,    # stairs down
                               0.10,    # discrete terrain
                               0.00,    # stepping stones
                               0.15]    # bumpy wave
        
        terrain_stairs      = [0.00,    # smooth slope
                               0.00,    # rough slope
                               0.75,    # stairs up
                               0.25,    # stairs down
                               0.00,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # bumpy wave
        
        terrain_proportions = terrain_default

    class domain_rand:      
        randomize_friction = True
        friction_range = [0.25, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1.1, 1.1]
        
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.75
    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]      # [x, y, z] (metres)
        
        default_joint_angles = {
            'FL_hip_joint':  0.1, 'FL_thigh_joint': 1.0, 'FL_calf_joint': -1.8, 
            'FR_hip_joint': -0.1, 'FR_thigh_joint': 1.0, 'FR_calf_joint': -1.8,
            'RL_hip_joint':  0.1, 'RL_thigh_joint': 1.0, 'RL_calf_joint': -1.8, 
            'RR_hip_joint': -0.1, 'RR_thigh_joint': 1.0, 'RR_calf_joint': -1.8
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
        penalize_contacts_on = ["thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter



    class commands ( LeggedRobotCfg.commands ):
        heading_command = False
        curriculum = False
        max_curriculum = 1.25
        resampling_time = 10. # [s] TODO: Mess with this
        
        class ranges:
            # # 1. Default
            lin_vel_x = [-1.25, 1.25]     # [m/s]
            lin_vel_y = [-1.25, 1.25]     # [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [rad/s]
            heading = [-3.14, 3.14]

            # 2. Stairs (forward backwards)
            # lin_vel_x = [-1.25, 1.25]   # [m/s]
            # lin_vel_y = [-0.25, 0.25]   # [m/s]
            # ang_vel_yaw = [-0.25, 0.25] # [rad/s]
            # heading = [-3.14, 3.14]

        # user_command = [1., 0., 0., 0.] # [v_x, v_y, w_yaw, heading]


    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 3.14
        
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0   # (Deprecated)
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0


    class noise( LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales( LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.1   # (Deprecated)
            dof_pos = 0.01*2
            dof_vel = 0.05*2
            ang_vel = 0.05*2
            gravity = 0.02
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            # ======================
            lin_vel_z = -1.0
            ang_vel_xy = -0.75  # orig: -0.01 
            torques = -0.00001
            dof_acc = -2.5e-7
            action_rate = -0.1 
            collision = -20.0  # orig: -10.0
            delta_torques = -1.0e-7
            # ====================== 
            contact_phase_match = 1.0
            stumble = -1.0           
            orientation = -2.5      
            dof_error = -0.04       
            hip_pos = -0.5          
            base_height = -2.5
            # stand_still_v2 = -0.1 # new         


class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'mk14_part1'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 50000
        save_interval = 100

    # recipe: 50k eps on normal terrain, ~1.5k eps on stairs mainly forwards, then ~150 eps on rehab

    # STAIRS: Usually it's ~1000-2000, but really I just train until the mean terrain level
            #   (i.e the difficulty) levels off. This usually occurs at around ~1500 episodes.

    # REHAB: 150 seems to actually be the sweet spot for harder (i.e more steep) staircases.
    #        The tradeoff seems to be between stairs performance and backwards performanceMar25_20-27-59_mk13