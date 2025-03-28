from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RnnCfg( LeggedRobotCfg ):

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
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.005,
            "downsampled_scale": 0.2
        }

        curriculum = True
        terrain_default     = [0.20,    # smooth slope
                               0.20,    # rough slope
                               0.10,    # stairs up
                               0.15,    # stairs down
                               0.25,    # discrete terrain
                               0.00,    # stepping stones
                               0.10]    # random uniform
        
        terrain_stairs      = [0.00,    # smooth slope
                               0.00,    # rough slope
                               0.75,    # stairs up
                               0.25,    # stairs down
                               0.00,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # random unifrom
        
        terrain_proportions = terrain_default

    class domain_rand:      
        randomize_friction = True
        friction_range = [0.1, 1.0]

        randomize_base_mass = True
        added_mass_range = [-1.1, 1.1]
        
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.5
    

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
        penalize_contacts_on = ["thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter



    class commands ( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 10.
        
        curriculum = False
        max_curriculum = 1.25
        
        class ranges:
            # Default
            lin_vel_x = [-1.0, 1.0]         # [m/s]
            lin_vel_y = [-1.0, 1.0]         # [m/s]
            ang_vel_yaw = [-1.0, 1.0]     # [rad/s]
            heading = [-3.14, 3.14]

            # Stairs (forward backwards)
            # lin_vel_x = [-1.25, 1.25]   # [m/s]
            # lin_vel_y = [-0.25, 0.25]   # [m/s]
            # ang_vel_yaw = [-0.25, 0.25] # [rad/s]
            # heading = [-3.14, 3.14]
        
        #               [vx, vy, yaw, heading]
        # user_command = [1., 0., 0., 0.]


    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 10.0 # NEW FOR MK5
        
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
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.26 # originally 0.26
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            # ======================
            lin_vel_z = -1.0
            ang_vel_xy = -0.01  # -0.01 for base policy, -0.75 for stairs
            torques = -0.00001
            dof_acc = -2.5e-7
            action_rate = -0.1 
            collision = -10.0   # -10.0 for base policy, -20.0 for stairs
            delta_torques = -1.0e-7
            # ====================== 
            contact_phase_match = 1.0
            stumble = -1.0           
            orientation = -2.5      
            dof_error = -0.04   # orig: -0.04      
            hip_pos = -0.5          
            base_height = -2.5  # orig: -2.5 


class Go2RnnCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'mk15_default'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 10000
        save_interval = 100

    # recipe: 50k eps on normal terrain, ~1.5-2k eps on stairs (constraining fwd and backwards) 