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
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = 'trimesh'
        measure_heights = False
        curriculum = True
        selected = False

        terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.01,
            "downsampled_scale": 0.2,
        }

        terrain_proportions = [0.10,    # smooth slope
                               0.10,    # rough slope
                               0.45,    # stairs up
                               0.25,    # stairs down
                               0.10,    # discrete terrain
                               0.00,    # stepping stones
                               0.00]    # bumpy wave

    class domain_rand:      
        randomize_friction = True
        friction_range = [0.25, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1.1, 1.1]
        
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
    

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
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter



    # =================================================
    class commands ( LeggedRobotCfg.commands ):
        heading_command = False
        curriculum = False
        max_curriculum = 2.0
        # user_command = [1.0, 0.0, 0.0, 0.0] # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
    # =================================================


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
            # ang_vel = 0.2
            # dof_pos = 0.01
            # dof_vel = 1.5
            # gravity = 0.05
            # height_measurements = 0.1
            
            # From the hangman
            dof_pos = 0.01
            dof_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.26
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.25
            tracking_ang_vel = 0.75
            # ======================
            lin_vel_z = -1.0
            ang_vel_xy = -0.01
            torques = -0.00001
            dof_acc = -2.5e-7
            action_rate = -0.1
            collision = -10.0
            delta_torques = -1.0e-7
            # ====================== 
            # feet_air_time = 0.5
            contact_phase_match = 0.5
            stumble = -1.0          
            orientation = -2.5  # orig: -5.0
            dof_error = -0.04  # orig: -0.04
            hip_pos = -0.5      # orig: -0.5
            calf_pos = -0.05    # new
            base_height = -0.5  # new



class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'rudolf11'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 10000
        save_interval = 100