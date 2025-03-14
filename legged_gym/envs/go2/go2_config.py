from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        enable_history = True
        buffer_length = 5 # number of previous obs to keep in buffer
        num_proprio = 53 
        num_observations = (num_proprio * buffer_length if enable_history else num_proprio)
        num_envs = 4096
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = 'trimesh'
        measure_heights = False # Enable heightmap in obs
        curriculum = True
        selected = False

        terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.01,
            "downsampled_scale": 0.25,
        }

        # types: [smoothSlope, roughSlope, stairsUp, stairsDown, discrete, bumpyWave, bumpyHole], else flat
        terrain_proportions = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]


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
        default_joint_angles = {    # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,    # [rad]
            'RL_hip_joint': 0.1,    # [rad]
            'FR_hip_joint': -0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
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
        penalize_contacts_on = ["thigh", "calf", "imu"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter


    # # ============ NEVER USE WHEN TRAINING ============
    # class commands( LeggedRobotCfg.commands ):
    #     user_command = [1.5, 0.0, 0.0, 0.0] # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
    # # =================================================


    # ============== COMMAND CURRICULUM ===============
    class commands ( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 2.0
    # =================================================


    class normalization( LeggedRobotCfg.normalization ):
        clip_observations = 100.
        clip_actions = 100.
        
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_accel = 1.0 # (Deprecated)
            lin_vel = 2.0   # (Deprecated)
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0


    class noise( LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales( LeggedRobotCfg.noise.noise_scales):
            lin_accel = 0.1 # (Deprecated)
            lin_vel = 0.1   # (Deprecated)
            ang_vel = 0.2
            dof_pos = 0.01
            dof_vel = 1.5
            gravity = 0.05
            height_measurements = 0.1
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9 # +/- 90% of 50% of limit range
        base_height_target = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):

            # Rudolf 6
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001  # Reverted to original
            dof_acc = -2.5e-7
            action_rate = -0.1
            collision = -10.0
            stumble = -1.0
            feet_air_time = 0.5
            delta_torques = -1.0e-7 
            hip_pos = -1.0 
            dof_error = -0.04 
            contact_phase_match = 0.5


class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid


    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01


    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'rudolf6'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 50000
        save_interval = 1000