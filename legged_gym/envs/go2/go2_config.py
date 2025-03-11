from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 53
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = 'trimesh'
        measure_heights = False # Enable heightmap in obs
        curriculum = True
        selected = False

        # terrain_kwargs = {
        #     "type": "terrain_utils.wave_terrain",
        #     "num_waves": 1,
        #     "amplitude": 0.5
        # }
        terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.08,
            "max_height": 0.08,
            "step": 0.01,
            "downsampled_scale": 0.25,
        }

        # types: [smooth slope, rough slope, stairs up, stairs down, discrete, rocky bump, bumpy wave]
        # terrain_proportions = [0.07, 0.07, 0.18, 0.18, 0.14, 0.18, 0.18]
        terrain_proportions = [0.07, 0.07, 0.0, 0.0, 0.14, 0.20, 0.20]


    class domain_rand:      
        randomize_friction = True
        friction_range = [0.3, 1.2]

        randomize_base_mass = True
        added_mass_range = [-1.1, 1.1]
        
        push_robots = True
        push_interval_s = 30
        max_push_vel_xy = 2.5
    

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
        # PD Drive prameters:
        control_type = 'P'          # Position control 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}    # [N*m*s/rad]
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
    #     user_command = [0.75, 0.0, 0.0, 0.0] # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
    # # =================================================

    # # ============== COMMAND CURRICULUM ===============
    # class commands ( LeggedRobotCfg.commands ):
    #     curriculum = True
    #     max_curriculum = 1.5
    # # =================================================

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_accel = 1.0 # (NEW)
    

    class noise( LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales):
            lin_accel = 0.1 # (NEW)


    class rewards( LeggedRobotCfg.rewards ):
        # From Unitree
        soft_dof_pos_limit = 0.9 # +/- 90% of 50% of limit range
        base_height_target = 0.25 # 0.25 original
        
        class scales( LeggedRobotCfg.rewards.scales ):

            # Custom
            # tracking_lin_vel = 1.5
            # tracking_ang_vel = 1.0
            # feet_air_time = 0.5
            # ang_vel_xy = -0.05
            # base_height = -0.0001
            # orientation = -2.5
            # stand_still = -0.009
            # dof_acc = -5.5e-7
            # delta_torques = -1.0e-7 # New
            # hip_pos = -0.5 # New
            # dof_error = -0.04 # New
            # contact_phase_match = 0.6 # New
            # foot_swing_height = 0.5 # New


            # Extreme Parkour
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -5.0 # -1.0 original
            torques = -0.0002  # -0.00001 original
            dof_acc = -2.5e-7
            action_rate = -0.1
            collision = -10.0
            stumble = -1.0
            feet_air_time = 0.5
            delta_torques = -1.0e-7     # New
            hip_pos = -1.0              # New (was -0.5)
            dof_error = -0.04           # New
            contact_phase_match = 0.5   # New
            # foot_swing_height = 0.5   # New


class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128*4, 64*4, 32*4]
        critic_hidden_dims = [128*4, 64*4, 32*4]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'rudolf'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 50000
        save_interval = 1000