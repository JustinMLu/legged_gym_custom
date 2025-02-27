from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 48 # 48 when mesh_type = 'plane', 235 otherwise...
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        static_friction = 1.0
        dynamic_friction = 1.0

        mesh_type = 'trimesh'
        measure_heights = False # True for rough terrain only
        curriculum = True
        selected = False

        # terrain_kwargs = {
        #     "type": "terrain_utils.wave_terrain",
        #     "num_waves": 1,
        #     "amplitude": 0.5
        # }
        # terrain_kwargs = {
        #     "type": "terrain_utils.random_uniform_terrain",
        #     "min_height": -0.08*2,
        #     "max_height": 0.08*2,
        #     "step": 0.005*2,
        #     "downsampled_scale": 0.2*2,
        # }

        # types: [smooth slope, rough slope, stairs up, stairs down, discrete, random uniform, wave]
        terrain_proportions = [0.0, 0.25, 0.20, 0.20, 0.05, 0.10, 0.20]

    class domain_rand:      
        randomize_friction = True
        friction_range = [0.3, 1.0] # try not going over 1.0

        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]
        
        push_robots = True
        push_interval_s = 30
        max_push_vel_xy = 2.0
    

    class commands( LeggedRobotCfg.commands ):
        pass
        # user_command = [0.0, 0.0, 0.0, 0.0]


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
        penalize_contacts_on = ["hip", "thigh", "calf", "imu"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter


    class rewards ( LeggedRobotCfg.rewards ):
        # From Unitree
        soft_dof_pos_limit = 0.9 # +/- 90% of 50% of limit range
        base_height_target = 0.25
        
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = -10.0
            torques = -0.0002

            # Custom
            feet_air_time = 0.5
            ang_vel_xy = -0.05
            base_height = -0.0001
            orientation = -2.5
            stand_still = -0.009
            dof_acc = -5.5e-7
            tracking_lin_vel = 1.1      # Rewards matching commanded linear velocity in XY-plane
            tracking_ang_vel = 0.6      # Rewards matching commanded yaw angular velocity


class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128*2, 64*2, 32*2]
        critic_hidden_dims = [128*2, 64*2, 32*2]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.001

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'bambot'
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 1000
        save_interval = 100