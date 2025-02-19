from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        # num_observations = 48 # 48 when mesh_type = 'plane', 235 otherwise...
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        measure_heights = True # True for rough terrain only
        curriculum = False
        selected = True
        # terrain_kwargs = {
        #     "type": "terrain_utils.random_uniform_terrain",
        #     "min_height": -0.025,
        #     "max_height": 0.025,
        #     "step": 0.01,
        #     "downsampled_scale": 0.1,
        # }
        terrain_kwargs = {
            "type": "terrain_utils.wave_terrain",
            "num_waves": 1,
            "amplitude": 0.75
        }

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]      # [x, y, z] (metres)
        default_joint_angles = {    # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,    # [rad]

            'RL_hip_joint': 0.1,    # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
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
        action_scale = 0.5
        decimation = 4


    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "imu"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards ( LeggedRobotCfg.rewards ):
        # From Unitree
        soft_dof_pos_limit = 0.9 # +/- 90% of 50% of limit range
        base_height_target = 0.25

        # Added by me
        soft_dof_vel_limit = 0.8 # +/- 90% of 50% of limit range
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # From Unitree
            torques = -0.0002
            dof_pos_limits = -10.0
            
            # Added by me
            tracking_lin_vel = 1.0  # Reward for tracking commanded velocity
            lin_vel_z = -2.0        # Penalize vertical movement
            ang_vel_xy = -0.2       # Penalize angular velocity in xy plane
            # base_height = -0.5      # Penalize deviation from target height (terrain-aware)
            orientation = -2.0      # Penalize non-flat orientation
            collision = -3.0        # Penalize collisions on target parts
            feet_air_time = 1.0     # Reward for taking steps
            
            # Even newer stuff
            forward_vel = 10.0      # Strong reward for moving forward

    
    class commands (LeggedRobotCfg.commands ):
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            pass
            # lin_vel_x = [0.5, 1.0] # min max sample ranges [m/s]
            # lin_vel_y = [0.0, 0.0]
            # ang_vel_yaw = [0.0, 0.0]
            # heading = [-0.52, 0.52] # -30 to 30 degrees


class Go2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2'
        load_run = -1
        max_iterations = 5000
        save_interval = 50