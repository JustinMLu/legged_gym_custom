from legged_gym.envs.go2.go2_parkour_config import Go2ParkourCfg, Go2ParkourCfgPPO

class Go2FinetuneCfg( Go2ParkourCfg ):
    
    class terrain( Go2ParkourCfg.terrain ):
        parkour = True
        curriculum = False # Turn off curriculum for finetuning
        # ====================== Jump Finetuning ======================
        gap_heights = [-2.0, 0.10, -2.0,
                       -2.0, 0.15, -2.0,
                       -2.0, 0.20, -2.0,
                       -2.0, 0.25, -2.0,
                       -2.0, 0.30, -2.0,
                       -2.0, 0.35, -2.0]
        
        gap_lengths = [0.3, 0.3, 0.3] * 6

        obstacle_x_positions = [6.0, 6.3, 6.6,
                                10.0, 10.3, 10.6,
                                14.0, 14.3, 14.6,
                                18.0, 18.3, 18.6,
                                22.0, 22.3, 22.6,
                                26.0, 26.3, 26.6]
        
        obstacle_y_positions = [0.0, 0.0, 0.0] * 6

        parkour_kwargs = {
            "start_platform_length": 3.,
            "start_platform_height": 0.,
    
            "x_positions": obstacle_x_positions,
            "y_positions": obstacle_y_positions,  # (-) right, (+) left
            
            "obstacle_heights": gap_heights,  
            "obstacle_lengths": gap_lengths,             

            "half_valid_width": 5.0,
            "border_width": 0.50,
            "border_height": -2.0,
        }
        # =============================================================
    
    class commands( Go2ParkourCfg.commands ):
        class ranges( Go2ParkourCfg.commands.ranges ):
            lin_vel_x = [0.5, 2.0]    # Increased ranges


class Go2FinetuneCfgPPO( Go2ParkourCfgPPO ):
    class runner( Go2ParkourCfgPPO.runner ):
        run_name = 'parkour_finetune'
        experiment_name = 'go2_parkour' # same as the base parkour cfg
        resume = True