from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO

from .go2.go2 import Go2Robot
from .go2.go2_config import Go2Cfg, Go2CfgPPO
from .go2.go2_parkour_config import Go2ParkourCfg, Go2ParkourCfgPPO
from .go2.go2_parkour_finetune_config import Go2FinetuneCfg, Go2FinetuneCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

# Task name string is used as a CLI argument to select the task, experiment name (in cfg) defines folder name
task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go2", Go2Robot, Go2Cfg(), Go2CfgPPO() )
task_registry.register( "go2_parkour", Go2Robot, Go2ParkourCfg(), Go2ParkourCfgPPO() )
task_registry.register( "go2_parkour_finetune", Go2Robot, Go2FinetuneCfg(), Go2FinetuneCfgPPO() )