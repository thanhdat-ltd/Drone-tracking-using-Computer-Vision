"""UAV — Crazyflie quadcopter hover task.

Robot: Crazyflie 2.x (4 propellers)
Task:  Hover at a randomly sampled target position.

NOTE: USD file chưa có — tạm thời dùng CRAZYFLIE_CFG từ isaaclab_assets.robots.
      Thay UAV_CFG trong uav_cfg.py khi có file USD riêng.

Train (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-UAV-Hover \
        --num_envs 4096 --headless

Train (visual debug):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-UAV-Hover \
        --num_envs 64

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-UAV-Hover \
        --num_envs 4
"""

from .uav_cfg import *
from .uav_hover_env_cfg import *

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-UAV-Hover",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_hover_env_cfg:UAVHoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UAVHoverPPORunnerCfg",
    },
)
