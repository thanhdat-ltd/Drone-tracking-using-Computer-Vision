"""RSL-RL PPO configuration for UAV hover task.

Tham khảo:
  - quadcopter_env (Direct, obs=12): [64,64], norm=False, lr=5e-4, iter=1000
  - drone_arl (ManagerBased):        [256,128,64], norm=False, lr=4e-4, iter=1500
  - legged_v3 (ManagerBased):        [256,256,256], norm=True, lr=1e-3, iter=5000

UAV hover (obs=20): dùng network nhỏ vừa như quadcopter, tắt norm (obs body-frame
đã bounded tự nhiên), lr và iter theo drone_arl.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UAVHoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config cho UAV hover task."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 10
    experiment_name = "uav_hover"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,   # body-frame obs đã bounded, không cần norm
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 64],
        critic_hidden_dims=[128, 64, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
