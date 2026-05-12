"""Custom reward functions for UAV (Crazyflie quadcopter).

Adapted from isaaclab_tasks/manager_based/drone_arl/mdp/rewards.py
"""
from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward khi UAV tiến gần vị trí mục tiêu (exp-kernel).

    Cả target_pos và current_pos đều trong world frame (UAVTargetPosCommand).

    Returns:
        Tensor (num_envs,) trong khoảng (0, 1]. = 1.0 khi distance = 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_command(command_name)[:, :3]      # world frame (num_envs, 3)
    current_pos = asset.data.root_pos_w                                    # world frame (num_envs, 3)

    distance_sq = torch.sum(torch.square(target_pos - current_pos), dim=-1)
    return torch.exp(-distance_sq / (std ** 2))


def distance_to_goal_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward khoảng cách đến target dùng tanh — giống quadcopter DirectRL.

    Công thức: 1 - tanh(d / scale). = 1 khi d=0, giảm dần khi xa.
    Dùng với weight dương.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    distance = torch.linalg.norm(target_pos - asset.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(distance / scale)


def distance_to_goal_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L1 khoảng cách đến target (world frame).

    Trả về |Δx| + |Δy| + |Δz|. Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    return scale * torch.sum(torch.abs(target_pos - asset.data.root_pos_w), dim=-1)

def distance_to_goal_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L2 khoảng cách Euclidean đến target (world frame).

    Trả về ||target - pos||₂ (không bình phương) → gradient không vanish khi gần target.
    Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(target_pos - asset.data.root_pos_w, dim=-1)


def ang_vel_xyz_exp(
    env: ManagerBasedRLEnv,
    std: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward giảm vận tốc góc (exp-kernel). = 1 khi omega = 0.

    Khuyến khích UAV không xoay — cần thiết để hover ổn định.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return torch.exp(-ang_vel_sq / (std ** 2))


def lin_vel_xyz_exp(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward giảm vận tốc tịnh tiến (exp-kernel). = 1 khi v = 0.

    Khuyến khích UAV giữ yên khi đã đến target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=-1)
    return torch.exp(-lin_vel_sq / (std ** 2))


def yaw_aligned(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward UAV giữ yaw = 0 (exp-kernel). = 1 khi yaw = 0.

    Crazyflie cần counter-torque từ cặp propellers — yaw không bằng 0
    khi thrust mất cân bằng → reward này giúp ổn định yaw.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = math_utils.wrap_to_pi(yaw)
    return torch.exp(-(yaw ** 2) / (std ** 2))


def lin_vel_xyz_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L1 cho toàn bộ vận tốc tịnh tiến (x + y + z).

    Trả về ||v||₂ (không bình phương) → gradient không vanish khi v nhỏ.
    Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_lin_vel_b, dim=-1)


def lin_vel_xyz_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L2-squared cho vận tốc tịnh tiến — giống quadcopter DirectRL.

    Trả về sum(v²) = ||v||²₂. Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)


def ang_vel_xyz_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L2-squared cho vận tốc góc — giống quadcopter DirectRL.

    Trả về sum(ω²) = ||ω||²₂. Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)


def action_rate_l1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty L1 tốc độ thay đổi action (không bình phương). Dùng với weight âm."""
    return torch.norm(
        env.action_manager.action - env.action_manager.prev_action, dim=-1
    )


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty L2 tốc độ thay đổi action (bình phương). Dùng với weight âm."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
    )


def ang_vel_xyz_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L1 cho toàn bộ vận tốc góc (x + y + z).

    Trả về ||omega||₂ (không bình phương). Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_ang_vel_b, dim=-1)


def rpy_alignment(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance: float = 0.05,
    scale: float = 3.0,
    axis_weights: tuple[float, float, float] = (2.0, 2.0, 0.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward RPY alignment về target_rpy — adapted từ legged_v3 rpy_alignment_imu.

    Dùng root_quat_w thay IMU. Trả về [0, 1], dùng với weight dương.

    Args:
        target_rpy:    (roll, pitch, yaw) mục tiêu [rad]. Mặc định (0,0,0) = thẳng đứng.
        tolerance:     dead zone [rad] — không phạt sai số nhỏ hơn ngưỡng này.
        scale:         hệ số scale exp kernel (lớn hơn = phạt chặt hơn).
        axis_weights:  (w_roll, w_pitch, w_yaw) — roll/pitch thường quan trọng hơn yaw.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    roll  = torch.clamp(roll,  -math.pi, math.pi)
    pitch = torch.clamp(pitch, -math.pi, math.pi)
    yaw   = torch.clamp(yaw,   -math.pi, math.pi)

    t_roll, t_pitch, t_yaw = target_rpy
    roll_err  = torch.clamp(torch.abs(math_utils.wrap_to_pi(roll  - t_roll))  - tolerance, min=0.0)
    pitch_err = torch.clamp(torch.abs(math_utils.wrap_to_pi(pitch - t_pitch)) - tolerance, min=0.0)
    yaw_err   = torch.clamp(torch.abs(math_utils.wrap_to_pi(yaw   - t_yaw))   - tolerance, min=0.0)

    w0, w1, w2 = axis_weights
    weighted_err = (w0 * roll_err**2 + w1 * pitch_err**2 + w2 * yaw_err**2) / (w0 + w1 + w2)

    return torch.clamp(torch.exp(-scale * weighted_err), 0.0, 1.0)


def flat_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty L1 cho roll/pitch (giữ UAV thẳng đứng).

    = ||projected_gravity_xy||₂ → 0 khi thẳng đứng. Dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
