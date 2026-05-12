"""Custom observation functions cho UAV."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def desired_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vị trí mục tiêu trong body frame của UAV (3 dims).

    Giống quadcopter_env: subtract_frame_transforms(root_pos_w, root_quat_w, target_w)
    UniformPoseCommand lưu tọa độ relative to env_origins → cộng lại để ra world frame.

    Returns:
        Tensor (num_envs, 3) — target position in robot body frame.
    """
    asset = env.scene[asset_cfg.name]

    # UAVTargetPosCommand trả về world frame trực tiếp
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]

    desired_pos_b_, _ = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        target_pos_w,
    )
    return desired_pos_b_
