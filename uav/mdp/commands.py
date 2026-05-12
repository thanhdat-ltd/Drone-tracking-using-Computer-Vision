"""Custom command term cho UAV: sample target position trong world frame.

Khác với UniformPoseCommandCfg (sample trong body frame),
command này sample tọa độ tuyệt đối trong world frame (cộng env_origins).
Giống cách quadcopter_env.py (DirectRL) set _desired_pos_w.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UAVTargetPosCommand(CommandTerm):
    """Sample target xyz trong world frame và giữ cố định đến lần resample tiếp theo.

    Command tensor shape: (num_envs, 3) — [x_w, y_w, z_w] trong world frame.
    Không dùng quaternion vì UAV hover chỉ cần target vị trí.
    """

    cfg: UAVTargetPosCommandCfg

    def __init__(self, cfg: UAVTargetPosCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Command: world-frame absolute position (num_envs, 3)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 3) target position in world frame."""
        return self._target_pos_w

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample target trong world frame = env_origins + uniform offset."""
        r = torch.empty(len(env_ids), device=self.device)

        self._target_pos_w[env_ids, 0] = (
            self._env.scene.env_origins[env_ids, 0]
            + r.uniform_(*self.cfg.ranges.pos_x)
        )
        self._target_pos_w[env_ids, 1] = (
            self._env.scene.env_origins[env_ids, 1]
            + r.uniform_(*self.cfg.ranges.pos_y)
        )
        self._target_pos_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        pass  # target cố định đến lần resample — không cần update mỗi step

    def _update_metrics(self):
        asset = self._env.scene["robot"]
        pos_error = torch.norm(
            self._target_pos_w - asset.data.root_pos_w[:, :3], dim=-1
        )
        self.metrics["position_error"] = pos_error

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.01
                marker_cfg.prim_path = "/Visuals/UAV/goal_position"
                self._goal_marker = VisualizationMarkers(marker_cfg)
            self._goal_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_goal_marker"):
            self._goal_marker.visualize(self._target_pos_w)


@configclass
class UAVTargetPosCommandCfg(CommandTermCfg):
    """Config cho UAVTargetPosCommand."""

    class_type: type = UAVTargetPosCommand

    asset_name: str = "robot"

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (-2.0, 2.0)
        pos_y: tuple[float, float] = (-2.0, 2.0)
        pos_z: tuple[float, float] = (0.5, 2.0)

    ranges: Ranges = Ranges()
