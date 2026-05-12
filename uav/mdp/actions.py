"""Custom action term cho UAV: apply thrust + moment trực tiếp lên body.

Cơ chế giống quadcopter_env.py (DirectRL):
  action[0]   → collective thrust (normalized [-1, 1])
  action[1:4] → moments roll/pitch/yaw
Applied via permanent_wrench_composer.set_forces_and_torques()
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UAVThrustAction(ActionTerm):
    """Apply collective thrust + moment lên body của UAV.

    action[0]   = collective thrust, chuẩn hóa [-1, 1]
                  → thrust_z = thrust_to_weight * weight * (a + 1) / 2
    action[1:4] = moments [roll, pitch, yaw] × moment_scale
    """

    cfg: UAVThrustActionCfg

    def __init__(self, cfg: UAVThrustActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._robot = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies(cfg.body_name)[0]

        # Tính robot weight (mass × gravity) — dùng để scale thrust
        robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(env.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (robot_mass * gravity).item()

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4  # 1 thrust + 3 moments

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def reset(self, env_ids):
        self._raw_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        # Thrust theo trục Z body frame
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight
            * (self._raw_actions[:, 0] + 1.0) / 2.0
        )
        # Moments (roll, pitch, yaw)
        self._moment[:, 0, :] = self.cfg.moment_scale * self._raw_actions[:, 1:]

    def apply_actions(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )


@configclass
class UAVThrustActionCfg(ActionTermCfg):
    """Config cho UAVThrustAction."""

    class_type: type[ActionTerm] = UAVThrustAction

    asset_name: str = MISSING
    """Tên robot asset trong scene."""

    body_name: str = "body"
    """Tên body nhận lực (mặc định: 'body' của Crazyflie)."""

    thrust_to_weight: float = 1.9
    """Tỉ lệ thrust tối đa / trọng lượng. action[0]=1 → thrust = ratio × weight."""

    moment_scale: float = 0.01
    """Scale cho moment (roll, pitch, yaw)."""
