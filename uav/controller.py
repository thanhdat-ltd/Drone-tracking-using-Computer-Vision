# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UAV controller helpers — dùng chung cho tất cả các script (open-loop, PID, RL)."""

from __future__ import annotations

import math
import os
import sys
import torch

# Tìm pid_controller.py trong thư mục pid_control cạnh file này
_PID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_control")
if _PID_DIR not in sys.path:
    sys.path.insert(0, _PID_DIR)

from pid_controller import PIDController, wrap_angle  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils


# ── Camera config (OV2640) ────────────────────────────────────────────────────

def make_front_camera_cfg(
    prim_path: str = "/World/Crazyflie/body/camera_front",
    fps: float = 60.0,
    width: int = 640,
    height: int = 480,
) -> CameraCfg:
    """CameraCfg cho camera gắn mũi drone, mô phỏng OV2640.

    OV2640 specs: sensor 1/4" (3.6×2.7 mm), lens 2.8 mm → HFOV ≈ 65°.
    Orientation: forward-facing, ROS convention (x-right, y-down, z-forward).
    """
    return CameraCfg(
        prim_path=prim_path,
        update_period=1.0 / fps,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            horizontal_aperture=3.6,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


# ── Hàm tiện ích apply lực ────────────────────────────────────────────────────

def apply_prop_wrench(
    robot,
    prop_body_ids,
    thrust_per_prop: "float | torch.Tensor",
    prop_spin_dirs: "torch.Tensor",
    spin_torque: float,
    root_body_ids=None,
    net_yaw: float = 0.0,
) -> None:
    """Áp lực và moment lên 4 props + (tuỳ chọn) yaw torque lên root body.

    Args:
        robot: Isaac Lab Articulation object.
        prop_body_ids: body IDs của 4 props (từ robot.find_bodies).
        thrust_per_prop: lực nâng mỗi prop [N]. Scalar hoặc tensor shape (4,).
        prop_spin_dirs: chiều quay mỗi prop, tensor shape (4,), giá trị +1/-1.
        spin_torque: moment quay mỗi prop [Nm].
        root_body_ids: body IDs của root body. Bắt buộc nếu net_yaw != 0.
        net_yaw: yaw torque áp lên root body [Nm].
    """
    device = prop_spin_dirs.device
    n = robot.num_instances

    forces  = torch.zeros(n, 4, 3, device=device)
    torques = torch.zeros(n, 4, 3, device=device)
    forces[..., 2]  = thrust_per_prop
    torques[..., 2] = prop_spin_dirs * spin_torque

    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=prop_body_ids,
    )

    if root_body_ids is not None:
        yaw_t = torch.tensor([[0.0, 0.0, net_yaw]], device=device).expand(n, -1).unsqueeze(1)
        robot.set_external_force_and_torque(
            forces=torch.zeros_like(yaw_t),
            torques=yaw_t,
            body_ids=root_body_ids,
        )


# ── Bộ điều khiển altitude + attitude (1 tầng) ───────────────────────────────
class AltitudeAttitudeController:
    """PID điều khiển độ cao (Z) và giữ thăng bằng (roll, pitch, yaw).

    Luồng:
        z_err   → pid_z     → thrust [N]
        roll    → pid_roll  → m_roll  [Nm]
        pitch   → pid_pitch → m_pitch [Nm]
        yaw_err → pid_yaw   → m_yaw   [Nm]
        [thrust, m_roll, m_pitch, 0] @ A_inv → F_props → apply_prop_wrench
    """

    def __init__(
        self,
        robot,
        prop_body_ids,
        root_body_ids,
        A_inv: torch.Tensor,
        z_kp: float = 1.5,   z_ki: float = 0.7,   z_kd: float = 0.12,  z_lim: float = 1.0,
        att_kp: float = 0.01, att_ki: float = 0.0, att_kd: float = 0.02, att_lim: float = 0.01,
        yaw_kp: float = 0.01, yaw_ki: float = 0.0, yaw_kd: float = 0.01, yaw_lim: float = 0.01,
        max_yaw_moment: float = 0.0015,
    ):
        self.robot          = robot
        self.prop_body_ids  = prop_body_ids
        self.root_body_ids  = root_body_ids
        self.A_inv          = A_inv
        self._dev           = A_inv.device
        self.max_yaw_moment = max_yaw_moment

        self.pid_z     = PIDController(z_kp,   z_ki,   z_kd,   integral_limit=z_lim)
        self.pid_roll  = PIDController(att_kp,  att_ki, att_kd, integral_limit=att_lim)
        self.pid_pitch = PIDController(att_kp,  att_ki, att_kd, integral_limit=att_lim)
        self.pid_yaw   = PIDController(yaw_kp,  yaw_ki, yaw_kd, integral_limit=yaw_lim)

        self._zero_spin = torch.zeros(4, device=self._dev)

    def step(self, dt: float, target_z: float, target_yaw: float = 0.0) -> dict:
        pos  = self.robot.data.root_pos_w[0]
        quat = self.robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        z_err  = target_z - pos[2].item()
        thrust = max(0.0, self.pid_z.update(z_err, dt))

        m_roll  = self.pid_roll.update(-roll,  dt)
        m_pitch = self.pid_pitch.update(-pitch, dt)
        yaw_err = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        m_yaw   = max(-self.max_yaw_moment,
                      min(self.max_yaw_moment, self.pid_yaw.update(yaw_err, dt)))

        wrench  = torch.tensor([thrust, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )

        return {
            "z": pos[2].item(), "z_err": z_err, "thrust": thrust,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "m_roll": m_roll, "m_pitch": m_pitch, "m_yaw": m_yaw,
        }

    def reset(self) -> None:
        for c in (self.pid_z, self.pid_roll, self.pid_pitch, self.pid_yaw):
            c.reset()

    def get_state(self) -> dict:
        pos  = self.robot.data.root_pos_w[0]
        vel  = self.robot.data.root_lin_vel_w[0]
        quat = self.robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        return {"pos": pos, "vel": vel, "roll": roll, "pitch": pitch, "yaw": yaw}


# ── Bộ điều khiển rate (vòng trong cùng) ─────────────────────────────────────

class RateController:
    """PID điều khiển tốc độ góc (p, q, r) → moments → motor forces.

    Vòng TRONG CÙNG của cascade. Tune file này trước hết.

    Luồng:
        rate_err [p,q,r] → pid_roll/pitch/yaw → moments [Nm]
        [thrust, m_roll, m_pitch, 0] @ A_inv → F_props → apply_prop_wrench
    """

    def __init__(
        self,
        robot,
        prop_body_ids,
        root_body_ids,
        A_inv: torch.Tensor,
        hover_thrust: float = 0.35,
        roll_kp:  float = 0.0006, roll_ki:  float = 0.0001, roll_kd:  float = 0.00001, roll_lim:  float = 0.5,
        pitch_kp: float = 0.0006, pitch_ki: float = 0.0001, pitch_kd: float = 0.00001, pitch_lim: float = 0.5,
        yaw_kp:   float = 0.0006, yaw_ki:   float = 0.00005, yaw_kd:  float = 0.0,     yaw_lim:   float = 0.1,
        max_moment:     float = 0.03,
        max_yaw_moment: float = 0.0003,
    ):
        self.robot          = robot
        self.prop_body_ids  = prop_body_ids
        self.root_body_ids  = root_body_ids
        self.A_inv          = A_inv
        self._dev           = A_inv.device
        self.hover_thrust   = hover_thrust
        self.max_moment     = max_moment
        self.max_yaw_moment = max_yaw_moment

        self.pid_roll  = PIDController(roll_kp,  roll_ki,  roll_kd,  integral_limit=roll_lim)
        self.pid_pitch = PIDController(pitch_kp, pitch_ki, pitch_kd, integral_limit=pitch_lim)
        self.pid_yaw   = PIDController(yaw_kp,   yaw_ki,   yaw_kd,   integral_limit=yaw_lim)

        self._zero_spin = torch.zeros(4, device=self._dev)

    def step(
        self,
        dt: float,
        p_des: float = 0.0,
        q_des: float = 0.0,
        r_des: float = 0.0,
        thrust: float | None = None,
    ) -> dict:
        ang_vel = self.robot.data.root_ang_vel_w[0]
        p, q, r = ang_vel[0].item(), ang_vel[1].item(), ang_vel[2].item()

        m_roll  = self.pid_roll.update(p_des - p, dt)
        m_pitch = self.pid_pitch.update(q_des - q, dt)
        m_yaw   = self.pid_yaw.update(r_des - r,  dt)

        m_roll  = max(-self.max_moment,     min(self.max_moment,     m_roll))
        m_pitch = max(-self.max_moment,     min(self.max_moment,     m_pitch))
        m_yaw   = max(-self.max_yaw_moment, min(self.max_yaw_moment, m_yaw))

        t = thrust if thrust is not None else self.hover_thrust
        wrench  = torch.tensor([t, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )

        return {"p": p, "q": q, "r": r, "m_roll": m_roll, "m_pitch": m_pitch, "m_yaw": m_yaw}

    def reset(self) -> None:
        for c in (self.pid_roll, self.pid_pitch, self.pid_yaw):
            c.reset()

# ── Bộ điều khiển attitude cascade 2 tầng ────────────────────────────────────

class AttitudeController:
    """Cascade 2 tầng: angle → rate → moment (kiểu Betaflight/FPV thực tế).

    Tầng ngoài (attitude): angle_error → rate_setpoint  [rad/s per rad]
    Tầng trong (rate):     rate_error  → moment          [Nm per rad/s]

    Luồng:
        angle_err → [PID_att] → rate_des  (rad/s)
        rate_des - rate_actual → [PID_rate] → moment (Nm)
        [thrust, m_roll, m_pitch, 0] @ A_inv → F_props → apply_prop_wrench
        m_yaw apply thẳng lên root body

    Args:
        att_kp/ki/kd: gains tầng ngoài (angle → rate).
        rate_kp/ki/kd/lim: gains tầng trong (rate → moment) — từ RateController.
        max_rate: clamp rate setpoint roll/pitch [rad/s].
        max_yaw_rate: clamp rate setpoint yaw [rad/s].
    """

    def __init__(
        self,
        robot,
        prop_body_ids,
        root_body_ids,
        A_inv: torch.Tensor,
        hover_thrust:    float = 0.35,
        # Tầng ngoài — angle → rate_des
        att_kp:      float = 0.5,   att_ki:      float = 0.0,    att_kd:      float = 0.0,
        yaw_att_kp:  float = 1.0,    yaw_att_ki:  float = 0.0,    yaw_att_kd:  float = 0.0,
        # Tầng trong — rate → moment (từ file 02)
        rate_kp:     float = 0.0009, rate_ki:     float = 0.0001, rate_kd:     float = 0.00001, rate_lim:     float = 0.5,
        yaw_rate_kp: float = 0.0009, yaw_rate_ki: float = 0.00005, yaw_rate_kd: float = 0.0,   yaw_rate_lim: float = 0.1,
        # Giới hạn
        max_rate:       float = math.radians(180.0),
        max_yaw_rate:   float = math.radians(90.0),
        max_moment:     float = 0.03,
        max_yaw_moment: float = 0.0003,
    ):
        self.robot          = robot
        self.prop_body_ids  = prop_body_ids
        self.root_body_ids  = root_body_ids
        self.A_inv          = A_inv
        self._dev           = A_inv.device
        self.hover_thrust   = hover_thrust
        self.max_rate       = max_rate
        self.max_yaw_rate   = max_yaw_rate
        self.max_moment     = max_moment
        self.max_yaw_moment = max_yaw_moment

        # Tầng ngoài
        self.pid_roll_att  = PIDController(att_kp,     att_ki,     att_kd)
        self.pid_pitch_att = PIDController(att_kp,     att_ki,     att_kd)
        self.pid_yaw_att   = PIDController(yaw_att_kp, yaw_att_ki, yaw_att_kd)

        # Tầng trong
        self.pid_roll_rate  = PIDController(rate_kp,     rate_ki,     rate_kd,     integral_limit=rate_lim)
        self.pid_pitch_rate = PIDController(rate_kp,     rate_ki,     rate_kd,     integral_limit=rate_lim)
        self.pid_yaw_rate   = PIDController(yaw_rate_kp, yaw_rate_ki, yaw_rate_kd, integral_limit=yaw_rate_lim)

        self._zero_spin = torch.zeros(4, device=self._dev)

    def step(
        self,
        dt: float,
        roll_d:  float = 0.0,
        pitch_d: float = 0.0,
        yaw_sp:  float = 0.0,
        thrust:  float | None = None,
    ) -> dict:
        """Một bước cascade: angle → rate → moment → lực motor.

        Returns:
            dict với key: roll, pitch, yaw, p, q, r, p_des, q_des, r_des,
                          m_roll, m_pitch, m_yaw.
        """
        quat      = self.robot.data.root_quat_w[0]
        ang_vel_b = self.robot.data.root_ang_vel_b[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        p, q, r = ang_vel_b[0].item(), ang_vel_b[1].item(), ang_vel_b[2].item()

        # Tầng ngoài: angle → desired rate
        p_des = self.pid_roll_att.update(roll_d  - roll,            dt)
        q_des = self.pid_pitch_att.update(pitch_d - pitch,           dt)
        r_des = self.pid_yaw_att.update(wrap_angle(yaw_sp - yaw),   dt)

        p_des = max(-self.max_rate,     min(self.max_rate,     p_des))
        q_des = max(-self.max_rate,     min(self.max_rate,     q_des))
        r_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, r_des))

        # Tầng trong: rate → moment
        m_roll  = self.pid_roll_rate.update(p_des - p, dt)
        m_pitch = self.pid_pitch_rate.update(q_des - q, dt)
        m_yaw   = self.pid_yaw_rate.update(r_des  - r, dt)

        m_roll  = max(-self.max_moment,     min(self.max_moment,     m_roll))
        m_pitch = max(-self.max_moment,     min(self.max_moment,     m_pitch))
        m_yaw   = max(-self.max_yaw_moment, min(self.max_yaw_moment, m_yaw))

        t       = thrust if thrust is not None else self.hover_thrust
        wrench  = torch.tensor([t, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )

        return {
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "p": p, "q": q, "r": r,
            "p_des": p_des, "q_des": q_des, "r_des": r_des,
            "m_roll": m_roll, "m_pitch": m_pitch, "m_yaw": m_yaw,
        }

    def step_outer(self, dt: float, roll_d: float, pitch_d: float, yaw_sp: float) -> tuple:
        """Tầng ngoài: angle → rate_des. Dùng cho multi-rate ZOH."""
        quat = self.robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        p_des = self.pid_roll_att.update(roll_d  - roll,          dt)
        q_des = self.pid_pitch_att.update(pitch_d - pitch,         dt)
        r_des = self.pid_yaw_att.update(wrap_angle(yaw_sp - yaw), dt)

        p_des = max(-self.max_rate,     min(self.max_rate,     p_des))
        q_des = max(-self.max_rate,     min(self.max_rate,     q_des))
        r_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, r_des))
        return p_des, q_des, r_des

    def step_inner(self, dt: float, p_des: float, q_des: float, r_des: float,
                   thrust: float | None = None) -> dict:
        """Tầng trong: rate_des (ZOH) → moment → lực motor. Dùng cho multi-rate ZOH."""
        quat      = self.robot.data.root_quat_w[0]
        ang_vel_b = self.robot.data.root_ang_vel_b[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        p, q, r = ang_vel_b[0].item(), ang_vel_b[1].item(), ang_vel_b[2].item()

        m_roll  = self.pid_roll_rate.update(p_des - p, dt)
        m_pitch = self.pid_pitch_rate.update(q_des - q, dt)
        m_yaw   = self.pid_yaw_rate.update(r_des  - r, dt)

        m_roll  = max(-self.max_moment,     min(self.max_moment,     m_roll))
        m_pitch = max(-self.max_moment,     min(self.max_moment,     m_pitch))
        m_yaw   = max(-self.max_yaw_moment, min(self.max_yaw_moment, m_yaw))

        t       = thrust if thrust is not None else self.hover_thrust
        wrench  = torch.tensor([t, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )
        return {"roll": roll, "pitch": pitch, "yaw": yaw,
                "p": p, "q": q, "r": r,
                "m_roll": m_roll, "m_pitch": m_pitch, "m_yaw": m_yaw}

    def reset(self) -> None:
        for c in (self.pid_roll_att, self.pid_pitch_att, self.pid_yaw_att,
                  self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate):
            c.reset()
class VelocityController:
    """3 vòng: vel_x/y/z → angle_des + thrust → rate_des → moment.

    Luồng:
        vel_z_err   → pid_vz  → thrust [N]
        vel_x/y_err → pid_vxy → roll_des / pitch_des [rad]  (clamp MAX_TILT)
        angle_err   → pid_att → rate_des [rad/s]             (tầng giữa)
        rate_err    → pid_rate → moment [Nm]                 (tầng trong)

    Args:
        thrust_base: hover thrust cộng thêm vào output pid_vz [N].
        max_tilt: giới hạn góc nghiêng từ vel_x/y [rad].
    """

    def __init__(
        self,
        robot,
        prop_body_ids,
        root_body_ids,
        A_inv: torch.Tensor,
        # Vel xy — vxy → angle_des  (thrust đến từ altitude PID bên ngoài)
        vxy_kp: float = 0.5,   vxy_ki: float = 0.05,  vxy_kd: float = 0.2,  vxy_lim: float = 0.5,
        max_tilt: float = math.radians(20.0),
        # Att middle — angle → rate_des
        att_kp:     float = 2.0,  att_ki:     float = 0.0, att_kd:     float = 0.0,
        yaw_att_kp: float = 1.0,  yaw_att_ki: float = 0.0, yaw_att_kd: float = 0.0,
        # Rate inner — từ file 02
        rate_kp:     float = 0.0006, rate_ki:     float = 0.0001, rate_kd:     float = 0.00001, rate_lim:     float = 0.5,
        yaw_rate_kp: float = 0.0006, yaw_rate_ki: float = 0.00005, yaw_rate_kd: float = 0.0,   yaw_rate_lim: float = 0.1,
        max_rate:       float = math.radians(180.0),
        max_yaw_rate:   float = math.radians(90.0),
        max_moment:     float = 0.03,
        max_yaw_moment: float = 0.0003,
    ):
        self.robot         = robot
        self.prop_body_ids = prop_body_ids
        self.root_body_ids = root_body_ids
        self.A_inv         = A_inv
        self._dev          = A_inv.device
        self.max_tilt      = max_tilt
        self.max_rate      = max_rate
        self.max_yaw_rate  = max_yaw_rate
        self.max_moment    = max_moment
        self.max_yaw_moment = max_yaw_moment

        # Vel xy
        self.pid_vx  = PIDController(vxy_kp, vxy_ki, vxy_kd, integral_limit=vxy_lim)
        self.pid_vy  = PIDController(vxy_kp, vxy_ki, vxy_kd, integral_limit=vxy_lim)

        # Att middle — có integral_limit để chống wind‑up
        self.pid_roll_att  = PIDController(att_kp, att_ki, att_kd, integral_limit=math.radians(30.0))
        self.pid_pitch_att = PIDController(att_kp, att_ki, att_kd, integral_limit=math.radians(30.0))
        self.pid_yaw_att   = PIDController(yaw_att_kp, yaw_att_ki, yaw_att_kd, integral_limit=math.radians(45.0))

        # Rate inner
        self.pid_roll_rate  = PIDController(rate_kp,     rate_ki,     rate_kd,     integral_limit=rate_lim)
        self.pid_pitch_rate = PIDController(rate_kp,     rate_ki,     rate_kd,     integral_limit=rate_lim)
        self.pid_yaw_rate   = PIDController(yaw_rate_kp, yaw_rate_ki, yaw_rate_kd, integral_limit=yaw_rate_lim)

        self._zero_spin = torch.zeros(4, device=self._dev)

    def step(
        self,
        dt: float,
        target_vx: float = 0.0,
        target_vy: float = 0.0,
        thrust:    float = 0.0,
        yaw_sp:    float = 0.0,
    ) -> dict:
        """Một bước: vxy → angle → rate → moment. Thrust đến từ altitude PID bên ngoài.

        Returns:
            dict với key: vx, vy, vz, roll, pitch, yaw, roll_des, pitch_des,
                          p_des, q_des, r_des, thrust.
        """
        vel_w     = self.robot.data.root_lin_vel_w[0]
        quat      = self.robot.data.root_quat_w[0]
        ang_vel_b = self.robot.data.root_ang_vel_b[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        vx, vy, vz = vel_w[0].item(), vel_w[1].item(), vel_w[2].item()
        p, q, r    = ang_vel_b[0].item(), ang_vel_b[1].item(), ang_vel_b[2].item()

        # Vòng vel: vxy → angle_des  (thrust từ altitude PID ngoài)
        # Negate roll: positive roll tilts thrust in -Y → must invert to get +vy
        roll_des  = max(-self.max_tilt, min(self.max_tilt, -self.pid_vy.update(target_vy - vy, dt)))
        pitch_des = max(-self.max_tilt, min(self.max_tilt,  self.pid_vx.update(target_vx - vx, dt)))

        # Vòng att: angle → rate_des
        p_des = max(-self.max_rate,     min(self.max_rate,     self.pid_roll_att.update(roll_des  - roll,          dt)))
        q_des = max(-self.max_rate,     min(self.max_rate,     self.pid_pitch_att.update(pitch_des - pitch,         dt)))
        r_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.pid_yaw_att.update(wrap_angle(yaw_sp - yaw),   dt)))

        # Vòng rate: rate_des → moment
        m_roll  = max(-self.max_moment,     min(self.max_moment,     self.pid_roll_rate.update(p_des - p, dt)))
        m_pitch = max(-self.max_moment,     min(self.max_moment,     self.pid_pitch_rate.update(q_des - q, dt)))
        m_yaw   = max(-self.max_yaw_moment, min(self.max_yaw_moment, self.pid_yaw_rate.update(r_des  - r, dt)))

        wrench  = torch.tensor([thrust, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )

        return {
            "vx": vx, "vy": vy, "vz": vz,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "roll_des": roll_des, "pitch_des": pitch_des,
            "p_des": p_des, "q_des": q_des, "r_des": r_des,
            "thrust": thrust,
        }

    # ── Multi-rate ZOH methods ────────────────────────────────────────────────

    def step_vel(self, dt: float, target_vx: float, target_vy: float) -> tuple[float, float]:
        """Tầng vel (chậm nhất): vxy → angle_des.

        Returns:
            (roll_des, pitch_des) — giữ ZOH đến lần cập nhật tiếp.
        """
        vel_w = self.robot.data.root_lin_vel_w[0]
        vy, vx = vel_w[1].item(), vel_w[0].item()

        roll_des  = max(-self.max_tilt, min(self.max_tilt, -self.pid_vy.update(target_vy - vy, dt)))
        pitch_des = max(-self.max_tilt, min(self.max_tilt,  self.pid_vx.update(target_vx - vx, dt)))
        return roll_des, pitch_des

    def step_att(self, dt: float, roll_des: float, pitch_des: float, yaw_sp: float
                 ) -> tuple[float, float, float]:
        """Tầng att (trung): angle → rate_des. Dùng cho multi-rate ZOH.

        Returns:
            (p_des, q_des, r_des) — giữ ZOH đến lần cập nhật tiếp.
        """
        quat = self.robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        p_des = max(-self.max_rate,     min(self.max_rate,     self.pid_roll_att.update(roll_des  - roll,          dt)))
        q_des = max(-self.max_rate,     min(self.max_rate,     self.pid_pitch_att.update(pitch_des - pitch,         dt)))
        r_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.pid_yaw_att.update(wrap_angle(yaw_sp - yaw),   dt)))
        return p_des, q_des, r_des

    def step_rate(self, dt: float, p_des: float, q_des: float, r_des: float,
                  thrust: float = 0.0) -> dict:
        """Tầng rate (nhanh nhất): rate_des (ZOH) → moment → lực motor.

        Returns:
            dict với key: vx, vy, vz, roll, pitch, yaw, thrust.
        """
        vel_w     = self.robot.data.root_lin_vel_w[0]
        quat      = self.robot.data.root_quat_w[0]
        ang_vel_b = self.robot.data.root_ang_vel_b[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]
        vx, vy, vz = vel_w[0].item(), vel_w[1].item(), vel_w[2].item()
        p, q, r    = ang_vel_b[0].item(), ang_vel_b[1].item(), ang_vel_b[2].item()

        m_roll  = max(-self.max_moment,     min(self.max_moment,     self.pid_roll_rate.update(p_des - p, dt)))
        m_pitch = max(-self.max_moment,     min(self.max_moment,     self.pid_pitch_rate.update(q_des - q, dt)))
        m_yaw   = max(-self.max_yaw_moment, min(self.max_yaw_moment, self.pid_yaw_rate.update(r_des  - r, dt)))

        wrench  = torch.tensor([thrust, m_roll, m_pitch, 0.0], device=self._dev)
        F_props = (self.A_inv @ wrench).clamp(min=0.0)

        apply_prop_wrench(
            robot=self.robot,
            prop_body_ids=self.prop_body_ids,
            thrust_per_prop=F_props,
            prop_spin_dirs=self._zero_spin,
            spin_torque=0.0,
            root_body_ids=self.root_body_ids,
            net_yaw=m_yaw,
        )

        return {"vx": vx, "vy": vy, "vz": vz,
                "roll": roll, "pitch": pitch, "yaw": yaw, "thrust": thrust}

    def reset(self) -> None:
        for c in (self.pid_vx, self.pid_vy,
                  self.pid_roll_att, self.pid_pitch_att, self.pid_yaw_att,
                  self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate):
            c.reset()

class PositionController:
    """4‑tầng cascade: position → velocity → angle → rate → moment.

    Tầng ngoài cùng (position): sai lệch vị trí → vận tốc mong muốn (vx_des, vy_des)
    Tầng trong: sử dụng VelocityController để xử lý vận tốc → angle → rate → moment.

    Phù hợp cho bám mục tiêu (H1) với yêu cầu giữ khoảng cách.

    Args:
        robot: Articulation object.
        prop_body_ids, root_body_ids: body IDs.
        A_inv: allocation matrix inverse.
        hover_thrust: lực nâng hover [N].
        # Position gains (vxy_des)
        pos_kp_x: P gain position → vx_des [1/s]
        pos_ki_x, pos_kd_x, pos_lim_x
        pos_kp_y, pos_ki_y, pos_kd_y, pos_lim_y
        max_vel: giới hạn vận tốc mong muốn [m/s]
        # Velocity gains (vxy → angle) – dùng từ VelocityController, có thể truyền vào
        vxy_kp, vxy_ki, vxy_kd, vxy_lim, max_tilt
        # Attitude gains (angle → rate)
        att_kp, att_ki, att_kd, yaw_att_kp, yaw_att_ki, yaw_att_kd
        # Rate gains (rate → moment)
        rate_kp, rate_ki, rate_kd, rate_lim,
        yaw_rate_kp, yaw_rate_ki, yaw_rate_kd, yaw_rate_lim
        max_rate, max_yaw_rate, max_moment, max_yaw_moment
    """

    def __init__(
        self,
        robot,
        prop_body_ids,
        root_body_ids,
        A_inv: torch.Tensor,
        hover_thrust: float = 0.35,
        # Position gains
        pos_kp_x: float = 0.5, pos_ki_x: float = 0.05, pos_kd_x: float = 0.2, pos_lim_x: float = 0.5,
        pos_kp_y: float = 0.5, pos_ki_y: float = 0.05, pos_kd_y: float = 0.2, pos_lim_y: float = 0.5,
        max_vel: float = 2.0,
        # Velocity gains (dùng để khởi tạo VelocityController bên trong)
        vxy_kp: float = 0.5, vxy_ki: float = 0.05, vxy_kd: float = 0.2, vxy_lim: float = 0.5, max_tilt: float = math.radians(20.0),
        att_kp: float = 2.5, att_ki: float = 0.0, att_kd: float = 0.5,
        yaw_att_kp: float = 1.0, yaw_att_ki: float = 0.0, yaw_att_kd: float = 0.0,
        rate_kp: float = 0.0002, rate_ki: float = 0.00015, rate_kd: float = 0.0000185, rate_lim: float = 1.0,
        yaw_rate_kp: float = 0.00015, yaw_rate_ki: float = 0.0005, yaw_rate_kd: float = 0.00001, yaw_rate_lim: float = 0.2,
        max_rate: float = math.radians(180.0), max_yaw_rate: float = math.radians(90.0),
        max_moment: float = 0.03, max_yaw_moment: float = 0.0003,
    ):
        self.robot = robot
        self.prop_body_ids = prop_body_ids
        self.root_body_ids = root_body_ids
        self.A_inv = A_inv
        self._dev = A_inv.device
        self.hover_thrust = hover_thrust
        self.max_vel = max_vel

        # Position PID (tầng ngoài)
        self.pid_pos_x = PIDController(pos_kp_x, pos_ki_x, pos_kd_x, integral_limit=pos_lim_x)
        self.pid_pos_y = PIDController(pos_kp_y, pos_ki_y, pos_kd_y, integral_limit=pos_lim_y)

        # VelocityController bên trong (xử lý vận tốc → moment)
        self.vel_ctrl = VelocityController(
            robot=robot, prop_body_ids=prop_body_ids, root_body_ids=root_body_ids, A_inv=A_inv,
            vxy_kp=vxy_kp, vxy_ki=vxy_ki, vxy_kd=vxy_kd, vxy_lim=vxy_lim, max_tilt=max_tilt,
            att_kp=att_kp, att_ki=att_ki, att_kd=att_kd,
            yaw_att_kp=yaw_att_kp, yaw_att_ki=yaw_att_ki, yaw_att_kd=yaw_att_kd,
            rate_kp=rate_kp, rate_ki=rate_ki, rate_kd=rate_kd, rate_lim=rate_lim,
            yaw_rate_kp=yaw_rate_kp, yaw_rate_ki=yaw_rate_ki, yaw_rate_kd=yaw_rate_kd, yaw_rate_lim=yaw_rate_lim,
            max_rate=max_rate, max_yaw_rate=max_yaw_rate,
            max_moment=max_moment, max_yaw_moment=max_yaw_moment
        )

        # Lưu các tham số để dùng cho step_rate
        self.max_moment = max_moment
        self.max_yaw_moment = max_yaw_moment
        self._zero_spin = torch.zeros(4, device=self._dev)

    def step(
        self,
        dt: float,
        target_x: float, target_y: float,
        current_x: float, current_y: float,
        thrust: float,
        yaw_sp: float = 0.0,
        dt_pos: float | None = None,
    ) -> dict:
        """Một bước đầy đủ với tần số cao (nên gọi ở tốc độ rate loop)."""
        error_x = target_x - current_x
        error_y = target_y - current_y
        dt_used = dt_pos if dt_pos is not None else dt
        vx_des = self.pid_pos_x.update(error_x, dt_used)
        vy_des = self.pid_pos_y.update(error_y, dt_used)
        vx_des = max(-self.max_vel, min(self.max_vel, vx_des))
        vy_des = max(-self.max_vel, min(self.max_vel, vy_des))

        return self.vel_ctrl.step(dt, target_vx=vx_des, target_vy=vy_des, thrust=thrust, yaw_sp=yaw_sp)

    # Các phương thức multi‑rate ZOH
    def step_pos(self, dt: float, target_x: float, target_y: float,
                 current_x: float, current_y: float) -> tuple[float, float]:
        """Tầng position (chậm): error → vx_des, vy_des. Trả về (vx_des, vy_des)."""
        error_x = target_x - current_x
        error_y = target_y - current_y
        vx_des = self.pid_pos_x.update(error_x, dt)
        vy_des = self.pid_pos_y.update(error_y, dt)
        vx_des = max(-self.max_vel, min(self.max_vel, vx_des))
        vy_des = max(-self.max_vel, min(self.max_vel, vy_des))
        return vx_des, vy_des

    def step_vel(self, dt: float, vx_des: float, vy_des: float) -> tuple[float, float]:
        """Tầng velocity: vx_des, vy_des → (roll_des, pitch_des)."""
        return self.vel_ctrl.step_vel(dt, vx_des, vy_des)

    def step_att(self, dt: float, roll_des: float, pitch_des: float, yaw_sp: float) -> tuple[float, float, float]:
        """Tầng attitude: angle → rate_des. Trả về (p_des, q_des, r_des)."""
        return self.vel_ctrl.step_att(dt, roll_des, pitch_des, yaw_sp)

    def step_rate(self, dt: float, p_des: float, q_des: float, r_des: float,
                  thrust: float) -> dict:
        """Tầng rate: rate_des → moment. Trả về dict state."""
        return self.vel_ctrl.step_rate(dt, p_des, q_des, r_des, thrust)

    def reset(self) -> None:
        self.pid_pos_x.reset()
        self.pid_pos_y.reset()
        self.vel_ctrl.reset()