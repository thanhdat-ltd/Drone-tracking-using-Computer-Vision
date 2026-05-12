# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PID Controller cho Crazyflie quadcopter.

Cấu trúc:
  PIDController   — single-axis PID (base class)
  QuadcopterPID   — outer loops: position error → thrust (N) + desired attitude (rad)
                    Không dùng feedforward trọng lực — integral tự bù theo thời gian.
                    Không cần biết robot_mass hay gravity.
"""

from __future__ import annotations

import math
import torch
from collections import deque

# ── Thông số Crazyflie 2.x ────────────────────────────────────────────────────
_ARM_M = 0.046                      # chiều dài cánh [m]
_D     = _ARM_M / math.sqrt(2)      # khoảng cách vuông góc tâm → prop ≈ 0.0325 m
_KM    = 0.005                      # tỉ lệ drag-torque / thrust [m] (đo từ thực nghiệm)


def make_alloc_inv(device: str = "cpu") -> torch.Tensor:
    """Tính ma trận phân bổ nghịch đảo cho Crazyflie cf2x.

    Chuyển đổi wrench [Fz, Tx, Ty, Tz] → lực từng prop [F1, F2, F3, F4] (Newton).

    Sơ đồ bố trí motor (nhìn từ trên xuống):

        m1(CCW)  m2(CW)       ← phía trước
            \\      /
             [body]
            /      \\
        m4(CW)  m3(CCW)       ← phía sau

    Ma trận phân bổ A (4×4):
        [Fz]   [ 1    1    1    1  ] [F1]
        [Tx] = [ D   -D   -D    D  ] [F2]
        [Ty]   [-D   -D    D    D  ] [F3]
        [Tz]   [-KM  KM  -KM   KM ] [F4]

    Trong đó:
        Fz  = tổng lực nâng (N)
        Tx  = moment roll  (Nm) — quay quanh trục X (tiến/lùi)
        Ty  = moment pitch (Nm) — quay quanh trục Y (trái/phải)
        Tz  = moment yaw   (Nm) — quay quanh trục Z, tạo bởi drag khác chiều
        D   = khoảng cách tâm → prop theo phương vuông góc
        KM  = drag-torque / thrust ratio

    Returns:
        A_inv: tensor (4, 4) trên device chỉ định.
    """
    A = torch.tensor([
        [ 1.0,   1.0,   1.0,   1.0],   # Fz  = F1+F2+F3+F4
        [ _D,   -_D,   -_D,    _D ],   # Tx  = D*(F1-F2-F3+F4)
        [-_D,   -_D,    _D,    _D ],   # Ty  = D*(-F1-F2+F3+F4)
        [-_KM,  _KM,  -_KM,   _KM],   # Tz  = KM*(-F1+F2-F3+F4)
    ], device=device)
    return torch.linalg.inv(A)


class PIDController:
    """Single-axis PID controller.

    Args:
        kp: Hệ số tỉ lệ.
        ki: Hệ số tích phân.
        kd: Hệ số vi phân.
        integral_limit: Giới hạn chống wind-up. None = không giới hạn.
        derivative_limit: Giới hạn đạo hàm để tránh spike khi error đột biến.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = 10.0,
        derivative_limit: float = 10.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit  = integral_limit
        self.derivative_limit = derivative_limit

        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """Tính output PID.

        Args:
            error: Sai số hiện tại (setpoint − actual).
            dt:    Bước thời gian [s]. Nếu dt=0 thì bỏ qua vi phân.

        Returns:
            Giá trị điều khiển (chưa clamp ở output).
        """
        # Tích phân
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(
                -self.integral_limit,
                min(self.integral_limit, self._integral),
            )

        # Vi phân — khởi tạo = 0 để tránh UnboundLocalError khi dt=0
        derivative = 0.0
        if dt > 0.0:
            derivative = (error - self._prev_error) / dt
            if self.derivative_limit:
                derivative = max(
                    -self.derivative_limit,
                    min(self.derivative_limit, derivative),
                )

        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def apply_wrench(
    robot,
    sim,
    prop_body_ids,
    A_inv: "torch.Tensor",
    thrust: float,
    m_roll: float,
    m_pitch: float,
    m_yaw: float,
) -> None:
    """Phân bổ wrench [Fz, Tx, Ty, Tz] → lực motor và áp dụng lên robot."""
    wrench  = torch.tensor([thrust, m_roll, m_pitch, m_yaw], device=sim.device)
    F_props = (A_inv @ wrench).clamp(min=0.0)
    forces  = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
    forces[0, :, 2] = F_props
    robot.set_external_force_and_torque(
        forces=forces,
        torques=torch.zeros_like(forces),
        body_ids=prop_body_ids,
    )


def reset_robot(robot) -> None:
    """Reset robot về trạng thái mặc định."""
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
    robot.reset()


class LivePlotter:
    """Đồ thị thời gian thực n-axis (actual vs target).

    Dùng chung cho tất cả các script PID.
    """

    def __init__(self, title: str, ylabels: list, window_s: float, dt: float):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        n = len(ylabels)
        self.window_s = window_s
        maxlen = int(window_s / dt) + 50
        self.times  = deque(maxlen=maxlen)
        self.actual = [deque(maxlen=maxlen) for _ in range(n)]
        self.target = [deque(maxlen=maxlen) for _ in range(n)]

        plt.ion()
        self.fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        self.axes = [axes] if n == 1 else list(axes)
        self.lines_actual, self.lines_target = [], []

        for ax, ylabel in zip(self.axes, ylabels):
            la, = ax.plot([], [], "b-",  lw=2,   label="actual")
            lt, = ax.plot([], [], "r--", lw=1.5, label="target")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)
            self.lines_actual.append(la)
            self.lines_target.append(lt)

        self.axes[-1].set_xlabel("Time [s]")
        self.fig.suptitle(title, fontsize=13, fontweight="bold")
        self.fig.tight_layout()

    def update(self, t: float, actual_vals: list, target_vals: list) -> None:
        self.times.append(t)
        for i, (a, tg) in enumerate(zip(actual_vals, target_vals)):
            self.actual[i].append(a)
            self.target[i].append(tg)

        ts = list(self.times)
        if not ts:
            return

        for i, (la, lt) in enumerate(zip(self.lines_actual, self.lines_target)):
            a  = list(self.actual[i])
            tg = list(self.target[i])
            la.set_data(ts, a)
            lt.set_data(ts, tg)
            all_v = a + tg
            if all_v:
                span   = max(all_v) - min(all_v)
                margin = max(0.05, span * 0.1 + 0.05)
                self.axes[i].set_ylim(min(all_v) - margin, max(all_v) + margin)

        lo = max(0.0, ts[-1] - self.window_s)
        for ax in self.axes:
            ax.set_xlim(lo, ts[-1] + 0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()