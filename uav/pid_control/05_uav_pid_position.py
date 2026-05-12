# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mô phỏng Position Control cho quadcopter dùng PID cascade 4 tầng.

Kiến trúc multi-rate ZOH:
    Position (10 Hz) → velocity (50 Hz) → attitude (100 Hz) → rate (200 Hz)
    Altitude hold riêng (25 Hz) dùng PID z.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/07_uav_pid_position.py
"""

import argparse
import math
import collections

import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Position PID control cho Crazyflie.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.uav.uav_cfg import UAV_CFG
from pid_controller import PIDController, make_alloc_inv
from controller import PositionController  # class đã fix

WINDOW_S   = 20.0
PLOT_EVERY = 5

# ──────────────────────────────────────────────────────────────────────────────
# Tham số điều khiển
# ──────────────────────────────────────────────────────────────────────────────
class Cfg:
    SIM_HZ = 200
    DT = 1.0 / SIM_HZ

    # Multi-rate ZOH (các tần số là ước của SIM_HZ)
    POS_HZ = 10      # vị trí
    VEL_HZ = 50      # vận tốc
    ATT_HZ = 100     # attitude
    # rate chạy ở SIM_HZ

    POS_EVERY = SIM_HZ // POS_HZ   # 20
    VEL_EVERY = SIM_HZ // VEL_HZ   # 4
    ATT_EVERY = SIM_HZ // ATT_HZ   # 2

    DT_POS = 1.0 / POS_HZ
    DT_VEL = 1.0 / VEL_HZ
    DT_ATT = 1.0 / ATT_HZ

    # Altitude PID (z → thrust)
    Z_KP = 2.0
    Z_KI = 0.8
    Z_KD = 0.3
    Z_ILIM = 1.0
    Z_MAX_THRUST = 0.8

    # Position PID (xy → vx_des, vy_des)
    POS_KP = 0.8
    POS_KI = 0.1
    POS_KD = 0.3
    POS_LIM = 1.0
    MAX_VEL = 2.0

    # Velocity PID (vxy → angle) — đồng bộ gains từ 04 đã tune
    VXY_KP = 2.5
    VXY_KI = 0.0
    VXY_KD = 0.75
    VXY_LIM = 0.3
    MAX_TILT = math.radians(20.0)

    # Attitude PID (angle → rate)
    ATT_KP = 0.5
    ATT_KI = 0.0
    ATT_KD = 0.1
    YAW_ATT_KP = 1.0
    YAW_ATT_KI = 0.0
    YAW_ATT_KD = 0.0

    # Rate PID (rate → moment)
    RATE_KP = 0.0002
    RATE_KI = 0.00015
    RATE_KD = 0.0000185
    RATE_LIM = 0.5
    YAW_RATE_KP = 0.00015
    YAW_RATE_KI = 0.0005
    YAW_RATE_KD = 0.00001
    YAW_RATE_LIM = 0.2

    MAX_RATE = math.radians(180.0)
    MAX_YAW_RATE = math.radians(90.0)
    MAX_MOMENT = 0.03
    MAX_YAW_MOMENT = 0.0003

    TARGET_POS = (2.0, 2.0, 1.0)  # x, y, z


def make_plot():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].set_title("UAV Position Response — X/Y/Z(t)")
    for ax, lbl in zip(axes, ["X [m]", "Y [m]", "Z [m]"]):
        ax.set_ylabel(lbl)
        ax.grid(True)
    axes[2].set_xlabel("Time [s]")
    lines_pos, lines_tgt = [], []
    for ax in axes:
        lp, = ax.plot([], [], "b-",  lw=2,   label="pos")
        lt, = ax.plot([], [], "r--", lw=1.5, label="target")
        ax.legend(loc="upper right", fontsize=8)
        lines_pos.append(lp)
        lines_tgt.append(lt)
    fig.tight_layout()
    return fig, axes, lines_pos, lines_tgt


def update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data):
    t = list(times)
    if not t:
        return
    for i, (lp, lt) in enumerate(zip(lines_pos, lines_tgt)):
        p  = list(pos_data[i])
        tv = list(tgt_data[i])
        lp.set_data(t, p)
        lt.set_data(t, tv)
        all_v = p + tv
        margin = 0.1
        axes[i].set_xlim(max(0.0, t[-1] - WINDOW_S), t[-1] + 0.5)
        axes[i].set_ylim(min(all_v) - margin, max(all_v) + margin)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=Cfg.DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv = make_alloc_inv(device=sim.device)

    # PositionController đã fix (multi-rate ZOH)
    ctrl = PositionController(
        robot=robot,
        prop_body_ids=prop_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        hover_thrust=0.35,
        pos_kp_x=Cfg.POS_KP, pos_ki_x=Cfg.POS_KI, pos_kd_x=Cfg.POS_KD, pos_lim_x=Cfg.POS_LIM,
        pos_kp_y=Cfg.POS_KP, pos_ki_y=Cfg.POS_KI, pos_kd_y=Cfg.POS_KD, pos_lim_y=Cfg.POS_LIM,
        max_vel=Cfg.MAX_VEL,
        vxy_kp=Cfg.VXY_KP, vxy_ki=Cfg.VXY_KI, vxy_kd=Cfg.VXY_KD, vxy_lim=Cfg.VXY_LIM, max_tilt=Cfg.MAX_TILT,
        att_kp=Cfg.ATT_KP, att_ki=Cfg.ATT_KI, att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP, rate_ki=Cfg.RATE_KI, rate_kd=Cfg.RATE_KD, rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP, yaw_rate_ki=Cfg.YAW_RATE_KI, yaw_rate_kd=Cfg.YAW_RATE_KD, yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE, max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT, max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    # Altitude PID riêng (z → thrust)
    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    # Live plot
    maxlen = int(WINDOW_S / Cfg.DT) + 50
    times = collections.deque(maxlen=maxlen)
    pos_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    tgt_data = [collections.deque(maxlen=maxlen) for _ in range(3)]
    fig, axes, lines_pos, lines_tgt = make_plot()

    sim_time = 0.0
    step = 0

    # ZOH buffers
    thrust_hold = 0.0
    vx_des_hold = 0.0
    vy_des_hold = 0.0
    roll_des_hold = 0.0
    pitch_des_hold = 0.0
    p_des_hold = 0.0
    q_des_hold = 0.0
    r_des_hold = 0.0

    target_x, target_y, target_z = Cfg.TARGET_POS
    target_pos = torch.tensor([target_x, target_y, target_z], device=sim.device)

    print(f"[INFO] Target position: ({target_x}, {target_y}, {target_z}) m")

    # Altitude loop frequency: 25 Hz
    ALT_EVERY = Cfg.SIM_HZ // 25  # = 8

    while simulation_app.is_running():
        # Altitude hold (z → thrust) chạy ở 25 Hz
        if step % ALT_EVERY == 0:
            z = robot.data.root_pos_w[0][2].item()
            thrust_hold = max(0.0, pid_z.update(target_z - z, 1.0 / 25.0))

        # Position loop (10 Hz)
        if step % Cfg.POS_EVERY == 0:
            vx_des_hold, vy_des_hold = ctrl.step_pos(Cfg.DT_POS, target_x, target_y)

        # Velocity loop (50 Hz)
        if step % Cfg.VEL_EVERY == 0:
            roll_des_hold, pitch_des_hold = ctrl.step_vel(Cfg.DT_VEL, vx_des_hold, vy_des_hold)

        # Attitude loop (100 Hz)
        if step % Cfg.ATT_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_att(Cfg.DT_ATT, roll_des_hold, pitch_des_hold, yaw_sp=0.0)

        # Rate loop (200 Hz, mỗi bước)
        data = ctrl.step_rate(Cfg.DT, p_des_hold, q_des_hold, r_des_hold, thrust=thrust_hold)

        robot.write_data_to_sim()
        sim.step()
        sim_time += Cfg.DT
        step += 1
        robot.update(Cfg.DT)

        # Log mỗi giây
        if step % Cfg.SIM_HZ == 0:
            pos = robot.data.root_pos_w[0]
            err = (target_pos - pos).norm().item()
            print(f"t={sim_time:5.1f}s | pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f}) | err={err:.3f}m | thrust={thrust_hold:.4f}N")

        # Plot
        times.append(sim_time)
        pos = robot.data.root_pos_w[0]
        for i in range(3):
            pos_data[i].append(pos[i].item())
            tgt_data[i].append(target_pos[i].item())
        if step % PLOT_EVERY == 0:
            update_plot(fig, axes, lines_pos, lines_tgt, times, pos_data, tgt_data)

        # Camera follow
        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 0.8, p[1] - 0.8, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()