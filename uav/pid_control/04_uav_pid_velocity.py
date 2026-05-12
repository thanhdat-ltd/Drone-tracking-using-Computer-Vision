# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Velocity PID Control — multi-rate ZOH 4 tầng cascade.

Kiến trúc (mỗi tầng chạy ở tần số riêng):
    Physics (sim): 200 Hz  (dt = 0.005 s)
    Tầng 0 Alt:    50 Hz   (mỗi  4 bước) — z_err → thrust_hold
    Tầng 1 Vel:    50 Hz   (mỗi  4 bước) — vxy → roll_des/pitch_des
    Tầng 2 Att:   100 Hz   (mỗi  2 bước) — angle → rate_des
    Tầng 3 Rate:  200 Hz   (mỗi bước)    — rate → moment

Tất cả tần số là ước của 200 Hz → ZOH chia đều, không lệch bước.

Step test vx/vy được định nghĩa trong Cfg.STEPS.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/04_uav_pid_velocity.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Velocity PID control cho Crazyflie.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext
from isaaclab_assets.uav.uav_cfg import UAV_CFG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import VelocityController, make_front_camera_cfg  # noqa: E402
from pid_controller import PIDController, make_alloc_inv, LivePlotter  # noqa: E402


# ══════════════════════════════════════════════════════════════
#  THAM SỐ
# ══════════════════════════════════════════════════════════════
class Cfg:
    # --- Vị trí ban đầu ---
    INIT_Z = 1.0

    # --- Multi-rate ZOH ---
    # Tất cả phải là ước của SIM_HZ để ZOH chia đều
    SIM_HZ  = 200   # vật lý sim
    ALT_HZ  = 50    # tầng 0: altitude hold   → mỗi 4 bước
    VEL_HZ  = 85    # tầng 1: velocity xy      → mỗi 4 bước
    ATT_HZ  = 100   # tầng 2: attitude          → mỗi 2 bước
    # tầng 3: rate = 200 Hz → mỗi bước (= sim)

    ALT_EVERY = SIM_HZ // ALT_HZ   # = 4
    VEL_EVERY = SIM_HZ // VEL_HZ   # = 4
    ATT_EVERY = SIM_HZ // ATT_HZ   # = 2

    DT_ALT  = 1.0 / ALT_HZ         # = 0.020 s
    DT_VEL  = 1.0 / VEL_HZ         # = 0.020 s
    DT_ATT  = 1.0 / ATT_HZ         # = 0.010 s

    # --- Step test vx/vy [(t_start_s, vx_m/s, vy_m/s)] ---
    STEPS = [
        ( 0.0,  0.0,  0.0),
        ( 2.0,  0.5,  0.0),
        ( 6.0,  0.0,  0.5),
        ( 9.0, -0.5,  0.0),
        (12.0,  0.0, -0.5),
        (15.0,  0.0,  0.0),
    ]

    # --- Tầng 0: Altitude PID (z_err → thrust_hold) ---
    Z_KP   = 2.0;   Z_KI  = 0.8;   Z_KD  = 0.3;   Z_ILIM = 1.0
    Z_MAX_THRUST = 0.8  # clamp để tránh spike thrust khi khởi động (hover = 0.35 N)

    # --- Tầng 1: Velocity xy PID (vxy_err → angle_des) ---
    # KD cao hơn nhiều để tắt dao động ~1 Hz quan sát được
    VXY_KP = 2.5;   VXY_KI = 0.0; VXY_KD = 0.75; VXY_LIM = 0.3
    MAX_TILT = math.radians(20.0)

    # --- Tầng 2: Attitude PID (angle → rate_des) ---
    ATT_KP     = 0.5;   ATT_KI     = 0.0;  ATT_KD     = 0.1
    YAW_ATT_KP = 1.0;   YAW_ATT_KI = 0.0;  YAW_ATT_KD = 0.0

    # --- Tầng 3: Rate PID (rate → moment) — giữ nguyên gains gốc đã hoạt động ---
    RATE_KP     = 0.0002;    RATE_KI     = 0.00015;   RATE_KD     = 0.0000185;  RATE_LIM     = 0.5
    YAW_RATE_KP = 0.00015;   YAW_RATE_KI = 0.0005;    YAW_RATE_KD = 0.00001;    YAW_RATE_LIM = 0.2

    MAX_RATE       = math.radians(180.0)
    MAX_YAW_RATE   = math.radians(90.0)
    MAX_MOMENT     = 0.03
    MAX_YAW_MOMENT = 0.0003

    # --- Plotter ---
    WINDOW_S   = 20.0
    PLOT_EVERY = 5


# ══════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════
def setup_simulation():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / Cfg.SIM_HZ, device=args_cli.device))
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot  = Articulation(robot_cfg)
    camera = Camera(make_front_camera_cfg())

    sim.reset()
    camera.reset()
    return sim, robot, camera


# ══════════════════════════════════════════════════════════════
#  SETPOINT từ step table
# ══════════════════════════════════════════════════════════════
def get_setpoint(sim_time: float) -> tuple[float, float]:
    vx, vy = 0.0, 0.0
    for t_start, svx, svy in Cfg.STEPS:
        if sim_time >= t_start:
            vx, vy = svx, svy
    return vx, vy


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("VELOCITY PID — step test tự động")
    print(f"ZOH: alt@{Cfg.ALT_HZ}Hz (dt={Cfg.DT_ALT:.3f}s)  vel@{Cfg.VEL_HZ}Hz (dt={Cfg.DT_VEL:.3f}s)  att@{Cfg.ATT_HZ}Hz (dt={Cfg.DT_ATT:.4f}s)  rate@{Cfg.SIM_HZ}Hz")
    print("=" * 60)

    sim, robot, camera = setup_simulation()
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv    = make_alloc_inv(device=sim.device)
    dt       = sim.get_physics_dt()

    ctrl = VelocityController(
        robot=robot,
        prop_body_ids=prop_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        vxy_kp=Cfg.VXY_KP,     vxy_ki=Cfg.VXY_KI,     vxy_kd=Cfg.VXY_KD,     vxy_lim=Cfg.VXY_LIM,
        max_tilt=Cfg.MAX_TILT,
        att_kp=Cfg.ATT_KP,         att_ki=Cfg.ATT_KI,         att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP,         rate_ki=Cfg.RATE_KI,         rate_kd=Cfg.RATE_KD,         rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP, yaw_rate_ki=Cfg.YAW_RATE_KI, yaw_rate_kd=Cfg.YAW_RATE_KD, yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE,
        max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT,
        max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    plotter = LivePlotter(
        "Velocity Control — Z / Vx / Vy",
        ["Z [m]", "Vx [m/s]", "Vy [m/s]"],
        Cfg.WINDOW_S, dt,
    )

    sim_time = 0.0
    step     = 0
    log_every = Cfg.SIM_HZ  # log mỗi giây

    # ZOH buffers — giữ nguyên giá trị cho đến khi tầng tương ứng chạy lại
    thrust_hold    = 0.0
    roll_des_hold  = 0.0
    pitch_des_hold = 0.0
    p_des_hold     = 0.0
    q_des_hold     = 0.0
    r_des_hold     = 0.0

    while simulation_app.is_running():
        cmd_vx, cmd_vy = get_setpoint(sim_time)

        # ── Tầng 0: Altitude hold — 25 Hz ──────────────────────
        if step % Cfg.ALT_EVERY == 0:
            z = robot.data.root_pos_w[0][2].item()
            thrust_hold = max(0.0, min(Cfg.Z_MAX_THRUST, pid_z.update(Cfg.INIT_Z - z, Cfg.DT_ALT)))

        # ── Tầng 1: Velocity xy — 50 Hz ────────────────────────
        # step_vel bây giờ chỉ trả về (roll_des, pitch_des), không cần thrust
        if step % Cfg.VEL_EVERY == 0:
            roll_des_hold, pitch_des_hold = ctrl.step_vel(
                Cfg.DT_VEL, cmd_vx, cmd_vy
            )

        # ── Tầng 2: Attitude — 100 Hz ──────────────────────────
        if step % Cfg.ATT_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_att(
                Cfg.DT_ATT, roll_des_hold, pitch_des_hold, yaw_sp=0.0
            )

        # ── Tầng 3: Rate — 200 Hz (mỗi bước) ──────────────────
        data = ctrl.step_rate(
            dt, p_des_hold, q_des_hold, r_des_hold, thrust=thrust_hold
        )

        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step     += 1
        robot.update(dt)
        camera.update(dt)

        # ── Log mỗi giây ───────────────────────────────────────
        if step % log_every == 0:
            z   = robot.data.root_pos_w[0][2].item()
            pos = robot.data.root_pos_w[0]
            print(
                f"t={sim_time:5.1f}s | "
                f"z={z:+.2f}m(→{Cfg.INIT_Z:+.2f}) | "
                f"cmd=({cmd_vx:+.1f},{cmd_vy:+.1f})m/s | "
                f"vel=({data['vx']:+.2f},{data['vy']:+.2f},{data['vz']:+.2f})m/s | "
                f"pos=({pos[0].item():+.2f},{pos[1].item():+.2f},{pos[2].item():+.2f})m"
            )

        # ── Plotter ─────────────────────────────────────────────
        if step % Cfg.PLOT_EVERY == 0:
            z = robot.data.root_pos_w[0][2].item()
            plotter.update(
                sim_time,
                [z,          data["vx"], data["vy"]],
                [Cfg.INIT_Z, cmd_vx,     cmd_vy],
            )

        # ── Follow camera ───────────────────────────────────────
        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 0.5, p[1] - 0.5, p[2] + 0.5],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()