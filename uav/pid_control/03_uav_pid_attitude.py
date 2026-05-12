# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Attitude PID Control — Cascade 2 tầng: angle → rate → moment + altitude PID.

Kiến trúc:
    z_error     → [PID_z]    → thrust       (từ file 01)
    angle_error → [PID_att]  → rate_des     (tầng ngoài)
    rate_error  → [PID_rate] → moment       (tầng trong, từ file 02)

Step test:
    0s  → z=1.0m, level (0°, 0°, 0°)
    5s  → roll +10°
    12s → roll -10°, pitch +5°
    19s → level + yaw +20°  (ramp dần)
    26s → về 0°

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/03_uav_pid_attitude.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attitude cascade 2 tầng + altitude PID.")
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import AttitudeController, make_front_camera_cfg  # noqa: E402
from pid_controller import PIDController, LivePlotter, make_alloc_inv, wrap_angle  # noqa: E402

from isaaclab_assets.uav.uav_cfg import UAV_CFG


# === THAM SỐ ===
class Cfg:
    TARGET_Z = 1.0   # [m]

    # Step test [(t_start_s, roll_deg, pitch_deg, yaw_deg)]
    STEPS = [
        ( 0.0,   0.0,  0.0,   0.0),
        ( 2.0,  10.0,  10.0,  0.0),
        (5.0, -10.0,  5.0,   0.0),
        (9.0,   0.0,  0.0,  20.0),
        (12.0,   0.0,  0.0,   0.0),
    ]

    # Altitude PID (từ file 01)
    Z_KP = 2.5;  Z_KI = 0.85;  Z_KD = 0.52;  Z_ILIM = 1.0

    # Tầng ngoài — angle → rate_des  [rad/s per rad]
    # Outer bandwidth phải << inner bandwidth (~30-40 rad/s)
    # ATT_KP=2.0 → bandwidth ~2 rad/s → đủ margin (~15-20x slower than inner)
    ATT_KP     = 5.5;  ATT_KI     = 0.55;  ATT_KD     = 1.2

    YAW_ATT_KP = 1.0;  YAW_ATT_KI = 0.0;  YAW_ATT_KD = 0.0

    # Tầng trong — rate → moment  (từ file 02)
    RATE_KP     = 0.0002; RATE_KI     = 0.00015;  RATE_KD     = 0.0000185; RATE_LIM     = 0.5 * 2
    YAW_RATE_KP = 0.00015; YAW_RATE_KI = 0.0005; YAW_RATE_KD = 0.00001;    YAW_RATE_LIM = 0.1 * 2

    # Multi-rate: outer chạy 50 Hz, inner chạy 200 Hz (physics rate)
    SIM_HZ   = 200
    OUTER_HZ = 50                       # outer loop tần số [Hz]
    OUTER_EVERY = SIM_HZ // OUTER_HZ   # = 4 steps

    MAX_RATE     = math.radians(180.0)
    MAX_YAW_RATE = math.radians(90.0)
    YAW_RATE_LIMIT = math.radians(30.0)

    MAX_MOMENT     = 0.03
    MAX_YAW_MOMENT = 0.0003

    WINDOW_S   = 30.0
    PLOT_EVERY = 5


# === SETUP ===
def setup_simulation():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
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


def get_setpoint(sim_time: float) -> tuple[float, float, float]:
    roll_d = pitch_d = yaw_d = 0.0
    for t_start, r, p, y in Cfg.STEPS:
        if sim_time >= t_start:
            roll_d, pitch_d, yaw_d = r, p, y
    return math.radians(roll_d), math.radians(pitch_d), math.radians(yaw_d)


# === MAIN ===
def main():
    print("=" * 60)
    print("ATTITUDE CASCADE 2 tầng + ALTITUDE PID")
    print("=" * 60)

    sim, robot, camera = setup_simulation()
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv    = make_alloc_inv(device=sim.device)
    dt       = sim.get_physics_dt()

    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    ctrl = AttitudeController(
        robot=robot,
        prop_body_ids=prop_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        att_kp=Cfg.ATT_KP,         att_ki=Cfg.ATT_KI,         att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP,           rate_ki=Cfg.RATE_KI,           rate_kd=Cfg.RATE_KD,           rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP,   yaw_rate_ki=Cfg.YAW_RATE_KI,   yaw_rate_kd=Cfg.YAW_RATE_KD,   yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE,
        max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT,
        max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    plotter = LivePlotter(
        "Cascade 2 tầng + Altitude — Z / Roll / Pitch / Yaw",
        ["Z [m]", "Roll [°]", "Pitch [°]", "Yaw [°]"],
        Cfg.WINDOW_S, dt,
    )

    sim_time   = 0.0
    step       = 0
    log_hz     = int(1.0 / dt)
    yaw_sp     = 0.0
    dt_outer   = 1.0 / Cfg.OUTER_HZ
    # ZOH buffers — giữ rate_des giữa các lần outer cập nhật
    p_des_hold = q_des_hold = r_des_hold = 0.0

    while simulation_app.is_running():
        roll_d, pitch_d, yaw_d = get_setpoint(sim_time)

        yaw_err_sp = wrap_angle(yaw_d - yaw_sp)
        yaw_sp    += max(-Cfg.YAW_RATE_LIMIT * dt,
                         min( Cfg.YAW_RATE_LIMIT * dt, yaw_err_sp))

        pos = robot.data.root_pos_w[0]
        z   = pos[2].item()
        thrust = max(0.0, pid_z.update(Cfg.TARGET_Z - z, dt))

        # Tầng ngoài: chỉ chạy mỗi OUTER_EVERY bước (ZOH)
        if step % Cfg.OUTER_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_outer(
                dt_outer, roll_d, pitch_d, yaw_sp
            )

        # Tầng trong: chạy mỗi bước với rate_des được ZOH giữ
        data = ctrl.step_inner(dt, p_des_hold, q_des_hold, r_des_hold, thrust=thrust)

        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step     += 1
        robot.update(dt)
        camera.update(dt)

        if step % log_hz == 0:
            print(
                f"t={sim_time:5.1f}s | z={z:+.3f}m | "
                f"roll={math.degrees(data['roll']):+6.1f}°(→{math.degrees(roll_d):+5.1f}°) | "
                f"pitch={math.degrees(data['pitch']):+6.1f}°(→{math.degrees(pitch_d):+5.1f}°) | "
                f"yaw={math.degrees(data['yaw']):+6.1f}°(→{math.degrees(yaw_d):+5.1f}°)"
            )

        if step % Cfg.PLOT_EVERY == 0:
            plotter.update(
                sim_time,
                [z, math.degrees(data["roll"]),  math.degrees(data["pitch"]),  math.degrees(data["yaw"])],
                [Cfg.TARGET_Z, math.degrees(roll_d), math.degrees(pitch_d), math.degrees(yaw_d)],
            )

        p = pos.cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 1.5, p[1] - 1.5, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
